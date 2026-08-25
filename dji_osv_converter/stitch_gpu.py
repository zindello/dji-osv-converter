"""GPU stitch - the whole pipeline resident on the device (PyTorch).

``stitcher.py`` is the CPU reference; this runs warp -> exposure -> linear
blend -> levelling entirely on the GPU, so the only CPU<->GPU transfers are one
fisheye pair up and one panorama down per frame. ``grid_sample`` is the remap
(warp and the levelling rotation); everything else is elementwise / conv tensor
ops. Output matches the CPU path to bilinear/rounding.

Backend-agnostic: uses CUDA (NVIDIA) when present, else Apple MPS (Metal), else
falls back to CPU tensors. The stitch math is identical on all three - only the
device changes. Construct only when ``available()`` is true.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:                       # torch not installed
    torch = None
    F = None

from .calibration import CameraMap
from .stitcher import seam_weights
from .orientation import directions


def pick_device(prefer: str | None = None) -> str | None:
    """Best available torch device name: 'cuda' > 'mps' > None (torch absent)."""
    if torch is None:
        return None
    if prefer and prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def available(prefer: str | None = None) -> bool:
    dev = pick_device(prefer)
    return dev is not None and dev != "cpu"


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


class _PinnedSlot:
    """One pinned staging buffer for a p010 fisheye (Y + interleaved UV).

    ``Y``/``UV`` are numpy views the decode worker fills; ``event`` fences
    reuse (recorded after the slot's H2D copy is queued, waited by alloc).
    """

    def __init__(self, fh: int, fw: int):
        self.tY = torch.empty((fh, fw), dtype=torch.uint16).pin_memory()
        self.tUV = torch.empty((fh // 2, fw), dtype=torch.uint16).pin_memory()
        self.Y, self.UV = self.tY.numpy(), self.tUV.numpy()
        self.event = None


class PinnedInputPool:
    """Round-robin ring of pinned fisheye slots handed to the decode workers.

    The staging copy (decoded plane -> pinned) runs on the decode threads,
    which have headroom; the stitch thread then uploads with a true async
    copy on a side stream instead of a ~9 ms/plane pageable transfer. The
    ring is sized past the deepest queue chain so alloc() rarely waits.
    """

    def __init__(self, fh: int, fw: int, slots: int | None = None):
        from .constants import PINNED_INPUT_SLOTS
        slots = PINNED_INPUT_SLOTS if slots is None else slots
        import threading
        self._slots = [_PinnedSlot(fh, fw) for _ in range(slots)]
        self._i = 0
        self._lock = threading.Lock()

    def alloc(self) -> _PinnedSlot:
        with self._lock:
            slot = self._slots[self._i]
            self._i = (self._i + 1) % len(self._slots)
        if slot.event is not None:
            slot.event.synchronize()      # previous upload finished; reusable
            slot.event = None
        return slot


class _AsyncPlanes:
    """In-flight stitched frame: its D2H copy is running on a side stream.

    ``planes()`` blocks until the copy completes and returns (Y, UV) uint16
    arrays viewing pinned memory — valid until ``release()``, which returns
    the slot to the stitcher's pool. Holding the device tensors until the
    event has synchronized is what makes freeing them safe.
    """

    def __init__(self, pool, slot, event, dev_refs):
        self._pool, self._slot, self._event = pool, slot, event
        self._dev_refs = dev_refs

    def planes(self):
        self._event.synchronize()
        self._dev_refs = None
        return self._slot[0].numpy(), self._slot[1].numpy()

    def release(self):
        self._pool.put(self._slot)
        self._slot = None


class GpuStitcher:
    """GPU-resident remap grids, masks and weights for one OSV geometry.

    ``device`` may be "cuda", "mps", "cpu", or None/"auto" to pick the best.
    """

    def __init__(self, cams: list[CameraMap], band_px: float,
                 width: int, height: int, device: str | None = None,
                 lsc_gain: list[np.ndarray] | None = None,
                 rs_tables=None, fisheye_size: tuple[int, int] = (3840, 3840),
                 input_fmt: str = "yuv420p10le"):
        if torch is None:
            raise RuntimeError("torch is not installed")
        dev = pick_device(device)
        self.dev = torch.device(dev)
        self.W, self.H = width, height
        if input_fmt not in ("yuv420p10le", "p010le", "cuda"):
            raise ValueError(f"unsupported input_fmt {input_fmt!r}")
        # p010 input (NVDEC's native layout): samples top-aligned (value <<
        # 6, exact to undo in fp32) with interleaved chroma rows; _split_yuv
        # unpacks on-device, which is free next to a CPU swscale. "cuda" is
        # the same layout already resident on the GPU (decode-to-device):
        # (Y, UV) torch uint16 tensors, no upload at all.
        self.input_p010 = input_fmt == "p010le"
        self.input_cuda = input_fmt == "cuda"
        self._down_stream = None      # lazy: see _ensure_async / stitch_yuv_async
        self._out_pool = None
        self._up_stream = None        # lazy: side stream for pinned H2D

        # warp grids: map_x/map_y (pixel coords in the source fisheye) -> the
        # normalised [-1, 1] sampling grid grid_sample expects, uploaded once.
        self.grid = []
        self.valid = []
        self.fw, self.fh = fisheye_size          # from the OSV calibration
        for c in cams:
            gx = torch.from_numpy(2.0 * c.map_x / (self.fw - 1) - 1.0)
            gy = torch.from_numpy(2.0 * c.map_y / (self.fh - 1) - 1.0)
            self.grid.append(torch.stack([gx, gy], -1).unsqueeze(0).to(self.dev))
            v = torch.from_numpy(c.valid.astype(np.float32))[None, None].to(self.dev)
            self.valid.append(v)

        w = seam_weights(cams, band_px).astype(np.float32)           # (H, W) in [0,1]
        self.w3 = torch.from_numpy(w)[None, None].to(self.dev)       # (1,1,H,W)
        # static overlap mask, pooled to 1/8 (exposure works at low res)
        self.ov8 = F.avg_pool2d(self.valid[0] * self.valid[1], 8)

        # --- YUV-native (4:2:0) statics ---
        # Chroma runs at half res, so it needs the warp grid and the seam weight
        # at half res. DECIMATE the grid (stride-2 slice), never interpolate it:
        # the levelling/warp grids carry a +-1 longitude-wrap discontinuity, and
        # averaging across it corrupts the wrap meridian.
        self.grid_half = [g[:, ::2, ::2, :].contiguous() for g in self.grid]
        self.w3h = F.avg_pool2d(self.w3, 2)                          # (1,1,H/2,W/2)
        # BT.709 limited-range centred YUV<->RGB matrices; used only by the opt-in
        # per-channel white-balance exposure path (_expose_wb).
        self._Minv = torch.tensor(                                   # centred YUV -> RGB
            [[1.164384, 0.0, 1.792741],
             [1.164384, -0.213249, -0.532909],
             [1.164384, 2.112402, 0.0]], dtype=torch.float32, device=self.dev)
        self._M = torch.tensor(                                      # RGB -> centred YUV
            [[0.182586, 0.614231, 0.062007],
             [-0.100644, -0.338572, 0.439216],
             [0.439216, -0.398942, -0.040274]], dtype=torch.float32, device=self.dev)

        # The level-grid build was 48% of the whole stitch (91 ms at 8K on MPS):
        # 15 separate elementwise kernels for the rotation plus a no-op
        # remainder (atan2 already returns [-pi, pi]) and a 16 ms stack.
        # torch.compile fuses the computation; the equirect direction field is
        # SEPARABLE - d = [cos(lat)sin(lon), sin(lat), -cos(lat)cos(lon)] - so
        # the fused kernel reads four 1-D sin/cos tables (~90 KB) instead of
        # three precomputed (H*W,) components (354 MB): 13.5 -> 5.7 ms at 8K.
        # Scalars are passed as floats so a changing rotation never retriggers
        # compilation.
        # Precision: recomputing the products in f32 in-kernel (vs the f64
        # precompute) shifts gx by at most 7e-5 - only where the SAMPLED
        # pano's pole (the lens optical axis) lands, where atan2 amplifies the
        # ~1-ulp ray error as the sampled cos(lat) -> 0. There, adjacent
        # sample columns are near-duplicates of the same physical direction,
        # so old and new both sample valid values. Measured over spread
        # frames: <= 0.0004% of output pixels differ by >= 1 10-bit LSB
        # (isolated singles; calm frames bit-exact), chroma bit-identical.
        # The old wrap-meridian note still applies (gx = +1 vs -1: same
        # longitude - 1 px in 59M).
        lon = (np.arange(width, dtype=np.float64) / width - 0.5) * 2 * np.pi
        lat = (0.5 - np.arange(height, dtype=np.float64) / height) * np.pi
        as32 = lambda a: torch.from_numpy(a.astype(np.float32)).to(self.dev)
        sl, cl = as32(np.sin(lon)), as32(np.cos(lon))                # (W,)
        sa, ca = as32(np.sin(lat))[:, None], as32(np.cos(lat))[:, None]  # (H,1)

        def _grid_math(r00, r01, r02, r10, r11, r12, r20, r21, r22):
            dx = ca * sl
            dy = sa
            dz = -(ca * cl)
            nx = dx * r00 + dy * r01 + dz * r02
            ny = dx * r10 + dy * r11 + dz * r12
            nz = dx * r20 + dy * r21 + dz * r22
            gx = torch.atan2(nx, -nz) * (1.0 / np.pi)
            gy = torch.asin(ny.clamp(-1, 1)) * (-2.0 / np.pi)
            return torch.stack([gx, gy], -1)

        self._grid_fused = None
        self._d3 = None          # (3, H*W) rays, built lazily for the fallback
        try:
            fn = torch.compile(_grid_math)
            fn(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   # compile now
            self._grid_fused = fn
        except Exception:                    # inductor unavailable: matmul path
            pass

        # p010 pack: fuse the clamp/scale/permute float math (the uint16 cast
        # can't be compiled - inductor's MPS backend lacks the dtype - so it
        # stays an eager kernel). 20 -> 10 ms at 8K, bit-identical.
        def _pack_math(Y, UV):
            Yp = Y[0, 0].clamp(64, 940) * 64.0 + 0.5
            UVp = (UV[0].permute(1, 2, 0).reshape(UV.shape[2], -1)
                   .clamp(64, 960) * 64.0 + 0.5)
            return Yp, UVp

        self._pack_fused = None
        try:
            pf = torch.compile(_pack_math)
            pf(torch.zeros(1, 1, height, width, device=self.dev),
               torch.zeros(1, 2, height // 2, width // 2, device=self.dev))
            self._pack_fused = pf
        except Exception:
            self._pack_fused = _pack_math    # eager, same math
        # Direct float->uint16 cast halves the device->host transfer vs the old
        # int32 route (37 -> 21 ms measured) and is bit-identical. Probe it:
        # torch's uint16 is "limited support" and not every backend casts.
        try:
            torch.zeros(2, device=self.dev).to(torch.uint16).cpu()
            self._u16 = True
        except Exception:
            self._u16 = False

        # Factory lens-shading (devignette) gain per lens, in equirect space -
        # multiplied into each warped lens before the blend to lift the vignetted
        # rim at the seam. Full res for Y, pooled to half for chroma. None = off.
        self.lsc_Y = self.lsc_UV = None
        if lsc_gain is not None:
            self.lsc_Y = [torch.from_numpy(np.ascontiguousarray(g))[None, None].to(self.dev)
                          for g in lsc_gain]
            self.lsc_UV = [F.avg_pool2d(g, 2) for g in self.lsc_Y]

        # Rolling-shutter statics (pipeline.Stitcher360.rs_tables): per lens the
        # source-row time fraction tau (h,w) and the sensor-shift Jacobian
        # J (h,w,2,3) px/rad, at reduced res. None = RS correction off.
        self.rs_tau = self.rs_J = None
        if rs_tables is not None:
            self.rs_tau = [torch.from_numpy(np.ascontiguousarray(t)).to(self.dev)
                           for t, _ in rs_tables]
            self.rs_J = [torch.from_numpy(np.ascontiguousarray(j)).to(self.dev)
                         for _, j in rs_tables]

    def _level_grid(self, R: np.ndarray) -> "torch.Tensor":
        """Full-res sampling grid (1,H,W,2) for rotating the equirect by R.

        The rotated ray is d' = R @ dirs; gx = atan2/pi (already in [-1,1] - no
        wrap remainder needed), gy = -2*asin/pi. Fused via torch.compile when
        available, else one (3,3)@(3,N) matmul + trig.
        """
        r = [float(v) for v in np.asarray(R, dtype=np.float64).flatten()]
        if self._grid_fused is not None:
            return self._grid_fused(*r).view(1, self.H, self.W, 2)
        if self._d3 is None:
            d = directions(self.H, self.W).reshape(-1, 3).astype(np.float32)
            self._d3 = torch.from_numpy(np.ascontiguousarray(d.T)).to(self.dev)
        Rt = torch.tensor(np.asarray(R, np.float32), device=self.dev)
        n = Rt @ self._d3
        gx = torch.atan2(n[0], -n[2]) * (1.0 / np.pi)
        gy = torch.asin(n[1].clamp(-1, 1)) * (-2.0 / np.pi)
        return torch.stack([gx, gy], -1).view(1, self.H, self.W, 2)

    def level(self, pano, R: np.ndarray):
        """Rotate the equirect by R via grid_sample, with longitude wrap."""
        grid = self._level_grid(R)
        return F.grid_sample(pano, grid, mode="bilinear",
                             align_corners=True, padding_mode="border")

    # ---- 10-bit YUV-native path (BT.709 limited range) ----------------------
    #   in : yuv420p10le packed uint16 (H*3/2, W); out: p010le planes (Y, UV).
    #   The whole stitch stays in 4:2:0 YUV - no RGB round-trip. warp, blend and
    #   levelling are linear/resampling ops that work directly on the Y and
    #   half-res UV planes, so the ~180ms/frame of YUV<->RGB conversion is gone
    #   and chroma is never up- then down-sampled (sharper). Luma Y is warped at
    #   full res, chroma UV at half res on the decimated grids.

    def _split_yuv(self, packed):
        fh, fw = self.fh, self.fw
        if self.input_cuda:
            # (Y, UV) uint16 tensors already on-device (decode-to-device);
            # same p010 unpack as below, minus any transfer.
            Yd, UVd = packed
            cur = torch.cuda.current_stream()
            Yd.record_stream(cur)
            UVd.record_stream(cur)
            Y = (Yd.float() * (1.0 / 64.0)).view(1, 1, fh, fw)
            uv = UVd.float() * (1.0 / 64.0)
            U = uv[:, 0::2].unsqueeze(0).unsqueeze(0)
            V = uv[:, 1::2].unsqueeze(0).unsqueeze(0)
            return Y, U, V
        if self.input_p010:
            if isinstance(packed, _PinnedSlot):
                # Pinned staging (filled by the decode worker): the H2D runs
                # non-blocking on a side stream and overlaps the previous
                # frame's kernels; the slot's event fences its reuse.
                if self._up_stream is None:
                    self._up_stream = torch.cuda.Stream()
                cur = torch.cuda.current_stream()
                with torch.cuda.stream(self._up_stream):
                    Yd = packed.tY.to(self.dev, non_blocking=True)
                    UVd = packed.tUV.to(self.dev, non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record()
                packed.event = ev
                cur.wait_event(ev)
                Yd.record_stream(cur)
                UVd.record_stream(cur)
                Y = (Yd.float() * (1.0 / 64.0)).view(1, 1, fh, fw)
                uv = UVd.float() * (1.0 / 64.0)
            else:
                # (Y, UV) p010 plane views straight from the decoder
                # (top-aligned uint16, interleaved chroma), pageable upload.
                Ynp, UVnp = packed
                Y = (torch.from_numpy(Ynp).to(self.dev).float() * (1.0 / 64.0)
                     ).view(1, 1, fh, fw)
                uv = torch.from_numpy(UVnp).to(self.dev).float() * (1.0 / 64.0)
            U = uv[:, 0::2].unsqueeze(0).unsqueeze(0)
            V = uv[:, 1::2].unsqueeze(0).unsqueeze(0)
            return Y, U, V
        t = torch.from_numpy(np.ascontiguousarray(packed)).to(self.dev).float()
        Y = t[:fh].view(1, 1, fh, fw)
        U = t[fh:fh + fh // 4].reshape(1, 1, fh // 2, fw // 2)
        V = t[fh + fh // 4:fh + fh // 2].reshape(1, 1, fh // 2, fw // 2)
        return Y, U, V

    def _rs_grids(self, i: int, rs_theta: "torch.Tensor", t0: float, scale: float):
        """Rolling-shutter-adjusted warp grids for lens ``i``.

        Each output pixel's ray was captured at sub-frame time t0 + scale*tau;
        the linearised correction shifts its sensor sample by J @ theta(t).
        """
        th = rs_theta                                       # (S,3) radians
        S = th.shape[0]
        t = ((t0 + scale * self.rs_tau[i]).clamp(0, 1) * (S - 1))
        i0 = t.floor().long().clamp(0, S - 2)
        fr = (t - i0.float()).unsqueeze(-1)
        thp = th[i0] * (1 - fr) + th[i0 + 1] * fr           # (h,w,3)
        shift = torch.einsum("hwct,hwt->hwc", self.rs_J[i], thp)   # (h,w,2) px
        shift = (shift * (2.0 / (self.fw - 1))).permute(2, 0, 1).unsqueeze(0)  # normalized
        sh_f = F.interpolate(shift, size=(self.H, self.W), mode="bilinear",
                             align_corners=False).permute(0, 2, 3, 1)
        sh_h = F.interpolate(shift, size=(self.H // 2, self.W // 2), mode="bilinear",
                             align_corners=False).permute(0, 2, 3, 1)
        return self.grid[i] + sh_f, self.grid_half[i] + sh_h

    def _warp_planes(self, packed: np.ndarray, i: int, rs=None):
        """Warp one fisheye's YUV -> (Yw (1,1,H,W), UVw (1,2,H/2,W/2)).

        Applies the factory lens-shading gain (devignette) if present, and the
        rolling-shutter grid adjustment when ``rs`` = (theta, t0, scale) is
        given (see pipeline.Stitcher360.rs_args_for).
        """
        Y, U, V = self._split_yuv(packed)
        gY, gUV = self.grid[i], self.grid_half[i]
        if rs is not None and self.rs_tau is not None:
            gY, gUV = self._rs_grids(i, *rs)
        Yw = F.grid_sample(Y, gY, mode="bilinear",
                           align_corners=True, padding_mode="zeros")
        UVw = F.grid_sample(torch.cat([U, V], 1), gUV,
                            mode="bilinear", align_corners=True, padding_mode="zeros")
        if self.lsc_Y is not None:
            Yw = (Yw - 64.0) * self.lsc_Y[i] + 64.0
            UVw = (UVw - 512.0) * self.lsc_UV[i] + 512.0
        return Yw, UVw

    def _spatial_gain(self, Ya, Yb, max_gain: float):
        """Smooth luma gain field from the overlap, at low res.

        Returns the global luma gain ``gl`` (scalar), the low-res ``lit`` mask,
        its sum ``den``, and the four upsampled scalar spatial gain maps
        (full/half res, for image a/b): g = exp(-/+ 0.5 * logratio).
        """
        lo, hi = np.log(1 / max_gain), np.log(max_gain)
        thr = 6.0 * 1023.0 / 255.0
        a_s = F.avg_pool2d(Ya, 8)
        b_s = F.avg_pool2d(Yb, 8)
        lit = ((self.ov8 > 0.5) & (a_s > thr) & (b_s > thr)).float()
        den = lit.sum().clamp_min(1)
        lma = (a_s * lit).sum()
        lmb = (b_s * lit).sum()
        ratio = torch.where((lma > 1e-3) & (lmb > 1e-3), lmb / lma,
                            torch.ones((), device=self.dev))
        gl = ratio.sqrt().clamp(1 / max_gain, max_gain)
        logr = torch.log(((a_s * gl) / (b_s / gl).clamp_min(1e-3)).clamp(1e-3, 1e3)) * lit
        num = F.avg_pool2d(logr, 4)
        dn = F.avg_pool2d(lit, 4).clamp_min(1e-6)
        num = F.avg_pool2d(num, 15, 1, 7)
        dn = F.avg_pool2d(dn, 15, 1, 7).clamp_min(1e-6)
        lr = (num / dn).clamp(lo, hi)
        s0, s1 = torch.exp(-0.5 * lr), torch.exp(0.5 * lr)
        up = lambda s, sz: F.interpolate(s, size=sz, mode="bilinear", align_corners=False)
        full, half = (self.H, self.W), (self.H // 2, self.W // 2)
        return gl, lit, den, a_s, b_s, (up(s0, full), up(s0, half),
                                        up(s1, full), up(s1, half))

    @staticmethod
    def _scale_centred(Y, UV, gY, gUV):
        """Apply a (broadcastable) gain to centred Y and UV planes."""
        return (Y - 64.0) * gY + 64.0, (UV - 512.0) * gUV + 512.0

    def _expose_luma(self, Ya, UVa, Yb, UVb, max_gain=1.6):
        """Luma-only exposure: one scalar gain field per image (the default)."""
        gl, _, _, _, _, (s0f, s0h, s1f, s1h) = self._spatial_gain(Ya, Yb, max_gain)
        Ya, UVa = self._scale_centred(Ya, UVa, gl * s0f, gl * s0h)
        Yb, UVb = self._scale_centred(Yb, UVb, (1 / gl) * s1f, (1 / gl) * s1h)
        return Ya, UVa, Yb, UVb

    def _apply_wb(self, Y, UV, C, sf, sh):
        """Scalar spatial gain s(x,y) then a constant per-channel matrix C, in YUV.

        C = M @ diag(gain) @ M^-1 is the RGB per-channel gain expressed in centred
        YUV. The Y cross-terms need chroma at full res; that correction is tiny and
        smooth, so the half-res U/V are bilinear-upsampled just for it.
        """
        y = Y - 64.0
        u = UV[:, 0:1] - 512.0
        v = UV[:, 1:2] - 512.0
        # Y's chroma cross-term is a single smooth channel; build it at half res
        # and upsample once (rather than upsampling U and V separately).
        cross = F.interpolate(C[0, 1] * u + C[0, 2] * v, size=(self.H, self.W),
                              mode="bilinear", align_corners=False)
        yh = F.avg_pool2d(y, 2)
        yp = C[0, 0] * y + cross
        up = C[1, 0] * yh + C[1, 1] * u + C[1, 2] * v
        vp = C[2, 0] * yh + C[2, 1] * u + C[2, 2] * v
        return yp * sf + 64.0, torch.cat([up * sh + 512.0, vp * sh + 512.0], 1)

    def _expose_wb(self, Ya, UVa, Yb, UVb, max_gain=1.6):
        """Per-channel white-balance exposure (opt-in): luma spatial gain plus a
        constant per-channel gain, matching the RGB path's lens-colour match."""
        _, lit, den, a_s, b_s, (s0f, s0h, s1f, s1h) = self._spatial_gain(Ya, Yb, max_gain)
        # per-channel global gain from the low-res overlap, in RGB
        UVa_s, UVb_s = F.avg_pool2d(UVa, 4), F.avg_pool2d(UVb, 4)   # UV half -> 1/8

        def to_rgb(Ys, UVs):
            y, u, v = Ys - 64.0, UVs[:, 0:1] - 512.0, UVs[:, 1:2] - 512.0
            return torch.cat([1.164384 * y + 1.792741 * v,
                              1.164384 * y - 0.213249 * u - 0.532909 * v,
                              1.164384 * y + 2.112402 * u], 1)

        ma = (to_rgb(a_s, UVa_s) * lit).sum((0, 2, 3)) / den
        mb = (to_rgb(b_s, UVb_s) * lit).sum((0, 2, 3)) / den
        ratio = torch.where((ma > 1e-3) & (mb > 1e-3), mb / ma, torch.ones_like(ma))
        ga = ratio.sqrt().clamp(1 / max_gain, max_gain)            # (3,) per channel
        Ca = self._M @ torch.diag(ga) @ self._Minv
        Cb = self._M @ torch.diag(1.0 / ga) @ self._Minv
        Ya, UVa = self._apply_wb(Ya, UVa, Ca, s0f, s0h)
        Yb, UVb = self._apply_wb(Yb, UVb, Cb, s1f, s1h)
        return Ya, UVa, Yb, UVb

    def _pack_p010(self, Y, UV):
        """(Y (1,1,H,W), UV (1,2,H/2,W/2)) -> p010le planes, top-aligned uint16.

        +0.5-truncate == round-half-up for these positive values (1 LSB of
        16-bit at exact ties vs round-half-even); skips the separate round
        and second clamp kernels. Range is guaranteed by the first clamp.
        The float math is torch.compile-fused; the cast goes straight to
        uint16 where the backend supports it (halves the D2H bytes vs int32).
        """
        Yp, UVp = self._pack_fused(Y, UV)
        if self._u16:
            return (Yp.to(torch.uint16).cpu().numpy(),
                    UVp.to(torch.uint16).cpu().numpy())
        to = lambda t: t.to(torch.int32).cpu().numpy().astype(np.uint16)
        return to(Yp), to(UVp)

    def _stitch_dev(self, y0: np.ndarray, y1: np.ndarray, R: np.ndarray,
                    exposure: bool = False, wb: bool = False, rs=None):
        """The full stitch, minus the device->host fetch: (Y, UV) on-device."""
        if rs is not None and self.rs_tau is not None:
            th = torch.from_numpy(np.ascontiguousarray(rs[0])).to(self.dev)
            rs = (th, rs[1], rs[2])
        else:
            rs = None
        Ya, UVa = self._warp_planes(y0, 0, rs)
        Yb, UVb = self._warp_planes(y1, 1, rs)
        if exposure:
            expose = self._expose_wb if wb else self._expose_luma
            Ya, UVa, Yb, UVb = expose(Ya, UVa, Yb, UVb)
        Y = torch.lerp(Yb, Ya, self.w3)       # fused blend (see blend())
        UV = torch.lerp(UVb, UVa, self.w3h)
        if not np.allclose(R, np.eye(3), atol=1e-9):
            gf = self._level_grid(R)
            gh = gf[:, ::2, ::2, :].contiguous()
            Y = F.grid_sample(Y, gf, mode="bilinear", align_corners=True, padding_mode="border")
            UV = F.grid_sample(UV, gh, mode="bilinear", align_corners=True, padding_mode="border")
        return Y, UV

    def stitch_yuv(self, y0: np.ndarray, y1: np.ndarray, R: np.ndarray,
                   exposure: bool = False, wb: bool = False, rs=None):
        """Stitch a 10-bit YUV fisheye pair -> p010le planes (Y, UV) for encode.

        ``wb=True`` uses the per-channel white-balance exposure match (matches the
        two lenses' colour, ~15ms/frame dearer); the default luma-only match is
        the shipping path. ``rs`` = (theta (S,3), t0, scale) enables per-pixel
        rolling-shutter correction (pipeline.Stitcher360.rs_args_for).
        """
        return self._pack_p010(*self._stitch_dev(y0, y1, R, exposure, wb, rs))

    @property
    def async_d2h(self) -> bool:
        """True when ``stitch_yuv_async`` is available (CUDA + uint16 cast)."""
        return self.dev.type == "cuda" and self._u16

    def input_pool(self, slots: int | None = None) -> "PinnedInputPool | None":
        """A pinned staging pool for the decoder to fill (CUDA p010 path only).

        Pass its ``alloc`` to ``iter_pairs``; None where it wouldn't help.
        """
        if self.dev.type != "cuda" or not self.input_p010:
            return None
        return PinnedInputPool(self.fh, self.fw, slots)

    def _ensure_async(self, slots: int | None = None):
        if self._down_stream is None:
            import queue as _queue
            from .constants import ASYNC_D2H_SLOTS
            slots = ASYNC_D2H_SLOTS if slots is None else slots
            self._down_stream = torch.cuda.Stream()
            self._out_pool = _queue.Queue()
            for _ in range(slots):
                self._out_pool.put(
                    (torch.empty((self.H, self.W), dtype=torch.uint16).pin_memory(),
                     torch.empty((self.H // 2, self.W), dtype=torch.uint16).pin_memory()))

    def stitch_yuv_async(self, y0: np.ndarray, y1: np.ndarray, R: np.ndarray,
                         exposure: bool = False, wb: bool = False, rs=None):
        """``stitch_yuv``, but the D2H copy runs on a side stream (CUDA only).

        The packed planes are copied into pinned host memory by the copy
        engine while the default stream starts the next frame's kernels —
        on MPS the single kernel queue forbids this, on CUDA it hides the
        whole fetch (~26 ms pageable at 8K on PCIe V100). Returns a handle:
        ``planes()`` blocks until the copy lands, ``release()`` returns the
        pinned slot to the pool. Backpressure comes from the slot pool —
        callers must release each handle after consuming it.
        """
        Y, UV = self._stitch_dev(y0, y1, R, exposure, wb, rs)
        Yp, UVp = self._pack_fused(Y, UV)
        Yd, UVd = Yp.to(torch.uint16), UVp.to(torch.uint16)
        self._ensure_async()
        slot = self._out_pool.get()
        done = torch.cuda.Event()
        self._down_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._down_stream):
            slot[0].copy_(Yd, non_blocking=True)
            slot[1].copy_(UVd, non_blocking=True)
            done.record()
        return _AsyncPlanes(self._out_pool, slot, done, (Yd, UVd))


def p010_to_bgr(Y: np.ndarray, UV: np.ndarray) -> np.ndarray:
    """Convert stitch_yuv's p010le planes to 8-bit BGR (BT.709 limited range).

    Convenience for consumers that want stills or previews from the GPU
    stitch without knowing the p010 layout (top-aligned uint16, interleaved
    half-res chroma). For video, feed the planes to the encoder untouched.
    """
    import cv2
    h, w = Y.shape
    Yf = Y.astype(np.float32) / 64.0
    UVf = UV.astype(np.float32) / 64.0
    U = cv2.resize(UVf[:, 0::2], (w, h))
    V = cv2.resize(UVf[:, 1::2], (w, h))
    y, u, v = Yf - 64.0, U - 512.0, V - 512.0
    bgr = np.stack([1.164384 * y + 2.112402 * u,
                    1.164384 * y - 0.213249 * u - 0.532909 * v,
                    1.164384 * y + 1.792741 * v], -1)
    return np.clip(bgr / 1023.0 * 255.0, 0, 255).astype(np.uint8)


__all__ = ["available", "pick_device", "GpuStitcher", "p010_to_bgr"]
