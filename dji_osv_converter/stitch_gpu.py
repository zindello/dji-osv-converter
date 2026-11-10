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
                 input_fmt: str = "yuv420p10le", yuv_range: str = "limited",
                 compile: bool = True):
        if torch is None:
            raise RuntimeError("torch is not installed")
        dev = pick_device(device)
        self.dev = torch.device(dev)
        self.W, self.H = width, height
        if input_fmt not in ("yuv420p10le", "p010le", "cuda"):
            raise ValueError(f"unsupported input_fmt {input_fmt!r}")
        # Sample range of the 10-bit planes, from the source's declared
        # colour range: limited (BT.709 video levels, what the Osmo 360
        # records: Y 64-940, C 64-960) or full (0-1023). The clamp in the
        # pack and every black-level centring use these; a full-range source
        # through the old fixed 64-940 clamp would have been crushed silently.
        if yuv_range not in ("limited", "full"):
            raise ValueError(f"yuv_range must be 'limited' or 'full', not {yuv_range!r}")
        self.yuv_range = yuv_range
        self.y_lo, self.y_hi, self.c_hi = ((64.0, 940.0, 960.0) if yuv_range == "limited"
                                           else (0.0, 1023.0, 1023.0))
        self._compile = compile
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
        # Seam-flow band bounds (even-aligned for the half-res chroma), inner
        # blend-band mask (at 1/4 width, for the dynamic-seam cost) and lens
        # orientation. The per-frame flow statics build lazily on first use.
        from .stitcher import band_rows
        r0, r1 = band_rows(w)
        r0, r1 = r0 - r0 % 2, r1 + r1 % 2
        self._sf_rows = (r0, r1)
        wcb = w[r0:r1]
        inner = (wcb > 0.001) & (wcb < 0.999)
        self._sf_inner4 = inner.reshape(inner.shape[0], -1, 4).all(axis=2)
        self._sf_cam0_top = bool(wcb[0].mean() >= 0.5)
        self._sf = None
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
        if compile:
            try:
                fn = torch.compile(_grid_math)
                fn(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)   # compile now
                self._grid_fused = fn
            except Exception:                # inductor unavailable: matmul path
                pass
        else:
            self._grid_fused = _grid_math    # eager, same math

        # p010 pack: fuse the clamp/scale/permute float math (the uint16 cast
        # can't be compiled - inductor's MPS backend lacks the dtype - so it
        # stays an eager kernel). 20 -> 10 ms at 8K, bit-identical.
        y_lo, y_hi, c_hi = self.y_lo, self.y_hi, self.c_hi

        def _pack_math(Y, UV):
            Yp = Y[0, 0].clamp(y_lo, y_hi) * 64.0 + 0.5
            UVp = (UV[0].permute(1, 2, 0).reshape(UV.shape[2], -1)
                   .clamp(y_lo, c_hi) * 64.0 + 0.5)
            return Yp, UVp

        self._pack_fused = _pack_math        # eager, same math
        if compile:
            try:
                pf = torch.compile(_pack_math)
                pf(torch.zeros(1, 1, height, width, device=self.dev),
                   torch.zeros(1, 2, height // 2, width // 2, device=self.dev))
                self._pack_fused = pf
            except Exception:
                pass
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
            Yw = (Yw - self.y_lo) * self.lsc_Y[i] + self.y_lo
            UVw = (UVw - 512.0) * self.lsc_UV[i] + 512.0
        return Yw, UVw

    # ---- seam parallax flow (opt-in) ----------------------------------------

    def _sf_init(self):
        """Statics for the band-limited seam-flow morph (built on first use)."""
        r0, r1 = self._sf_rows
        H, W = self.H, self.W
        wc = self.w3[:, :, r0:r1, :]
        band = 4.0 * wc * (1.0 - wc)                     # 0 at edges, 1 mid-band
        t0, t1 = (1.0 - wc) * band, wc * band            # per-lens morph ramps
        ys = torch.linspace(-1.0, 1.0, H, device=self.dev)[r0:r1]
        xs = torch.linspace(-1.0, 1.0, W, device=self.dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy], -1).unsqueeze(0)    # (1,bh,W,2), full-plane coords
        ysh = torch.linspace(-1.0, 1.0, H // 2, device=self.dev)[r0 // 2:r1 // 2]
        xsh = torch.linspace(-1.0, 1.0, W // 2, device=self.dev)
        gyh, gxh = torch.meshgrid(ysh, xsh, indexing="ij")
        baseh = torch.stack([gxh, gyh], -1).unsqueeze(0)
        # Vertical ramp for the glare offset: 1 across the inner blend band,
        # falling linearly to 0 at the outer margin edges.
        bh = r1 - r0
        rows = np.where(self._sf_inner4.any(axis=1))[0]
        ri0, ri1 = int(rows.min()), int(rows.max())
        rp = np.ones(bh, dtype=np.float32)
        if ri0 > 0:
            rp[:ri0] = np.linspace(0.0, 1.0, ri0, endpoint=False)
        if ri1 < bh - 1:
            rp[ri1 + 1:] = np.linspace(1.0, 0.0, bh - ri1 - 1)
        ramp = torch.from_numpy(rp).to(self.dev)[:, None]
        self._sf = dict(t0=t0, t1=t1, t0h=t0[:, :, ::2, ::2], t1h=t1[:, :, ::2, ::2],
                        base=base, baseh=baseh, ramp=ramp)

    def _seam_flow_apply(self, Ya, UVa, Yb, UVb, auto: bool = False):
        """Morph both lenses toward each other inside the blend band.

        The flow itself runs on the CPU (Farneback on a ~90x1920 gray pair,
        see stitcher.seam_flow_small); only tiny fields cross the bus, and the
        band resample is two small grid_samples per lens on-device.

        ``auto=True`` engages each part only when its detector fires (see
        constants.SEAM_AUTO_*): the photometric fix above SEAM_AUTO_PHOTO_T,
        the flow morph + dynamic seam when any column's disagreement clears
        SEAM_DYN_GATE_LO. May return ``(..., None, None)`` = use the static
        blend weights.
        """
        import cv2
        from .stitcher import seam_flow_small
        from .constants import SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_SCALE
        if self._sf is None:
            self._sf_init()
        sf = self._sf
        r0, r1 = self._sf_rows
        H, W = self.H, self.W

        # Veiling-glare correction (see constants.SEAM_OFFSET_*): with bright
        # lights into one lens the ISP leaves an ADDITIVE shadow-lift at the
        # seam (+20/1024 measured). Estimate a per-column offset from
        # geometrically AGREEING overlap pixels, split it across both lenses,
        # ramped to zero at the band margins - the lenses meet exactly at the
        # seam and each fades into its own uncorrected interior. Runs before
        # flow/seam so the mismatch pollutes neither.
        from .constants import SEAM_OFFSET_AGREE_T, SEAM_OFFSET_SIGMA
        from .stitcher import seam_luma_offsets
        # ONE device->host sync feeds both the photometric estimator and the
        # flow grays (a second pooled download mid-pipeline stalls the MPS
        # queue - measured +100 ms on the whole stitch).
        an4 = F.avg_pool2d(Ya[:, :, r0:r1], (1, 4))
        bn4 = F.avg_pool2d(Yb[:, :, r0:r1], (1, 4))
        ab = torch.cat([an4, bn4], 0).cpu().numpy()
        an, bn = ab[0, 0], ab[1, 0]
        lut, spatial = seam_luma_offsets(an, bn, self._sf_inner4,
                                         lo=self.y_lo, hi=self.y_hi, n_lut=1024,
                                         sigma=SEAM_OFFSET_SIGMA,
                                         agree_t=SEAM_OFFSET_AGREE_T)
        from .constants import SEAM_AUTO_PHOTO_T, SEAM_DYN_GATE_LO
        need_photo = (not auto) or (np.abs(lut).max() > SEAM_AUTO_PHOTO_T
                                    or np.abs(spatial).max() > SEAM_AUTO_PHOTO_T)
        if not need_photo:
            lut = np.zeros_like(lut)
            spatial = np.zeros_like(spatial)
        else:
            lut_t = torch.from_numpy(0.5 * lut).to(self.dev)
            sp = torch.from_numpy(0.5 * spatial).to(self.dev)[None, None]
            sp = F.interpolate(sp, size=(r1 - r0, W), mode="bilinear",
                               align_corners=False)[0, 0]
            ramp = sf["ramp"]                             # (bh,1): 0 edges, 1 in band
            for plane, sgn in ((Ya, 1.0), (Yb, -1.0)):
                band = plane[:, :, r0:r1]
                corr = lut_t[band.clamp(0, 1023).long()] + sp
                plane[:, :, r0:r1] = band + sgn * corr * ramp

        # Flow grays on the CPU from the same pooled data, correction applied
        # (10-bit centred Y -> 8-bit gray for Farneback; the small pool-order
        # difference vs pooling the corrected full-res planes is negligible
        # for flow). Assumes SEAM_FLOW_SCALE == the estimator's 1/4 pooling.
        k = max(int(round(1.0 / SEAM_FLOW_SCALE)), 1)
        cA = an + 0.5 * (lut[np.clip(an.astype(np.int64), 0, 1023)] + spatial)
        cB = bn - 0.5 * (lut[np.clip(bn.astype(np.int64), 0, 1023)] + spatial)
        if auto:
            # Parallax detector: engage flow + dynamic seam only when some
            # column's mean disagreement clears the gate's own threshold -
            # below it the gate would zero the dynamic seam anyway.
            d = np.abs(cB - cA) * self._sf_inner4
            col = d.sum(0) / np.maximum(self._sf_inner4.sum(0), 1)
            if col.max() * 0.25 < SEAM_DYN_GATE_LO:      # 10-bit -> 8-bit
                return Ya, UVa, Yb, UVb, None, None
        bh4 = (cA.shape[0] // k) * k
        pr = lambda x: x[:bh4].reshape(bh4 // k, k, x.shape[1]).mean(1)
        ga = np.clip(pr(cA) * 0.25, 0, 255).astype(np.uint8)
        gb = np.clip(pr(cB) * 0.25, 0, 255).astype(np.uint8)
        f01s, f10s = seam_flow_small(ga, gb)             # small-res px units
        sh, sw = ga.shape
        bh = r1 - r0

        def up(f, size, w_full, h_full):
            t = torch.from_numpy(np.ascontiguousarray(f)).to(self.dev)
            t = t.permute(2, 0, 1).unsqueeze(0)          # (1,2,sh,sw)
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
            fx = (t[:, 0] * (W / sw)).clamp(-SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_MAX_SHIFT_PX)
            fy = (t[:, 1] * (bh / sh)).clamp(-SEAM_FLOW_MAX_SHIFT_PX, SEAM_FLOW_MAX_SHIFT_PX)
            # full-res px -> normalized coords of the sampled (h_full, w_full) plane
            return torch.stack([fx * (2.0 / (w_full - 1)),
                                fy * (2.0 / (h_full - 1))], -1)

        o01 = up(f01s, (bh, W), W, H)
        o10 = up(f10s, (bh, W), W, H)
        o01h = up(f01s * 0.5, (bh // 2, W // 2), W // 2, H // 2)
        o10h = up(f10s * 0.5, (bh // 2, W // 2), W // 2, H // 2)
        t0, t1 = sf["t0"].permute(0, 2, 3, 1), sf["t1"].permute(0, 2, 3, 1)
        t0h, t1h = sf["t0h"].permute(0, 2, 3, 1), sf["t1h"].permute(0, 2, 3, 1)
        gs = lambda img, grid: F.grid_sample(img, grid, mode="bilinear",
                                             align_corners=True, padding_mode="border")
        Ya[:, :, r0:r1] = gs(Ya, sf["base"] + o01 * t0)
        Yb[:, :, r0:r1] = gs(Yb, sf["base"] + o10 * t1)
        UVa[:, :, r0 // 2:r1 // 2] = gs(UVa, sf["baseh"] + o01h * t0h)
        UVb[:, :, r0 // 2:r1 // 2] = gs(UVb, sf["baseh"] + o10h * t1h)

        # Dynamic seam: flow aligns what both lenses see, but near an object
        # they see different SIDES of it (occlusion), and the wide cross-fade
        # dissolves it into the background. Route a min-disagreement path
        # through the morphed band and feather narrowly around it - GATED to
        # the columns where the lenses actually disagree; elsewhere the
        # static wide fade stays (a narrow cut there draws a visible ring).
        from .stitcher import seam_gate, seam_path
        from .constants import SEAM_DYN_FEATHER_PX
        d4 = F.avg_pool2d((Ya[:, :, r0:r1] - Yb[:, :, r0:r1]).abs(), (1, 4))[0, 0]
        cost = d4.cpu().numpy().astype(np.float32) * 0.25       # 10-bit -> 8-bit
        seam = seam_path(np.where(self._sf_inner4, cost, 1e6).astype(np.float32), W)
        gate = seam_gate(cost, self._sf_inner4, W)
        seam_t = torch.from_numpy(seam).to(self.dev)
        g_t = torch.from_numpy(gate).to(self.dev)[None, :]
        yy = torch.arange(bh, device=self.dev, dtype=torch.float32)[:, None]
        wd = (0.5 + (seam_t[None, :] - yy) / SEAM_DYN_FEATHER_PX).clamp(0, 1)
        if not self._sf_cam0_top:
            wd = 1.0 - wd
        wd = g_t * wd + (1.0 - g_t) * self.w3[0, 0, r0:r1]
        wd = wd[None, None]
        return Ya, UVa, Yb, UVb, wd, F.avg_pool2d(wd, 2)

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

    def _scale_centred(self, Y, UV, gY, gUV):
        """Apply a (broadcastable) gain to centred Y and UV planes."""
        lo = self.y_lo
        return (Y - lo) * gY + lo, (UV - 512.0) * gUV + 512.0

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
        lo = self.y_lo
        y = Y - lo
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
        return yp * sf + lo, torch.cat([up * sh + 512.0, vp * sh + 512.0], 1)

    def _expose_wb(self, Ya, UVa, Yb, UVb, max_gain=1.6):
        """Per-channel white-balance exposure (opt-in): luma spatial gain plus a
        constant per-channel gain, matching the RGB path's lens-colour match."""
        _, lit, den, a_s, b_s, (s0f, s0h, s1f, s1h) = self._spatial_gain(Ya, Yb, max_gain)
        # per-channel global gain from the low-res overlap, in RGB
        UVa_s, UVb_s = F.avg_pool2d(UVa, 4), F.avg_pool2d(UVb, 4)   # UV half -> 1/8

        def to_rgb(Ys, UVs):
            y, u, v = Ys - self.y_lo, UVs[:, 0:1] - 512.0, UVs[:, 1:2] - 512.0
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
                    exposure: bool = False, wb: bool = False, rs=None,
                    seam_flow=False):
        """The full stitch, minus the device->host fetch: (Y, UV) on-device."""
        if rs is not None and self.rs_tau is not None:
            th = torch.from_numpy(np.ascontiguousarray(rs[0])).to(self.dev)
            rs = (th, rs[1], rs[2])
        else:
            rs = None
        Ya, UVa = self._warp_planes(y0, 0, rs)
        Yb, UVb = self._warp_planes(y1, 1, rs)
        sfw = None
        if seam_flow:
            Ya, UVa, Yb, UVb, wd, wdh = self._seam_flow_apply(
                Ya, UVa, Yb, UVb, auto=(seam_flow == "auto"))
            sfw = (wd, wdh) if wd is not None else None
        if exposure:
            expose = self._expose_wb if wb else self._expose_luma
            Ya, UVa, Yb, UVb = expose(Ya, UVa, Yb, UVb)
        Y = torch.lerp(Yb, Ya, self.w3)       # fused blend (see blend())
        UV = torch.lerp(UVb, UVa, self.w3h)
        if sfw is not None:                   # narrow dynamic seam in the band
            r0, r1 = self._sf_rows
            Y[:, :, r0:r1] = torch.lerp(Yb[:, :, r0:r1], Ya[:, :, r0:r1], sfw[0])
            UV[:, :, r0 // 2:r1 // 2] = torch.lerp(UVb[:, :, r0 // 2:r1 // 2],
                                                   UVa[:, :, r0 // 2:r1 // 2], sfw[1])
        if not np.allclose(R, np.eye(3), atol=1e-9):
            gf = self._level_grid(R)
            gh = gf[:, ::2, ::2, :].contiguous()
            Y = F.grid_sample(Y, gf, mode="bilinear", align_corners=True, padding_mode="border")
            UV = F.grid_sample(UV, gh, mode="bilinear", align_corners=True, padding_mode="border")
        return Y, UV

    def stitch_yuv(self, y0: np.ndarray, y1: np.ndarray, R: np.ndarray,
                   exposure: bool = False, wb: bool = False, rs=None,
                   seam_flow=False):
        """Stitch a 10-bit YUV fisheye pair -> p010le planes (Y, UV) for encode.

        ``wb=True`` uses the per-channel white-balance exposure match (matches the
        two lenses' colour, ~15ms/frame dearer); the default luma-only match is
        the shipping path. ``rs`` = (theta (S,3), t0, scale) enables per-pixel
        rolling-shutter correction (pipeline.Stitcher360.rs_args_for).
        ``seam_flow``: True runs the full seam stack every frame (photometric
        match + flow morph + gated dynamic seam); "auto" runs a cheap
        per-frame detector and engages each part only when needed (the
        recommended default - see constants.SEAM_AUTO_*); False = static
        blend only.
        """
        return self._pack_p010(*self._stitch_dev(y0, y1, R, exposure, wb, rs,
                                                 seam_flow=seam_flow))

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
                         exposure: bool = False, wb: bool = False, rs=None,
                         seam_flow=False):
        """``stitch_yuv``, but the D2H copy runs on a side stream (CUDA only).

        The packed planes are copied into pinned host memory by the copy
        engine while the default stream starts the next frame's kernels —
        on MPS the single kernel queue forbids this, on CUDA it hides the
        whole fetch (~26 ms pageable at 8K on PCIe V100). Returns a handle:
        ``planes()`` blocks until the copy lands, ``release()`` returns the
        pinned slot to the pool. Backpressure comes from the slot pool —
        callers must release each handle after consuming it.
        """
        Y, UV = self._stitch_dev(y0, y1, R, exposure, wb, rs,
                                 seam_flow=seam_flow)
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
