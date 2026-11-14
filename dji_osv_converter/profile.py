"""Per-stage timing of a conversion, cheap enough to leave on.

``convert()`` activates a :class:`Profile` for its duration; every pipeline
thread (decode workers, the stitch thread, the frame builder, the encoder
sessions, the muxer) records its stages into it, and at the end of the
conversion two lines are logged for the caller to find::

    osv-profile: {"frames": 450, "fps": 31.2, "stages": {...}, ...}
    osv-mem: {"peak_rss_mib": 5123, "gpu_peak_alloc_mib": 6410, ...}

Both payloads are single-line JSON after a fixed prefix
(:data:`PROFILE_PREFIX`, :data:`MEM_PREFIX`). The same dictionaries are
available from :func:`last_profile`.

Every stage reports, in milliseconds per output frame:

``wall``  host wall time the owning thread spent inside the stage. For a GPU
          stage this is the time to LAUNCH the kernels (they run
          asynchronously) plus any host sync inside it - not the kernels' run
          time.
``cpu``   CPU time the owning thread burned inside the stage
          (``time.thread_time``): ``wall - cpu`` is time spent blocked
          (GIL, a queue, a device sync, the kernel).
``gpu``   (GPU stages only) time on the stitch stream's own timeline between
          this stage's last kernel and the previous stage's, from CUDA events
          (``torch.cuda.Event(enable_timing=True)``) that are resolved lazily
          - no host sync is added. Summed over the stages it is the stream's
          busy-or-starved time per frame; a stage whose ``gpu`` is close to
          its ``wall`` is launch-bound (the host is the limiter), one whose
          ``gpu`` is far above ``wall`` is compute-bound.

``mode="sync"`` synchronises the device at every GPU stage boundary instead
and reports the synchronised wall time as ``gpu``: exact attribution on any
backend (Apple MPS has no events), at the price of serialising the pipeline -
a diagnostic, never the production setting.

Stages named ``*.idle`` are a thread waiting for work (an empty input queue);
``*.wait`` are a thread blocked on a full output queue or an unfinished
upstream copy. They are reported because they show which stage is the
limiter: the limiter never idles, everything upstream of it waits and
everything downstream idles.

The first frames of a conversion (warm-up: kernel compilation, pool
allocation, encoder start) are excluded from the per-frame figures; see
``constants.PROFILE_WARMUP_FRAMES``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque

logger = logging.getLogger(__name__)

PROFILE_PREFIX = "osv-profile: "
MEM_PREFIX = "osv-mem: "

_perf = time.perf_counter
_tcpu = time.thread_time


class _NullStage:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_NULL_STAGE = _NullStage()


class NullProfile:
    """The do-nothing profile used when profiling is off."""

    enabled = False
    mode = "off"

    def stage(self, name, gpu=False):
        return _NULL_STAGE

    def add(self, name, wall, cpu=0.0):
        pass

    def frame_begin(self):
        pass

    def frame_end(self):
        pass

    def note(self, **kw):
        pass


class _Stage:
    __slots__ = ("p", "name", "gpu", "t0", "c0")

    def __init__(self, p, name, gpu):
        self.p, self.name, self.gpu = p, name, gpu

    def __enter__(self):
        self.t0 = _perf()
        self.c0 = _tcpu()
        return self

    def __exit__(self, *a):
        p = self.p
        if self.gpu:
            p._gpu_mark(self.name, self.t0)
        t1, c1 = _perf(), _tcpu()
        acc = p._acc
        e = acc.get(self.name)
        if e is None:
            with p._lock:
                e = acc.setdefault(self.name, [0.0, 0.0, 0])
        # list item updates are not atomic across threads; stages shared by
        # several threads (decode, encode sessions) take the lock
        if self.name in p._shared:
            with p._lock:
                e[0] += t1 - self.t0
                e[1] += c1 - self.c0
                e[2] += 1
        else:
            e[0] += t1 - self.t0
            e[1] += c1 - self.c0
            e[2] += 1
        return False


class Profile:
    """Stage accumulators for one conversion (see the module docstring)."""

    enabled = True

    def __init__(self, mode: str = "events", device=None, warmup: int | None = None):
        from .constants import PROFILE_WARMUP_FRAMES
        if mode not in ("events", "sync"):
            raise ValueError("profile mode must be 'events' or 'sync', "
                             f"not {mode!r}")
        self.mode = mode
        self.warmup = PROFILE_WARMUP_FRAMES if warmup is None else int(warmup)
        self._lock = threading.Lock()
        self._acc: dict[str, list] = {}        # name -> [wall_s, cpu_s, calls]
        self._gacc: dict[str, float] = {}      # name -> gpu seconds
        self._shared: set[str] = set()
        self._notes: dict = {}
        self._frames = 0
        self._snap = None                      # accumulators at the end of warm-up
        self._t_start = _perf()
        self._c_start = time.process_time()
        self._t_first = None
        self._t_warm = None
        self._t_end = None
        self._c_warm = None
        self._c_end = None
        # GPU timeline
        self._dev = None
        self._torch = None
        self._gpu_frame: list = []             # [(name, event)] of the open frame
        self._gpu_prev = None                  # previous boundary (event | time)
        self._gpu_pending: deque = deque()     # closed frames awaiting resolution
        self._gpu_snap = None
        self.set_device(device)

    # -- configuration ----------------------------------------------------

    def set_device(self, device) -> None:
        """Tell the profile which torch device the stitch runs on."""
        self._dev = device
        if device is not None:
            import torch
            self._torch = torch

    def share(self, *names: str) -> None:
        """Declare stages that several threads record into (locked updates)."""
        self._shared.update(names)

    def note(self, **kw) -> None:
        """Attach configuration facts to the report (decode format, ...)."""
        self._notes.update(kw)

    # -- recording --------------------------------------------------------

    def stage(self, name: str, gpu: bool = False):
        """Context manager timing one stage on the calling thread.

        ``gpu=True`` (stitch thread only) also closes a GPU-timeline segment
        named after the stage when the block exits.
        """
        return _Stage(self, name, gpu)

    def add(self, name: str, wall: float, cpu: float = 0.0) -> None:
        with self._lock:
            e = self._acc.setdefault(name, [0.0, 0.0, 0])
            e[0] += wall
            e[1] += cpu
            e[2] += 1

    def _event(self):
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _sync(self):
        d = self._dev
        if d is None:
            return
        if d.type == "cuda":
            self._torch.cuda.synchronize()
        elif d.type == "mps":
            self._torch.mps.synchronize()

    def frame_begin(self) -> None:
        """Open a frame on the stitch thread's GPU timeline."""
        if self._t_first is None:
            self._t_first = _perf()
        if self._dev is None:
            return
        if self.mode == "sync":
            self._sync()
            self._gpu_prev = _perf()
        elif self._dev.type == "cuda":
            self._gpu_prev = self._event()
            self._gpu_frame = [("", self._gpu_prev)]

    def _gpu_mark(self, name: str, t0: float) -> None:
        if self._dev is None or self._gpu_prev is None:
            return
        if self.mode == "sync":
            self._sync()
            t = _perf()
            # the segment starts at the stage's own entry: host work between
            # two GPU stages is not GPU time
            self._gacc[name] = self._gacc.get(name, 0.0) + (t - max(t0, self._gpu_prev))
            self._gpu_prev = t
        elif self._dev.type == "cuda":
            self._gpu_frame.append((name, self._event()))

    def frame_end(self) -> None:
        """Close the frame: count it and resolve finished GPU segments."""
        if self._gpu_frame:
            self._gpu_pending.append(self._gpu_frame)
            self._gpu_frame = []
            self._resolve(final=False)
        self._frames += 1
        if self._frames == self.warmup:
            self._resolve(final=True)
            with self._lock:
                self._snap = {k: list(v) for k, v in self._acc.items()}
            self._gpu_snap = dict(self._gacc)
            self._t_warm = _perf()
            self._c_warm = time.process_time()

    def _resolve(self, final: bool) -> None:
        pend = self._gpu_pending
        while pend:
            marks = pend[0]
            if not final and not marks[-1][1].query():
                break
            pend.popleft()
            prev = marks[0][1]
            for name, ev in marks[1:]:
                self._gacc[name] = self._gacc.get(name, 0.0) + prev.elapsed_time(ev) * 1e-3
                prev = ev

    # -- reporting --------------------------------------------------------

    def finish(self) -> dict:
        """Close the profile and return the report dictionary."""
        if self._dev is not None and self.mode == "events" and self._dev.type == "cuda":
            self._torch.cuda.synchronize()
        self._resolve(final=True)
        self._t_end = _perf()
        self._c_end = time.process_time()
        n = self._frames
        warm = self.warmup if (self._snap is not None and n > self.warmup) else 0
        snap = self._snap if warm else {}
        gsnap = self._gpu_snap if warm else {}
        m = max(n - warm, 1)
        t0 = self._t_warm if warm else (self._t_first or self._t_start)
        c0 = self._c_warm if warm else self._c_start
        wall = self._t_end - t0
        stages = {}
        for name in sorted(self._acc):
            w, c, k = self._acc[name]
            w0, c0_, k0 = snap.get(name, (0.0, 0.0, 0))
            st = {"wall": round((w - w0) / m * 1e3, 3),
                  "cpu": round((c - c0_) / m * 1e3, 3),
                  "calls": round((k - k0) / m, 2)}
            if name in self._gacc:
                st["gpu"] = round((self._gacc[name] - gsnap.get(name, 0.0)) / m * 1e3, 3)
            stages[name] = st
        rep = {
            "frames": n,
            "warmup_frames": warm,
            "mode": self.mode,
            "setup_s": round((self._t_first or self._t_end) - self._t_start, 3),
            "total_s": round(self._t_end - self._t_start, 3),
            "ms_per_frame": round(wall / m * 1e3, 3),
            "fps": round(m / wall, 3) if wall > 0 else 0.0,
            "cpu_cores": round((self._c_end - c0) / wall, 3) if wall > 0 else 0.0,
            "config": dict(self._notes),
            "stages": stages,
        }
        return rep


def memory_report(device=None) -> dict:
    """Peak resident memory of the process and peak torch GPU memory (MiB)."""
    import resource
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kibibytes on Linux
    rss_mib = rss / (1024.0 * 1024.0) if sys.platform == "darwin" else rss / 1024.0
    out = {"peak_rss_mib": round(rss_mib, 1)}
    if device is not None:
        import torch
        mib = 1024.0 * 1024.0
        if device.type == "cuda":
            out["gpu_peak_alloc_mib"] = round(torch.cuda.max_memory_allocated(device) / mib, 1)
            out["gpu_peak_reserved_mib"] = round(torch.cuda.max_memory_reserved(device) / mib, 1)
        elif device.type == "mps":
            # MPS keeps no peak counters: these are the values at the end of
            # the conversion (the driver pool only grows during a run)
            out["gpu_alloc_mib"] = round(torch.mps.current_allocated_memory() / mib, 1)
            out["gpu_driver_mib"] = round(torch.mps.driver_allocated_memory() / mib, 1)
    return out


def reset_gpu_peaks(device) -> None:
    if device is not None and device.type == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats(device)


_NULL = NullProfile()
_ACTIVE: "Profile | NullProfile" = _NULL
_LAST: dict | None = None


def active() -> "Profile | NullProfile":
    """The profile of the conversion in progress (a no-op one when off)."""
    return _ACTIVE


def activate(p: "Profile | None") -> None:
    global _ACTIVE
    _ACTIVE = p if p is not None else _NULL


def last_profile() -> dict | None:
    """``{"profile": ..., "memory": ...}`` of the last finished conversion."""
    return _LAST


def publish(profile: dict, memory: dict, emit: bool = False) -> None:
    """Store the report and log the two lines (also print them if ``emit``)."""
    global _LAST
    _LAST = {"profile": profile, "memory": memory}
    for line in (PROFILE_PREFIX + json.dumps(profile, separators=(",", ":")),
                 MEM_PREFIX + json.dumps(memory, separators=(",", ":"))):
        logger.info("%s", line)
        if emit:
            print(line, flush=True)


def format_table(profile: dict) -> str:
    """The stage table of a report as Markdown (ms per frame)."""
    rows = ["| stage | wall | cpu | gpu | calls/frame |", "|---|---:|---:|---:|---:|"]
    for name, st in profile["stages"].items():
        g = f"{st['gpu']:.2f}" if "gpu" in st else ""
        rows.append(f"| `{name}` | {st['wall']:.2f} | {st['cpu']:.2f} | {g} | {st['calls']:g} |")
    return "\n".join(rows)


__all__ = ["Profile", "NullProfile", "active", "activate", "last_profile", "publish",
           "memory_report", "reset_gpu_peaks", "format_table",
           "PROFILE_PREFIX", "MEM_PREFIX"]
