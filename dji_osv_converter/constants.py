"""Every constant that is NOT read from the OSV file, with its provenance.

The project's rule is that everything possible is derived from the OSV itself
(lens calibration, orientation, mount, exposure state...). What remains - and
lives here - falls into three provenance classes, labelled per constant:

``[MEASURED]``
    Fit against DJI Studio's own exports of reference footage. Believed
    universal for the camera model (they describe DJI's conventions, not a
    unit), but measured on one unit so far - re-verify if a new unit disagrees.

``[CONVENTION]``
    A coordinate/order convention that was *verified* against ground truth
    (e.g. the quaternion component order, checked via gravity). Not tunable.

``[ASSUMED]``
    A default with no ground truth. These are the first suspects when output
    disagrees with DJI, and the sweep-candidates for tuning.

Keeping them in one file is deliberate: a future OSV that breaks an assumption
should send you here first.
"""

from __future__ import annotations

import numpy as np


def _orthonormal(m) -> np.ndarray:
    """Nearest rotation matrix (SVD) - measured matrices carry fit noise."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


# ---- Layout / geometry ------------------------------------------------------

# [MEASURED] Our rig-frame equirect layout vs DJI's "plain" export layout:
# approximately Rx(90 deg) plus a ~1.6 deg residual, constant through a full
# tumble (spread <2 deg), so it is a render-layout convention, not motion data.
K_PLAIN = _orthonormal([
    [0.999620, 0.000390, 0.027440],
    [0.027440, 0.003060, -0.999620],
    [-0.000480, 1.000000, 0.003050],
])

# [MEASURED] Format-level bridge from the IMU body frame to the stitch equirect
# frame for world-referenced VECTORS (up/forward), with the PER-UNIT
# camera<->IMU mount rotation factored OUT. The full per-file bridge is
#
#     bridge = C_UP_FORMAT @ mount        (mount = OSV header field 1.3.1)
#
# and (bridge @ R(quat).T @ WORLD_UP) matches DJI's level up-vector to 0.24 deg
# mean / 0.7 deg max across an entire worst-case tumble. The original fit
# against DJI's horizonlevelling export produced the composite bridge for the
# reference unit; C_UP_FORMAT = composite @ mount_ref.T (exact to 2e-16), so a
# unit with a different mount skew recomposes correctly from its own file.
# det = -1 BY DESIGN: the equirect z-axis convention is left-handed relative to
# the physical IMU frame, so the bridge legitimately includes a reflection
# (a proper-rotation fit such as Procrustes cannot represent it - fit in the
# quaternion-consistent frame first, then apply diag(1,1,-1)).
C_UP_FORMAT = np.array([
    [-0.10275825, -0.98422164, -0.14404337],
    [-0.98455879, 0.08000611, 0.15570169],
    [0.14172063, -0.15781880, 0.97724536],
])

# [CONVENTION] Component order of every orientation quaternion in the OSV
# (per-frame field 3.2.9, high-rate 3.3.2.1.3, calibration extrinsics).
# Verified via gravity: with wxyz, world-up comes out as +Z.
QUAT_ORDER = "wxyz"

# [CONVENTION] World axes in the quaternion's world frame. Up verified via
# gravity; forward is the fixed heading the horizon lock pins yaw to (any
# horizontal direction works - world X is the natural choice).
WORLD_UP = np.array([0.0, 0.0, 1.0])
WORLD_FWD = np.array([1.0, 0.0, 0.0])

# [CONVENTION] Which calibration field carries the lens extrinsic rotation and
# its quaternion order (calibration block field 28; field 21 holds a
# near-identical second copy whose meaning is undecoded).
EXTRINSIC_SOURCE = "quat"
EXTRINSIC_ORDER = "wxyz"


# ---- Stitch geometry --------------------------------------------------------

# [MEASURED] DJI's blend-band width as a fraction of the 360 deg equirect,
# read once from DJI Studio's DualStitcher bundle (seam_overlap_ratio.json)
# and matching the 292 px band visible in DJI's own 8K exports. Constant here
# so behaviour never depends on DJI Studio being installed.
SEAM_OVERLAP_RATIO = 0.038

# Per-lens field of view implied by the overlap: each fisheye must cover half
# the sphere plus the shared band.
DEFAULT_FOV_DEG = 180.0 + SEAM_OVERLAP_RATIO * 360.0    # 193.68

# [ASSUMED] Width of the coverage-weight taper at the fisheye rim, as a
# fraction of the blend band (band/3). No ground truth; only affects the few
# outermost degrees.
FEATHER_FRACTION = 1.0 / 3.0


# ---- Stabilisation ----------------------------------------------------------

# [ASSUMED] Gaussian sigma (in frames) of the RockSteady trajectory smoother.
# Chosen to match DJI's smoothness by eye/measurement (rendered jitter lands at
# or below DJI's export with 15), but DJI's actual filter shape is unknown.
RS_SIGMA = 15.0


# ---- Rolling-shutter correction (DISABLED) ----------------------------------
#
# [ASSUMED - NEVER SUCCESSFULLY FIT] The RS subsystem is off (video.use_rs):
# on this camera's footage every parameter combination measured worse than no
# correction once the rocksteady composition was fixed. Retained for potential
# non-DJI sources; these values are placeholders, not calibration.
RS_TABLE_DS = 4          # tau/Jacobian tables at 1/RS_TABLE_DS output res
RS_READOUT = 0.72        # readout time as a fraction of the frame period
RS_ROW_DIR = 1.0         # +1: row 0 read first; -1: last row first
RS_SIGN = 1.0            # sign of the correction


# ---- Encoding ---------------------------------------------------------------

# [CONVENTION] Encoding-transfer gamma per AVCOL_TRC id: SDR OETFs
# (BT.709/601/2020) are a ~0.45 power (gamma ~2.2), linear 1.0, BT.470BG 2.8.
# [ASSUMED] the 2.2 fallback when the stream declares nothing.
TRC_GAMMA = {1: 2.2, 4: 2.2, 6: 2.2, 14: 2.2, 15: 2.2, 8: 1.0, 5: 2.8}
TRC_GAMMA_FALLBACK = 2.2

# [MEASURED] Output bitrate in bits per pixel per frame for the hardware
# encoder path: DJI's 8K exports run ~0.28 bpp; 0.25 is visually equivalent,
# 0.10 was visibly soft.
BITRATE_BPP = 0.25


# ---- Pipeline concurrency (CUDA) --------------------------------------------
# None of these are readable from hardware or the OSV; they are the knobs of
# the threaded GPU pipeline, sized on the V100 box (docs/architecture.md).

# Per-card video-engine counts. NVML exposes no engine-count query, so this
# is the dial to turn when the pipeline meets a new GPU: add a row, and the
# session-parallel encoder (and any future decode/batch tooling) picks it up.
# Keys match against torch's device name by substring, longest key first
# (so "L40" wins over "L4" on an L40S). Unknown cards get GPU_ENGINES_DEFAULT
# - deliberately conservative (single session = the pre-parallel behaviour).
#
# nvenc: encode engines - one HEVC session saturates exactly one, so
#        convert() runs this many parallel sessions by default.
# nvdec: decode engines - one OSV needs at most 2 decoder instances (two
#        fisheye streams share one file); the count records whether the
#        card can decode clips in parallel (batching) beyond that.
GPU_ENGINES = {
    # [MEASURED] on this project's V100 box: a lone NVENC session capped at
    # exactly 33% aggregate (1 of 3 engines at 100%); NVDEC saturates at
    # ~25 ms/fisheye-pair with both decode workers sharing one engine.
    "V100": {"nvenc": 3, "nvdec": 1},
    # [ASSUMED] from NVIDIA's public Video Codec SDK support matrix -
    # unverified by this codebase; verify with a utilization log
    # (nvidia-smi -q -d UTILIZATION during a convert) before trusting.
    "P100": {"nvenc": 3, "nvdec": 1},
    "T4": {"nvenc": 1, "nvdec": 2},
    "A10": {"nvenc": 1, "nvdec": 2},
    "A100": {"nvenc": 0, "nvdec": 5},   # no NVENC: falls back to libx265
    "L4": {"nvenc": 2, "nvdec": 4},
    "L40": {"nvenc": 3, "nvdec": 3},
    "H100": {"nvenc": 0, "nvdec": 7},   # no NVENC
}
# [ASSUMED] Unknown card: one session of each - correct everywhere, fast
# nowhere special. Add a GPU_ENGINES row to turn the dial up.
GPU_ENGINES_DEFAULT = {"nvenc": 1, "nvdec": 1}


def gpu_engines(device_name: str | None) -> dict:
    """Engine counts for a CUDA device name.

    Longest key first, and the key must not be followed by a further digit,
    so "A100-SXM4" matches A100 but "RTX A1000" does not, and "L40S" matches
    L40 rather than L4.
    """
    import re
    if device_name:
        for key in sorted(GPU_ENGINES, key=len, reverse=True):
            if re.search(re.escape(key) + r"(?!\d)", device_name):
                return GPU_ENGINES[key]
    return GPU_ENGINES_DEFAULT

# [ASSUMED] Seconds of video per closed GOP chunk in session-parallel encode.
# Each chunk opens with a forced IDR: shorter chunks spread load faster but
# cost bitrate (2 s measured ~+6% file size vs a single session's ~8 s GOP).
ENC_CHUNK_SECONDS = 2.0

# [ASSUMED] Bounded-queue depths, sized so no stage starves while memory
# stays modest: stage handoff queues (frames), per-decode-worker queues
# (fisheye frames), pinned H2D staging slots (>= prefetch chain: 2/pair x 4
# prefetched + 2 x 3 worker-queued + in-flight), pinned D2H slots
# (> stage queue + frames in flight on the copy stream).
PIPELINE_QUEUE_FRAMES = 4
DECODE_WORKER_QUEUE = 3
PINNED_INPUT_SLOTS = 16
ASYNC_D2H_SLOTS = 6

# [ASSUMED] Packets fed to the hevc_cuvid probe before concluding it cannot
# decode this file (the decoder has multi-packet latency before frame 1).
CUVID_PROBE_PACKETS = 64


__all__ = [
    "K_PLAIN", "C_UP_FORMAT", "QUAT_ORDER", "WORLD_UP", "WORLD_FWD",
    "EXTRINSIC_SOURCE", "EXTRINSIC_ORDER",
    "SEAM_OVERLAP_RATIO", "DEFAULT_FOV_DEG", "FEATHER_FRACTION",
    "RS_SIGMA", "RS_TABLE_DS", "RS_READOUT", "RS_ROW_DIR", "RS_SIGN",
    "TRC_GAMMA", "TRC_GAMMA_FALLBACK", "BITRATE_BPP",
    "GPU_ENGINES", "GPU_ENGINES_DEFAULT", "gpu_engines",
    "ENC_CHUNK_SECONDS", "PIPELINE_QUEUE_FRAMES",
    "DECODE_WORKER_QUEUE", "PINNED_INPUT_SLOTS", "ASYNC_D2H_SLOTS",
    "CUVID_PROBE_PACKETS",
]
