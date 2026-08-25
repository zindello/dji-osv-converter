# Constants & provenance

Everything possible is derived from the OSV file at runtime — per-unit lens
calibration, extrinsics, the camera↔IMU mount, orientation, exposure state,
encoding gamma. What cannot be derived lives in one place,
[`dji_osv_converter/constants.py`](../dji_osv_converter/constants.py), where
every value is labelled with its provenance:

| class | meaning | on a new camera |
|---|---|---|
| `[MEASURED]` | fit against DJI Studio's own exports; believed universal for the model (they describe DJI's conventions, not a unit) | re-verify if output disagrees |
| `[CONVENTION]` | a coordinate/order convention verified against ground truth | fixed |
| `[ASSUMED]` | a default with no ground truth | first suspects for tuning |

Highlights:

- **`K_PLAIN`** `[MEASURED]` — our equirect layout vs DJI's (≈ Rx(90°)).
- **`C_UP_FORMAT`** `[MEASURED]` — IMU-body → equirect bridge with the
  per-unit mount **factored out**; the pipeline recomposes
  `bridge = C_UP_FORMAT @ mount` from each file. det = −1 by design (the
  equirect z-convention is left-handed vs the physical IMU frame).
- **`QUAT_ORDER = "wxyz"`, `WORLD_UP = +Z`** `[CONVENTION]` — verified via
  gravity.
- **`SEAM_OVERLAP_RATIO = 0.038`** `[MEASURED]` — DJI's blend-band width;
  a constant here so behaviour never depends on DJI Studio being installed.
- **`RS_SIGMA = 15`** `[ASSUMED]` — trajectory smoother width; lands at or
  below DJI's rendered jitter, but DJI's true filter shape is unknown.
- **RS_… rolling-shutter values** `[ASSUMED — never successfully fit]` — the
  RS subsystem ships disabled; see [levelling](levelling.md).
- **`GPU_ENGINES`** — per-card NVENC/NVDEC engine counts, the dial to turn
  when the pipeline meets a new GPU: one HEVC session saturates one NVENC
  engine, so `convert()` runs `nvenc` parallel sessions by default, and
  `nvdec` records whether multi-clip batching can pay. The V100 row is
  `[MEASURED]` (utilization logs); other rows are `[ASSUMED]` from NVIDIA's
  Video Codec SDK matrix — verify with `nvidia-smi` utilization during a
  convert before trusting. Unknown cards get the conservative
  single-session default; add a row to speed a new card up.
- **Pipeline concurrency knobs** `[ASSUMED]` — chunk seconds, queue depths
  and pinned-slot counts for the threaded CUDA pipeline, sized on the V100
  box (see [architecture](architecture.md)).

The unit tests in `tests/test_pipeline.py` pin the mathematical properties of
these constants (orthogonality, the det = −1 reflection, the rocksteady
cancellation, the horizon-lock pole property).
