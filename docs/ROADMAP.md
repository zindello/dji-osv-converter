# Roadmap

**Status: 2026-08-25.** The converter is feature-complete against DJI Studio
for the recorded modes and validated on worst-case footage:

- ✅ `plain` — matches DJI's default export
- ✅ `rocksteady` — smoother than DJI (0.027/0.071 vs 0.046/0.118 °/frame)
- ✅ `rocksteady_horizon` — all-axis lock, steadier yaw than DJI
- ✅ colour path — clean (the OSV arrives ISP-corrected; our double-corrections
  are off), ~matches DJI apart from DJI's stylistic contrast curve
- ✅ 8K 10-bit HEVC output with audio + spherical metadata (hvc1)
- ✅ ~60 ms/frame stitch, 73 ms/frame end-to-end on Apple M3 (13.6 fps)
- ✅ concurrency: decode/stitch/encode pipelined on GPU by default
  (104 → 73 ms/frame on MPS; output verified frame-identical to serial).
  CPU concurrency measured as a regression in every variant — CPU stays
  serial (see `docs/benchmarks.md` and the notes in `video.py`)

## Next

1. **CUDA validation** on discrete GPUs (V100/P100) — the stitch is already
   device-agnostic torch and the pipeline auto-enables; needs profiling,
   NVENC, and a benchmark row.
2. **More cameras**: verify the `[MEASURED]` constants (see
   `docs/constants.md`) hold across units as new OSV samples arrive; per-unit
   data (calibration, mount) is already read per file.

## History

The reverse-engineering journey — the f-theta model, the metadata fields, the
levelling conventions, the stabilisation composition — is documented in
`docs/osv-format.md`, `docs/levelling.md` and `docs/validation.md`.
