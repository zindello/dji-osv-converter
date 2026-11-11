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
- ✅ seam repair (`seam_flow="auto"`, the default — `docs/seam.md`):
  photometric match + band-limited flow morph + parallax-gated dynamic
  seam, each detector-gated per frame; fixes parallax steps, occlusion
  dissolve and asymmetric-lighting mismatch at the seam

## Next

1. **More cameras**: verify the `[MEASURED]` constants (see
   `docs/constants.md`) hold across units as new OSV samples arrive; per-unit
   data (calibration, mount) is already read per file.
1b. **Seam CPU work → CUDA builder stage**: the auto seam repair adds
   ~65 ms of CPU on the stitch thread; it already rides
   `stitch_yuv_async`, so moving flow/DP/estimator into the V100
   pipeline's builder stage recovers its throughput. Re-measure both
   benchmark rows under merged defaults afterwards. Measured per
   component on an M3 (`IMPROVEMENTS.md` §8): DIS pair 11.2 ms (now
   two-threaded: ~6), seam DP 9.8, estimator 4.6, gate 2.2, grays 1.3 —
   ~29 ms, i.e. ~65 ms on the Ivy Bridge host, above the 36 ms NVDEC
   wall; the builder-stage move (or a cheaper estimator at 1/8 pooling)
   is what recovers it.
1c. **Single-pass resample on the V100** (`single_pass=True`,
   `IMPROVEMENTS.md` §2): measure its per-frame cost under the NVDEC wall
   and the DJI comparison on that render; if it stays hidden, make it the
   default there. The analytic-uv fused kernel (no map gathers) is the
   next step if it does not.
2. **Beyond the NVDEC wall** — segment-parallel NVENC landed
   (`enc_sessions`, 310.9 → 200.7 s on the full base clip) and the V100
   now runs its single decode engine at 94-100%. Device-frame NVENC
   encode was **prototyped and measured — no PyAV fork needed** (stock
   PyAV passes `cuda`-format frames through to NVENC; frames are minted
   by decoding a tiny generated clip with `is_hw_owned=True` and
   torch-writing their device pointers), but on Volta it LOSES 2.7×:
   the per-frame `nvEncMapInputResource` cost exceeds the saved upload,
   and encoder-pool frames (what a fork would add) ride the same
   register/map path in nvenc.c. Run `tools/hwframe_encode_probe.py`
   on any newer GPU (Turing+ maps cheaply) — if it reports a win, wire
   the mint into `convert()`'s builder stage. The other lever is
   multi-clip batching on multi-NVDEC cards (Ampere+ have 2-5 decode
   engines; the V100's single engine caps aggregate throughput no
   matter the arrangement). Floors underneath: ~25 ms NVDEC pair,
   ~10.5 ms GPU compute.

## Done (2026-08-25): CUDA validation + optimization on the V100

The stitch itself ran unmodified (bit-parity with torch-CPU verified on
real frames) — all the work was video I/O, each step verified
bit-identical, landing at **46 ms/frame e2e (22 fps), 1.6× the M3**:
a platform-gated hwaccel probe (the VideoToolbox probe silently
gave single-threaded software decode on Linux: 590 ms/frame),
decode-to-device NVDEC (`primary_ctx` + `is_hw_owned`: frames stay on
the GPU in torch's context, 25 ms/pair — the generic hwaccel path left
the engine 85% idle; `hevc_cuvid` + p010 ingest remains as the fallback
tier; CUDA decode lives in `nvdec.py`), async D2H through pinned slots
on a side stream, GIL-releasing torch copies in the encoder thread
(numpy's 88 MB/frame memcpy held the GIL and throttled kernel launches:
e2e 95 → 53 from that alone), and a three-stage CUDA pipeline (stitch →
frame-build → encode+mux: 53 → 46; the ~35 ms NVENC call is mostly
libav's internal pageable re-upload, hence the device-frame-encode item
above). Output validated against DJI per `docs/validation.md`; NVENC
auto-selected and verified (hvc1 + spherical metadata + audio). Numbers
in `docs/benchmarks.md`.

## History

The reverse-engineering journey — the f-theta model, the metadata fields, the
levelling conventions, the stabilisation composition — is documented in
`docs/osv-format.md`, `docs/levelling.md` and `docs/validation.md`.
