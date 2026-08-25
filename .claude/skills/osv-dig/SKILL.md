---
name: osv-dig
description: Explore unknown fields in the OSV container — how to dig protobuf wire metadata, find streams by content, and correlate candidate fields against ground truth.
---

# Digging the OSV format

When behaviour can't be explained, the answer has repeatedly been *in the
OSV*. `docs/osv-format.md` is the decoded-field ledger — update it with
every new finding, including negative results.

## How to dig

- `protobuf_wire.py` parses any sample schema-lessly into (field-number,
  wire-type, value) trees. Field numbers are ground truth; our names are
  guesses — record both.
- **Find streams by content, never by index** — DJI renumbers between
  firmware versions. Iterate all data streams and test for the structure you
  want (e.g. `read_highrate` probes for 3.3.2.1 and validates
  `hi[f][0] ≈ quats[f]`; stream 4's quats are NaN decoys).
- Dump the same field across many frames (and across clips) before naming
  it: constants, monotonic counters, and scene-correlated signals look alike
  in one sample.
- Correlate candidates against ground truth you already trust: the per-frame
  quats (3.2.9), DJI's own export, EXIF-style header strings, physics
  (gravity magnitude, shutter vs blur).
- Unknowns worth revisiting are listed in docs/osv-format.md (fields 2.6.N
  13/14/21/24/25/27, 3.3.2.1.4).

## Hard-won facts (do not relearn)

- Quats are wxyz, frame START, world-up +Z, drift-free enough to level with.
- 3.2.10 "gravity" carries translational acceleration — never level from it.
- Calibration k5 lives in field 15, apart from k1–k4 (fields 5–8) — a
  four-term read produces a subtly wrong, non-monotonic lens curve.
- The fisheyes are ISP-processed in camera (devignetted, exposure-matched):
  the factory LSC curve (22/23) describes the *sensor*, not the delivered
  pixels.
- Header 1.3.1 is the per-unit camera↔IMU mount rotation (~13° on both
  sample units) — required for all IMU-derived directions, read per file.
