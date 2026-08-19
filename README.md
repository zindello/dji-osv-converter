# DJI CLI Reframe

A CLI tool to convert DJI OSV (Object Stabilization Vector) files to DaVinci Resolve format, bypassing the need for the macOS GUI app.

## Goal

Replicate the functionality of DJI Reframe (macOS app) to enable:
- CLI conversion of raw OSV files from DJI cameras → DaVinci Resolve format
- Batch processing of video files
- Optional GUI wrapper (drag-drop interface)

## Project Status

**Phase 1: Exploration & Reverse Engineering**
- [ ] Understand OSV file format structure
- [ ] Document DaVinci Resolve XML/import format requirements
- [ ] Inspect DJI Reframe app internals (binary, frameworks, resources)
- [ ] Map the conversion algorithm
- [ ] Identify dependencies (libraries, frameworks)

## Key Questions

1. **OSV Format**: What's the exact structure of DJI OSV files? (binary/XML/JSON?)
2. **Resolve Format**: What XML/data format does DaVinci Resolve expect?
3. **The Conversion**: What's the mathematical transformation between the two?
4. **Dependencies**: Does DJI Reframe use private DJI frameworks or standard macOS APIs?

## Resources

- DJI Reframe app location: `/Applications/DJI Reframe.app`
- DaVinci Resolve documentation: https://www.blackmagicdesign.com/
- Relevant tools: `otool`, `strings`, `xxd` for macOS binary inspection
