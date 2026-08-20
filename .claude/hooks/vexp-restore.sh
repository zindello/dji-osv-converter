#!/bin/bash
# vexp-restore: context lifecycle restore on SessionStart (compact/resume). Fails open.
VEXP_BIN="/Users/zindello/.vscode/extensions/vexp.vexp-vscode-2.6.3-darwin-arm64/binaries/vexp-core-darwin-arm64/vexp-core"
[ -x "$VEXP_BIN" ] || exit 0
"$VEXP_BIN" session-context 2>/dev/null
exit 0
