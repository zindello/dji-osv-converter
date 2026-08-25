#!/bin/sh
# Regenerate the API reference (docs/api) from the code's docstrings.
# Run automatically by .githooks/pre-commit; also fine to run by hand.
set -e
cd "$(git rev-parse --show-toplevel)"
[ -x .venv/bin/pdoc ] && PDOC=.venv/bin/pdoc || PDOC=pdoc
$PDOC dji_osv_converter -o docs/api --no-show-source \
    --footer-text "generated from source by pdoc (pre-commit hook)"
