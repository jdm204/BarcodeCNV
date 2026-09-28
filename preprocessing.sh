#!/usr/bin/env bash
# Thin entry point; all input validation and orchestration live in Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    exec "$ROOT/.venv/bin/python" -m barcodecnv.cli preprocess "$@"
fi
exec uv run --project "$ROOT" --locked python -m barcodecnv.cli preprocess "$@"
