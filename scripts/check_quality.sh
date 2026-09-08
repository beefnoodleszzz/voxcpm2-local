#!/bin/zsh
set -euo pipefail
PROJECT_DIR="${0:A:h:h}"
cd "$PROJECT_DIR"
ruff check src scripts tests
ruff format --check src scripts tests
"$PROJECT_DIR/.venv/bin/python" -m pytest -q
qc_workdir=$(mktemp -d)
"$PROJECT_DIR/.venv/bin/python" scripts/prepare_episode.py configs/episode.source.example.json "$qc_workdir/prepared.json"
"$PROJECT_DIR/.venv/bin/python" scripts/batch_dub.py "$qc_workdir/prepared.json" --dry-run
