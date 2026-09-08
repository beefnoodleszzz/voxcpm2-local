#!/bin/zsh
set -euo pipefail
PROJECT_DIR="${0:A:h:h}"
cd "$PROJECT_DIR"
export VOXCPM_MODEL_PATH="$PROJECT_DIR/models/VoxCPM2-bf16"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
if [[ "${VOXCPM_OFFLINE:-1}" == "1" ]]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
exec "$PROJECT_DIR/.venv/bin/uvicorn" src.server:app --host 127.0.0.1 --port 7861

