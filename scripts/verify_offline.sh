#!/bin/zsh
set -euo pipefail
PROJECT_DIR="${0:A:h:h}"
cd "$PROJECT_DIR"
export VOXCPM_MODEL_PATH="$PROJECT_DIR/models/VoxCPM2-bf16"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_PROXY="*"
OUTPUT="$PROJECT_DIR/outputs/smoke/offline_$(date +%Y%m%d_%H%M%S).wav"
"$PROJECT_DIR/.venv/bin/python" scripts/smoke_test.py --steps 10 --output "$OUTPUT"
test -s "$OUTPUT"
echo "OFFLINE TTS VERIFIED"

