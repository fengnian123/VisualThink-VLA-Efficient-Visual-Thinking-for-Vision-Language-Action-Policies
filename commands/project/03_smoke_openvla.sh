#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs
require_dir "$VLA_PATH"

python "$OPENVLA_ROOT/scripts/run_openvla_inference_smoke.py" \
  --model_path "$VLA_PATH" \
  --attn_impl sdpa \
  2>&1 | tee "$WORK_ROOT/logs/03_smoke_openvla.log"

echo "[ok] log=$WORK_ROOT/logs/03_smoke_openvla.log"
