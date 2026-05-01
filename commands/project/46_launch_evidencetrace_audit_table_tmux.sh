#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
SESSION="evidencetrace_audit_20260501"
OUT_DIR="${ROOT}/runs/evidencetrace_audit_table_20260501/30_summary"
LOG_DIR="${ROOT}/runs/evidencetrace_audit_table_20260501/logs"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[info] tmux session already exists: ${SESSION}"
  echo "[info] attach with: tmux attach -t ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "bash -lc '
    set -euo pipefail
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate openpi 2>/dev/null || true
    echo \"[stage] EvidenceTrace audit started: \$(date)\"
    python scripts/benchmark_evidencetrace_audit_methods.py \
      --input BridgeDataV2=runs/EvidenceTrace-VLA/bridge_full/governance/release_hq_trace.jsonl \
      --input Fractal=runs/EvidenceTrace-VLA/fractal_official_full/governance/release_hq_trace.jsonl \
      --input LIBERO10=runs/EvidenceTrace-VLA/libero10_full_all/governance/release_hq_trace.jsonl \
      --input LIBERO-Goal=runs/EvidenceTrace-VLA/libero_goal_full_all/governance/release_hq_trace.jsonl \
      --input LIBERO-Object=runs/EvidenceTrace-VLA/libero_object_full_all/governance/release_hq_trace.jsonl \
      --input LIBERO-Spatial=runs/EvidenceTrace-VLA/libero_spatial_full_all/governance/release_hq_trace.jsonl \
      --input RoboTurk=runs/EvidenceTrace-VLA/roboturk_official_full/governance/release_hq_trace.jsonl \
      --input UT-Austin-MUTEX=runs/EvidenceTrace-VLA/utaustin_mutex/governance/release_hq_trace.jsonl \
      --output_dir \"${OUT_DIR}\" \
      --sample_per_dataset 12000 \
      --seed 20260501 \
      2>&1 | tee \"${LOG_DIR}/audit.log\"
    echo \"[done] EvidenceTrace audit finished: \$(date)\"
    echo \"[output] ${OUT_DIR}\"
    exec bash
  '"

echo "[ok] started tmux session: ${SESSION}"
echo "[ok] attach: tmux attach -t ${SESSION}"
echo "[ok] log: ${LOG_DIR}/audit.log"
