#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

MANIFEST="$WORK_ROOT/01_extract/manifest.jsonl"
require_file "$MANIFEST"
OUT_DIR="$WORK_ROOT/03_features"
mkdir -p "$OUT_DIR"
TOTAL_ROWS="$(wc -l < "$MANIFEST")"
if [[ "$FEATURE_LIMIT" != "0" ]]; then
  TARGET_ROWS="$FEATURE_LIMIT"
else
  TARGET_ROWS="$TOTAL_ROWS"
fi

SHARD_ROOT="$OUT_DIR/shards"
TMP_ROOT="$OUT_DIR/tmp_manifests"
mkdir -p "$SHARD_ROOT" "$TMP_ROOT"
python - <<PY
from pathlib import Path
import shutil
for path in [Path("$SHARD_ROOT"), Path("$TMP_ROOT")]:
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
out_manifest = Path("$OUT_DIR/feature_manifest.jsonl")
if out_manifest.exists():
    out_manifest.unlink()
PY

detect_gpu_ids() {
  if [[ "$FEATURE_GPU_IDS" != "auto" ]]; then
    echo "$FEATURE_GPU_IDS"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  python - <<PY
import subprocess
min_free = int("$FEATURE_GPU_MIN_FREE_MB")
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True,
)
ids = []
for line in out.strip().splitlines():
    idx_s, free_s = [x.strip() for x in line.split(",")]
    if int(free_s) >= min_free:
        ids.append(idx_s)
print(",".join(ids))
PY
}

GPU_IDS="$(detect_gpu_ids)"
if [[ -z "$GPU_IDS" ]]; then
  echo "[warn] no sufficiently free GPUs detected; falling back to a single visible device" >&2
  GPU_IDS="0"
fi
IFS=',' read -r -a GPU_ID_ARR <<< "$GPU_IDS"
NUM_SHARDS="${#GPU_ID_ARR[@]}"
echo "[info] target_rows=$TARGET_ROWS gpu_ids=$GPU_IDS num_shards=$NUM_SHARDS"

python - <<PY
import json
from pathlib import Path

manifest = Path("$MANIFEST")
tmp_root = Path("$TMP_ROOT")
num_shards = int("$NUM_SHARDS")
limit = int("$TARGET_ROWS")

for path in tmp_root.glob("shard_*.jsonl"):
    path.unlink()

files = [open(tmp_root / f"shard_{i:02d}.jsonl", "w", encoding="utf-8") for i in range(num_shards)]
counts = [0 for _ in range(num_shards)]
episode_to_shard = {}
next_shard = 0

try:
    written = 0
    with manifest.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            if written >= limit:
                break
            row = json.loads(line)
            episode_idx = int(row["episode_idx"])
            if episode_idx not in episode_to_shard:
                episode_to_shard[episode_idx] = next_shard
                next_shard = (next_shard + 1) % num_shards
            shard_idx = episode_to_shard[episode_idx]
            files[shard_idx].write(line)
            counts[shard_idx] += 1
            written += 1
finally:
    for f in files:
        f.close()

for i, c in enumerate(counts):
    print(f"[info] shard_{i:02d}_rows={c}")
PY

WORKER_PIDS=()
WORKER_LOGS=()
SHARD_MANIFESTS=()
for idx in "${!GPU_ID_ARR[@]}"; do
  gpu_id="${GPU_ID_ARR[$idx]}"
  shard_manifest="$TMP_ROOT/shard_$(printf '%02d' "$idx").jsonl"
  shard_rows="$(wc -l < "$shard_manifest")"
  if [[ "$shard_rows" == "0" ]]; then
    continue
  fi
  shard_out="$SHARD_ROOT/gpu_${gpu_id}"
  mkdir -p "$shard_out"
  worker_log="$WORK_ROOT/logs/06_batch_features.gpu_${gpu_id}.log"
  WORKER_LOGS+=("$worker_log")
  SHARD_MANIFESTS+=("$shard_out/feature_manifest.jsonl")

  CMD=(
    python -u "$OPENVLA_ROOT/scripts/batch_extract_features.py"
    --manifest "$shard_manifest"
    --output_dir "$shard_out"
    --qwen-model-id "$QWEN_MODEL_ID"
    --qwen-image-edit-model-id "$QWEN_IMAGE_EDIT_MODEL_ID"
    --qwen-image-edit-api-url "$QWEN_IMAGE_EDIT_API_URL"
    --qwen-image-edit-api-key "$QWEN_IMAGE_EDIT_API_KEY"
    --owl-model-id "$OWL_MODEL_ID"
    --limit "$shard_rows"
    --log_every "$FEATURE_LOG_EVERY"
    --caption_batch_size "$FEATURE_CAPTION_BATCH_SIZE"
  )
  if [[ -n "$QUERY_API_URL" ]]; then
    CMD+=(--query_api_url "$QUERY_API_URL")
  fi
  if [[ -n "$QUERY_API_KEY" ]]; then
    CMD+=(--query_api_key "$QUERY_API_KEY")
  fi
  if [[ "$ENABLE_QWEN" != "1" ]]; then
    CMD+=(--disable_qwen)
  fi
  if [[ "$ENABLE_QWEN_IMAGE_EDIT" != "1" ]]; then
    CMD+=(--disable_qwen_image_edit)
  fi
  if [[ "$ENABLE_OWL" != "1" ]]; then
    CMD+=(--disable_owl)
  fi
  if [[ "$ENABLE_SAM2" != "1" ]]; then
    CMD+=(--disable_sam2)
  fi
  if [[ "$ENABLE_MIDAS" != "1" ]]; then
    CMD+=(--disable_midas)
  fi
  if [[ "$FEATURE_SAVE_DEBUG_VISUALS" == "1" ]]; then
    CMD+=(--save_debug_visuals --debug_visual_limit "$FEATURE_DEBUG_VISUAL_LIMIT")
  fi

  echo "[info] launch worker gpu=$gpu_id rows=$shard_rows log=$worker_log"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${CMD[@]}" >"$worker_log" 2>&1 &
  WORKER_PIDS+=("$!")
done

if [[ "${#WORKER_PIDS[@]}" == "0" ]]; then
  echo "No feature workers launched" >&2
  exit 1
fi

start_ts="$(date +%s)"
while :; do
  alive=0
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive=1
      break
    fi
  done

  done_rows=0
  for mf in "${SHARD_MANIFESTS[@]}"; do
    if [[ -f "$mf" ]]; then
      done_rows="$((done_rows + $(wc -l < "$mf")))"
    fi
  done
  now_ts="$(date +%s)"
  elapsed="$((now_ts - start_ts))"
  if [[ "$elapsed" -le 0 ]]; then
    elapsed=1
  fi
  speed="$(python - <<PY
done_rows = int("$done_rows")
elapsed = int("$elapsed")
print(f"{done_rows / elapsed:.2f}")
PY
)"
  remain="$((TARGET_ROWS - done_rows))"
  if [[ "$remain" -lt 0 ]]; then
    remain=0
  fi
  eta="$(python - <<PY
done_rows = int("$done_rows")
elapsed = int("$elapsed")
target = int("$TARGET_ROWS")
speed = done_rows / elapsed if elapsed > 0 else 0.0
remain = max(0, target - done_rows)
print(int(remain / speed) if speed > 0 else -1)
PY
)"
  echo "[progress] total=${done_rows}/${TARGET_ROWS} speed=${speed} samples/s eta=${eta}s"

  if [[ "$alive" == "0" ]]; then
    break
  fi
  sleep "$FEATURE_MONITOR_INTERVAL_SEC"
done

for i in "${!WORKER_PIDS[@]}"; do
  pid="${WORKER_PIDS[$i]}"
  log="${WORKER_LOGS[$i]}"
  if ! wait "$pid"; then
    echo "[error] feature worker failed; tailing $log" >&2
    tail -n 80 "$log" >&2 || true
    exit 1
  fi
done

python - <<PY
import json
from pathlib import Path
manifests = sorted(Path("$SHARD_ROOT").glob("gpu_*/feature_manifest.jsonl"))
rows = []
for mf in manifests:
    if not mf.exists():
        continue
    with mf.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
rows.sort(key=lambda r: (int(r["episode_idx"]), int(r["step_idx"])))
out_path = Path("$OUT_DIR/feature_manifest.jsonl")
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\\n")
print(f"[ok] merged_rows={len(rows)}")
print(f"[ok] manifest={out_path}")
PY
