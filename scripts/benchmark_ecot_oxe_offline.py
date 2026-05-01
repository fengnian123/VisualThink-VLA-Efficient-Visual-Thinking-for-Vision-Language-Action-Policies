#!/usr/bin/env python3
"""Offline action-prediction benchmark for ECoT/OpenVLA OXE checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor


OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def register_fast_ecot_hf_classes(fast_ecot_root: Path) -> None:
    """Use Fast-ECoT's HF wrapper so ECoT generation returns action and token ids."""

    if not fast_ecot_root.exists():
        raise FileNotFoundError(f"Fast-ECoT root not found: {fast_ecot_root}")
    sys.path.insert(0, str(fast_ecot_root))

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig, exist_ok=True)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor, exist_ok=True)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor, exist_ok=True)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction, exist_ok=True)


def make_ecot_prompt(instruction: str) -> str:
    return (
        f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to "
        f"{instruction.lower()}? ASSISTANT: TASK:"
    )


def load_rows(manifest_path: Path, limit: int, skip_empty_instruction: bool) -> list[dict]:
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if skip_empty_instruction and not (row.get("instruction") or "").strip():
                continue
            if not Path(row.get("image_path", "")).exists():
                continue
            rows.append(row)
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"No usable rows selected from {manifest_path}")
    return rows


def action_l1(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        raise ValueError(f"action shape mismatch: pred={pred.shape} target={target.shape}")
    return float(np.mean(np.abs(pred - target)))


def decode_generated(processor, generated_ids) -> str:
    if generated_ids is None:
        return ""
    try:
        return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    except Exception:
        try:
            return processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        except Exception:
            return ""


def summarize(rows: list[dict]) -> dict:
    success = np.asarray([1.0 if r["success"] else 0.0 for r in rows], dtype=np.float32)
    latency = np.asarray([float(r["inference_time_s"]) for r in rows], dtype=np.float32)
    l1 = np.asarray([float(r["l1"]) for r in rows], dtype=np.float32)
    return {
        "n": len(rows),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "avg_step_latency_s": float(latency.mean()) if len(rows) else 0.0,
        "inference_fps": float(np.mean(1.0 / np.maximum(latency, 1e-6))) if len(rows) else 0.0,
        "avg_l1": float(l1.mean()) if len(rows) else 0.0,
        "median_l1": float(np.median(l1)) if len(rows) else 0.0,
    }


def write_markdown(summary_by_dataset: dict[str, dict], output_dir: Path) -> None:
    lines = [
        "| Dataset | N | Success rate | Avg. step latency (s) | FPS | Avg. L1 | Median L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in summary_by_dataset.items():
        lines.append(
            f"| {dataset} | {row['n']} | {row['success_rate']:.4f} | "
            f"{row['avg_step_latency_s']:.4f} | {row['inference_fps']:.4f} | "
            f"{row['avg_l1']:.4f} | {row['median_l1']:.4f} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dataset_outputs(
    dataset_dir: Path,
    eval_rows: list[dict],
    summary: dict,
    summary_by_dataset: dict[str, dict],
    output_dir: Path,
) -> None:
    (dataset_dir / "ecot_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows),
        encoding="utf-8",
    )
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary_by_dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary_by_dataset, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("NAME", "FEATURE_MANIFEST", "UNNORM_KEY"),
        required=True,
        help="Dataset triplet. Can be repeated.",
    )
    parser.add_argument("--fast_ecot_root", default="models/local/Fast-ECoT")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--skip_empty_instruction", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    register_fast_ecot_hf_classes(Path(args.fast_ecot_root))

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[info] loading model={args.model_path} device={device} dtype={dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=False)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_impl,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device)
    model.eval()

    available_keys = set(getattr(model, "norm_stats", {}).keys())
    if not available_keys:
        raise RuntimeError("Loaded model has no norm_stats.")

    summary_by_dataset: dict[str, dict] = {}
    for dataset_name, manifest, unnorm_key in args.dataset:
        if unnorm_key not in available_keys:
            raise RuntimeError(
                f"unnorm_key={unnorm_key!r} is not available for {dataset_name}. "
                f"Available keys: {sorted(available_keys)}"
            )

        dataset_dir = output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        rows = load_rows(Path(manifest), limit=args.limit, skip_empty_instruction=args.skip_empty_instruction)

        eval_rows: list[dict] = []
        for idx, row in enumerate(rows):
            print(
                f"[{dataset_name}] starting {idx + 1}/{len(rows)} "
                f"image={row.get('image_path')}",
                flush=True,
            )
            image = Image.open(row["image_path"]).convert("RGB")
            instruction = row.get("instruction") or ""
            prompt = make_ecot_prompt(instruction)
            inputs = processor(prompt, image).to(device, dtype=dtype)

            start = time.perf_counter()
            with torch.inference_mode():
                pred = model.predict_action(
                    **inputs,
                    unnorm_key=unnorm_key,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )
            elapsed = time.perf_counter() - start

            generated_ids = None
            if isinstance(pred, tuple):
                pred_action, generated_ids = pred
            else:
                pred_action = pred

            pred_action = np.asarray(pred_action, dtype=np.float32)
            target_action = np.asarray(row["action"], dtype=np.float32)
            l1 = action_l1(pred_action, target_action)
            decoded = decode_generated(processor, generated_ids)
            eval_row = {
                "idx": idx,
                "source_episode_idx": row.get("episode_idx"),
                "source_step_idx": row.get("step_idx"),
                "image_path": row.get("image_path"),
                "instruction": instruction,
                "unnorm_key": unnorm_key,
                "prediction": pred_action.tolist(),
                "target": target_action.tolist(),
                "l1": l1,
                "success": l1 <= args.success_l1_thresh,
                "inference_time_s": elapsed,
                "generated_text": decoded,
            }
            eval_rows.append(eval_row)
            print(
                f"[{dataset_name}] {idx + 1}/{len(rows)} "
                f"l1={l1:.4f} success={eval_row['success']} latency={elapsed:.4f}s",
                flush=True,
            )
            partial_summary = summarize(eval_rows)
            partial_summary["unnorm_key"] = unnorm_key
            partial_summary["feature_manifest"] = str(Path(manifest))
            partial_summary["partial"] = idx + 1 < len(rows)
            summary_by_dataset[dataset_name] = partial_summary
            write_dataset_outputs(dataset_dir, eval_rows, partial_summary, summary_by_dataset, output_dir)

        summary = summarize(eval_rows)
        summary["unnorm_key"] = unnorm_key
        summary["feature_manifest"] = str(Path(manifest))
        summary["partial"] = False
        summary_by_dataset[dataset_name] = summary
        write_dataset_outputs(dataset_dir, eval_rows, summary, summary_by_dataset, output_dir)

    (output_dir / "summary.json").write_text(json.dumps(summary_by_dataset, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(summary_by_dataset, output_dir)
    print(f"[ok] summary={output_dir / 'summary_table.md'}", flush=True)


if __name__ == "__main__":
    main()
