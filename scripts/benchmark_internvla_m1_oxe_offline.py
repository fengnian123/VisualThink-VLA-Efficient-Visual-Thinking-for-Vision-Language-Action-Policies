#!/usr/bin/env python3
"""Offline OXE-style action benchmark for InternVLA-M1 checkpoints.

The available InternVLA-M1 checkpoints in this workspace are LIBERO-specialized
and expose a single `franka` normalization key. Cross-dataset numbers are
therefore compatibility diagnostics, not official benchmark success rates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def add_internvla_root(internvla_root: Path) -> None:
    if not (internvla_root / "InternVLA").exists():
        raise FileNotFoundError(f"InternVLA package not found: {internvla_root / 'InternVLA'}")
    sys.path.insert(0, str(internvla_root))


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
    dims = min(pred.shape[-1], target.shape[-1])
    return float(np.mean(np.abs(pred[..., :dims] - target[..., :dims])))


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict) -> np.ndarray:
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)), dtype=bool)
    action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
    normalized_actions = np.clip(normalized_actions, -1, 1)
    if normalized_actions.shape[-1] >= 7:
        normalized_actions[..., 6] = np.where(normalized_actions[..., 6] < 0.5, 0, 1)
    return np.where(mask, 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low, normalized_actions)


def summarize(rows: list[dict]) -> dict:
    success = np.asarray([1.0 if r["success"] else 0.0 for r in rows], dtype=np.float32)
    latency = np.asarray([float(r["inference_time_s"]) for r in rows], dtype=np.float32)
    l1 = np.asarray([float(r["l1"]) for r in rows], dtype=np.float32)
    return {
        "n": len(rows),
        "success_rate_at_l1_threshold": float(success.mean()) if len(rows) else 0.0,
        "avg_step_latency_s": float(latency.mean()) if len(rows) else 0.0,
        "inference_fps": float(np.mean(1.0 / np.maximum(latency, 1e-6))) if len(rows) else 0.0,
        "avg_l1": float(l1.mean()) if len(rows) else 0.0,
        "median_l1": float(np.median(l1)) if len(rows) else 0.0,
    }


def write_outputs(dataset_dir: Path, eval_rows: list[dict], summary: dict, all_summary: dict, output_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "internvla_m1_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eval_rows),
        encoding="utf-8",
    )
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "| Dataset | N | L1-threshold rate | Avg. latency (s) | FPS | Avg. L1 | Median L1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in all_summary.items():
        lines.append(
            f"| {name} | {row['n']} | {row['success_rate_at_l1_threshold']:.4f} | "
            f"{row['avg_step_latency_s']:.4f} | {row['inference_fps']:.4f} | "
            f"{row['avg_l1']:.4f} | {row['median_l1']:.4f} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internvla_root", default="models/local/InternVLA-M1")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", nargs=3, action="append", metavar=("NAME", "FEATURE_MANIFEST", "UNNORM_NOTE"), required=True)
    parser.add_argument("--unnorm_key", default="franka")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--num_ddim_steps", type=int, default=10)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--skip_empty_instruction", action="store_true")
    args = parser.parse_args()

    add_internvla_root(Path(args.internvla_root))
    from InternVLA.model.framework.M1 import InternVLA_M1
    from InternVLA.model.framework.share_tools import read_mode_config

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading InternVLA-M1 checkpoint={args.checkpoint}", flush=True)
    _, norm_stats = read_mode_config(args.checkpoint)
    if args.unnorm_key not in norm_stats:
        raise RuntimeError(f"unnorm_key={args.unnorm_key!r} not found. Available keys: {sorted(norm_stats.keys())}")
    action_norm_stats = norm_stats[args.unnorm_key]["action"]

    model = InternVLA_M1.from_pretrained(args.checkpoint)
    model = model.to(torch.bfloat16).to("cuda").eval()

    all_summary: dict[str, dict] = {}
    for dataset_name, manifest, unnorm_note in args.dataset:
        rows = load_rows(Path(manifest), args.limit, args.skip_empty_instruction)
        dataset_dir = output_dir / dataset_name
        eval_rows: list[dict] = []
        for idx, row in enumerate(rows):
            print(f"[{dataset_name}] starting {idx + 1}/{len(rows)} image={row.get('image_path')}", flush=True)
            image = Image.open(row["image_path"]).convert("RGB")
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.predict_action(
                    batch_images=[[image]],
                    instructions=[row.get("instruction") or ""],
                    do_sample=False,
                    use_ddim=True,
                    num_ddim_steps=args.num_ddim_steps,
                )
            elapsed = time.perf_counter() - start
            normalized = np.asarray(output["normalized_actions"], dtype=np.float32)[0]
            pred_chunk = unnormalize_actions(normalized, action_norm_stats)
            pred_action = np.asarray(pred_chunk[0], dtype=np.float32)
            target_action = np.asarray(row["action"], dtype=np.float32)
            l1 = action_l1(pred_action, target_action)
            eval_rows.append(
                {
                    "idx": idx,
                    "source_episode_idx": row.get("episode_idx"),
                    "source_step_idx": row.get("step_idx"),
                    "image_path": row.get("image_path"),
                    "instruction": row.get("instruction") or "",
                    "normalization_note": unnorm_note,
                    "unnorm_key": args.unnorm_key,
                    "prediction": pred_action.tolist(),
                    "target": target_action.tolist(),
                    "l1": l1,
                    "success": l1 <= args.success_l1_thresh,
                    "inference_time_s": elapsed,
                }
            )
            summary = summarize(eval_rows)
            summary.update(
                {
                    "checkpoint": args.checkpoint,
                    "feature_manifest": str(Path(manifest)),
                    "unnorm_key": args.unnorm_key,
                    "normalization_note": unnorm_note,
                    "partial": idx + 1 < len(rows),
                    "benchmark_note": "cross-dataset offline action-error diagnostic; not official simulator success",
                }
            )
            all_summary[dataset_name] = summary
            write_outputs(dataset_dir, eval_rows, summary, all_summary, output_dir)
            print(f"[{dataset_name}] {idx + 1}/{len(rows)} l1={l1:.4f} success={l1 <= args.success_l1_thresh} latency={elapsed:.4f}s", flush=True)

        summary = summarize(eval_rows)
        summary.update(
            {
                "checkpoint": args.checkpoint,
                "feature_manifest": str(Path(manifest)),
                "unnorm_key": args.unnorm_key,
                "normalization_note": unnorm_note,
                "partial": False,
                "benchmark_note": "cross-dataset offline action-error diagnostic; not official simulator success",
            }
        )
        all_summary[dataset_name] = summary
        write_outputs(dataset_dir, eval_rows, summary, all_summary, output_dir)


if __name__ == "__main__":
    main()
