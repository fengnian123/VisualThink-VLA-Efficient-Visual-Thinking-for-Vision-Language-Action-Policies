#!/usr/bin/env python3
"""Prompt-only channel screening for dense textual evidence candidates."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import unpack_edge  # noqa: E402
from utils.relation_features import relation_text  # noqa: E402


VARIANT_ORDER = (
    "original",
    "bbox_only",
    "edge_only",
    "motion_only",
    "relation_only",
    "depth_only",
    "caption_query_only",
    "structured_no_text",
    "full_schema",
)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def choose_indices(total: int, take: int, seed: int, strategy: str) -> list[int]:
    if take <= 0 or take >= total:
        return list(range(total))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), take))
    if strategy == "stride":
        return sorted(set(int(round(x)) for x in np.linspace(0, total - 1, num=take)))
    raise ValueError(f"unknown strategy: {strategy}")


def action_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def prompt_from_schema(instruction: str, schema_text: str) -> str:
    q = f"{instruction}\nVisual evidence:\n{schema_text}\nReturn robot action."
    return f"In: What action should the robot take to {q}?\nOut:"


def predict_action(vla, processor, image: Image.Image, prompt: str, device: str, dtype, unnorm_key: str):
    inputs = processor(prompt, image).to(device, dtype=dtype)
    start = time.perf_counter()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    elapsed = time.perf_counter() - start
    return np.asarray(action, dtype=np.float32), elapsed


def load_dense(npz_path: str) -> dict[str, object]:
    with np.load(npz_path, allow_pickle=True) as npz:
        depth = npz["depth_u16"].astype(np.float32) / 65535.0 if "depth_u16" in npz.files else None
        edge = unpack_edge(npz["edge_packed"], npz["edge_shape"]) if "edge_packed" in npz.files else None
        motion = npz["motion_u8"].astype(np.uint8) if "motion_u8" in npz.files else None
    return {"depth": depth, "edge": edge, "motion": motion}


def bbox_snippet(row: dict) -> str:
    dets = row.get("detections") or []
    if not dets:
        return "detections: none"
    pieces = []
    for i, det in enumerate(dets[:8]):
        x1, y1, x2, y2 = det["bbox"]
        pieces.append(
            f"obj{i}:{det['label']} score={det['score']:.2f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )
    return "detections: " + "; ".join(pieces)


def edge_snippet(edge: np.ndarray | None) -> str:
    if edge is None:
        return "edge: none"
    edge_bin = (edge > 0).astype(np.float32)
    return f"edge: density={float(edge_bin.mean()):.4f}"


def motion_snippet(motion: np.ndarray | None) -> str:
    if motion is None:
        return "motion: none"
    motion_f = motion.astype(np.float32) / 255.0
    return (
        "motion: "
        f"mean={float(motion_f.mean()):.4f},std={float(motion_f.std()):.4f},"
        f"density={float((motion_f > 0.08).mean()):.4f}"
    )


def relation_snippet(row: dict) -> str:
    return "relation: " + relation_text(row.get("relation_stats"))


def depth_snippet(depth: np.ndarray | None) -> str:
    if depth is None:
        return "depth: none"
    return (
        "depth: "
        f"mean={float(depth.mean()):.4f},std={float(depth.std()):.4f},"
        f"min={float(depth.min()):.4f},max={float(depth.max()):.4f}"
    )


def caption_query_snippet(row: dict) -> str:
    query_words = ", ".join(row.get("query_words") or [])
    return f"caption: {row.get('caption', '')}\nqueries: {query_words if query_words else 'none'}"


def build_schema_text(variant: str, row: dict, dense: dict[str, object]) -> str | None:
    if variant == "original":
        return None
    if variant == "bbox_only":
        return bbox_snippet(row)
    if variant == "edge_only":
        return edge_snippet(dense["edge"])
    if variant == "motion_only":
        return motion_snippet(dense["motion"])
    if variant == "relation_only":
        return relation_snippet(row)
    if variant == "depth_only":
        return depth_snippet(dense["depth"])
    if variant == "caption_query_only":
        return caption_query_snippet(row)
    if variant == "structured_no_text":
        return "\n".join(
            [
                bbox_snippet(row),
                edge_snippet(dense["edge"]),
                motion_snippet(dense["motion"]),
                relation_snippet(row),
            ]
        )
    if variant == "full_schema":
        return str(row.get("schema_text", "")).strip()
    raise ValueError(f"unknown variant: {variant}")


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out = []
    baseline_success = None
    if "original" in grouped:
        baseline_success = float(np.mean([1.0 if r["success"] else 0.0 for r in grouped["original"]]))
    for variant in VARIANT_ORDER:
        items = grouped.get(variant)
        if not items:
            continue
        success = float(np.mean([1.0 if r["success"] else 0.0 for r in items]))
        latency = float(np.mean([float(r["inference_time_s"]) for r in items]))
        avg_l1 = float(np.mean([float(r["l1"]) for r in items]))
        out.append(
            {
                "variant": variant,
                "n": len(items),
                "success_rate": success,
                "dense_gain_vs_original": None if baseline_success is None else success - baseline_success,
                "avg_l1": avg_l1,
                "avg_step_latency_s": latency,
                "inference_fps": float(1.0 / max(1e-6, latency)),
            }
        )
    return out


def write_summary(output_dir: Path, summary_rows: list[dict]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| Variant | N | Success | Dense gain | Avg L1 | Latency(s) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        gain = row["dense_gain_vs_original"]
        gain_text = "--" if gain is None else f"{gain:+.4f}"
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['success_rate']:.4f} | {gain_text} | "
            f"{row['avg_l1']:.4f} | {row['avg_step_latency_s']:.4f} | {row['inference_fps']:.4f} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=384)
    parser.add_argument("--sample_strategy", choices=("random", "stride"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.feature_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(manifest_path)
    indices = choose_indices(len(rows), args.sample_count, args.seed, args.sample_strategy)
    selected_rows = [rows[i] for i in indices]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    result_rows: list[dict] = []
    total = len(selected_rows) * len(VARIANT_ORDER)
    done = 0
    for idx, row in enumerate(selected_rows):
        dense = load_dense(row["npz_path"])
        image = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(row["action"], dtype=np.float32)
        try:
            for variant in VARIANT_ORDER:
                schema_text = build_schema_text(variant, row, dense)
                if schema_text is None:
                    prompt = f"In: What action should the robot take to {row['instruction'].lower()}?\nOut:"
                else:
                    prompt = prompt_from_schema(row["instruction"], schema_text)
                action, elapsed = predict_action(vla, processor, image, prompt, device, dtype, args.unnorm_key)
                l1 = action_l1(action, gt)
                result_rows.append(
                    {
                        "dataset": args.dataset,
                        "variant": variant,
                        "source_idx": indices[idx],
                        "episode_idx": int(row["episode_idx"]),
                        "step_idx": int(row["step_idx"]),
                        "success": bool(l1 <= args.success_l1_thresh),
                        "l1": l1,
                        "inference_time_s": elapsed,
                    }
                )
                done += 1
        finally:
            image.close()
        if (idx + 1) % 10 == 0 or (idx + 1) == len(selected_rows):
            print(f"[progress] rows={idx + 1}/{len(selected_rows)} preds={done}/{total}", flush=True)

    (output_dir / "eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result_rows),
        encoding="utf-8",
    )
    summary_rows = summarize(result_rows)
    write_summary(output_dir, summary_rows)
    meta = {
        "dataset": args.dataset,
        "feature_manifest": str(manifest_path),
        "sample_count": len(selected_rows),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "source_indices_preview": indices[:16],
    }
    (output_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
