#!/usr/bin/env python3
"""Prepare aligned feature/trace subsets for interpretability ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


CHANNELS = ("bbox", "edge", "motion", "relation")


def load_lines(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty jsonl: {path}")
    return lines


def parse_row(line: str) -> dict:
    return json.loads(line)


def row_key(row: dict) -> tuple[object, object, str]:
    return (
        row.get("episode_idx", row.get("episode_id")),
        row.get("step_idx", row.get("step_id")),
        Path(str(row.get("npz_path") or row.get("feature_ref") or "")).name,
    )


def validate_alignment(feature_lines: list[str], trace_lines: list[str]) -> None:
    if len(feature_lines) != len(trace_lines):
        raise RuntimeError(f"feature/trace mismatch: {len(feature_lines)} vs {len(trace_lines)}")
    probe_indices = sorted(set([0, len(feature_lines) // 3, (2 * len(feature_lines)) // 3, len(feature_lines) - 1]))
    for idx in probe_indices:
        feature_row = parse_row(feature_lines[idx])
        trace_row = parse_row(trace_lines[idx])
        if row_key(feature_row) != row_key(trace_row):
            raise RuntimeError(
                f"alignment mismatch at idx={idx}: feature={row_key(feature_row)} trace={row_key(trace_row)}"
            )


def choose_indices(total: int, take: int, seed: int, strategy: str) -> list[int]:
    if take <= 0 or take >= total:
        return list(range(total))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), take))
    if strategy == "stride":
        return sorted(set(int(round(x)) for x in np.linspace(0, total - 1, num=take)))
    raise ValueError(f"unknown strategy: {strategy}")


def write_subset(path: Path, lines: list[str], indices: list[int]) -> None:
    path.write_text("".join(lines[i] + "\n" for i in indices), encoding="utf-8")


def build_freeform_rationale(trace_row: dict) -> str:
    stage = str(trace_row.get("stage", "interaction")).strip() or "interaction"
    primitive = str(trace_row.get("primary_primitive", "manipulation")).strip() or "manipulation"
    instruction = str(trace_row.get("instruction", "")).strip()
    action_intent = str(trace_row.get("action_intent_summary", trace_row.get("action_intent", ""))).strip()
    base = action_intent or instruction or primitive
    if not base:
        base = "complete the current action"
    text = f"At the {stage} stage, the policy should focus on the scene context and instruction to {base}."
    lower = text.lower()
    for channel in CHANNELS:
        if channel in lower:
            text = text.replace(channel, "relevant cue")
            text = text.replace(channel.capitalize(), "Relevant cue")
    return text


def write_freeform_trace(path: Path, trace_lines: list[str], indices: list[int]) -> None:
    out_lines = []
    for idx in indices:
        row = parse_row(trace_lines[idx])
        row["visual_rationale"] = build_freeform_rationale(row)
        out_lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--trace_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_count", type=int, default=4096)
    parser.add_argument("--eval_count", type=int, default=2048)
    parser.add_argument("--faithfulness_count", type=int, default=768)
    parser.add_argument("--sample_strategy", choices=("random", "stride"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    feature_src = Path(args.feature_manifest)
    trace_src = Path(args.trace_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_lines = load_lines(feature_src)
    trace_lines = load_lines(trace_src)
    validate_alignment(feature_lines, trace_lines)

    total_needed = args.train_count + args.eval_count
    chosen = choose_indices(len(feature_lines), total_needed, args.seed, args.sample_strategy)
    if len(chosen) < total_needed:
        raise RuntimeError(f"requested {total_needed} rows, got {len(chosen)}")

    rng = random.Random(args.seed + 17)
    shuffled = chosen[:]
    rng.shuffle(shuffled)
    train_idx = sorted(shuffled[: args.train_count])
    eval_idx = sorted(shuffled[args.train_count : args.train_count + args.eval_count])
    if len(train_idx) != args.train_count or len(eval_idx) != args.eval_count:
        raise RuntimeError("split count mismatch")

    faithfulness_take = min(args.faithfulness_count, len(eval_idx))
    faithfulness_rng = random.Random(args.seed + 29)
    faithfulness_idx = sorted(faithfulness_rng.sample(eval_idx, faithfulness_take))

    write_subset(out_dir / "train_manifest.jsonl", feature_lines, train_idx)
    write_subset(out_dir / "eval_manifest.jsonl", feature_lines, eval_idx)
    write_subset(out_dir / "train_trace.jsonl", trace_lines, train_idx)
    write_subset(out_dir / "eval_trace.jsonl", trace_lines, eval_idx)
    write_subset(out_dir / "faithfulness_manifest.jsonl", feature_lines, faithfulness_idx)
    write_subset(out_dir / "faithfulness_trace.jsonl", trace_lines, faithfulness_idx)
    write_freeform_trace(out_dir / "faithfulness_trace_freeform.jsonl", trace_lines, faithfulness_idx)

    meta = {
        "feature_manifest": str(feature_src),
        "trace_manifest": str(trace_src),
        "total_rows": len(feature_lines),
        "train_count": len(train_idx),
        "eval_count": len(eval_idx),
        "faithfulness_count": len(faithfulness_idx),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "train_indices_preview": train_idx[:10],
        "eval_indices_preview": eval_idx[:10],
        "faithfulness_indices_preview": faithfulness_idx[:10],
    }
    (out_dir / "split_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] train_manifest={out_dir / 'train_manifest.jsonl'} rows={len(train_idx)}", flush=True)
    print(f"[ok] eval_manifest={out_dir / 'eval_manifest.jsonl'} rows={len(eval_idx)}", flush=True)
    print(f"[ok] faithfulness_manifest={out_dir / 'faithfulness_manifest.jsonl'} rows={len(faithfulness_idx)}", flush=True)
    print(f"[ok] freeform_trace={out_dir / 'faithfulness_trace_freeform.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
