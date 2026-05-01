#!/usr/bin/env python3
"""Prepare fixed train/eval feature-manifest subsets for recipe ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_lines(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty manifest: {path}")
    return lines


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_count", type=int, default=4096)
    parser.add_argument("--eval_count", type=int, default=2048)
    parser.add_argument("--sample_strategy", choices=("random", "stride"), default="random")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    src = Path(args.feature_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = load_lines(src)
    total_needed = args.train_count + args.eval_count
    chosen = choose_indices(len(lines), total_needed, args.seed, args.sample_strategy)
    if len(chosen) < total_needed:
        raise RuntimeError(f"requested {total_needed} rows, only sampled {len(chosen)}")

    rng = random.Random(args.seed + 17)
    shuffled = chosen[:]
    rng.shuffle(shuffled)
    train_idx = sorted(shuffled[: args.train_count])
    eval_idx = sorted(shuffled[args.train_count : args.train_count + args.eval_count])
    if len(train_idx) != args.train_count or len(eval_idx) != args.eval_count:
        raise RuntimeError("subset split count mismatch")

    write_subset(out_dir / "train_manifest.jsonl", lines, train_idx)
    write_subset(out_dir / "eval_manifest.jsonl", lines, eval_idx)

    meta = {
        "feature_manifest": str(src),
        "total_rows": len(lines),
        "train_count": len(train_idx),
        "eval_count": len(eval_idx),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "train_indices_preview": train_idx[:10],
        "eval_indices_preview": eval_idx[:10],
    }
    (out_dir / "split_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] train_manifest={out_dir / 'train_manifest.jsonl'} rows={len(train_idx)}", flush=True)
    print(f"[ok] eval_manifest={out_dir / 'eval_manifest.jsonl'} rows={len(eval_idx)}", flush=True)


if __name__ == "__main__":
    main()
