#!/usr/bin/env python3
"""Run budget-aware learned gating for multiple seeds and aggregate results."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def load_summary_table(path: Path) -> dict[str, float]:
    metrics = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| Metric"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) != 4:
            continue
        metric, _, learned, _ = parts
        try:
            metrics[metric] = float(learned)
        except ValueError:
            continue
    return metrics


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--disturb_ratio", type=float, default=0.9)
    parser.add_argument("--disturb_scale", type=float, default=0.35)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for seed in args.seeds:
        seed_dir = root / f"seed_{seed}"
        ckpt_dir = seed_dir / "ckpt"
        eval_dir = seed_dir / "eval"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "scripts/train_learned_gating.py",
            "--feature_manifest",
            args.feature_manifest,
            "--output_dir",
            str(ckpt_dir),
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--log_every",
            "20",
        ]
        if args.limit > 0:
            train_cmd += ["--limit", str(args.limit)]
        subprocess.run(train_cmd, check=True)

        bench_cmd = [
            sys.executable,
            "scripts/benchmark_learned_gating.py",
            "--feature_manifest",
            args.feature_manifest,
            "--checkpoint_dir",
            str(ckpt_dir),
            "--config",
            args.config,
            "--output_dir",
            str(eval_dir),
            "--disturb_ratio",
            str(args.disturb_ratio),
            "--disturb_scale",
            str(args.disturb_scale),
            "--success_l1_thresh",
            str(args.success_l1_thresh),
            "--seed",
            str(seed),
        ]
        if args.limit > 0:
            bench_cmd += ["--limit", str(args.limit)]
        subprocess.run(bench_cmd, check=True)

        metrics = load_summary_table(eval_dir / "summary_table.md")
        metrics["seed"] = seed
        all_metrics.append(metrics)

    metric_names = [k for k in all_metrics[0].keys() if k != "seed"]
    agg = {}
    for metric in metric_names:
        vals = [row[metric] for row in all_metrics]
        mu, sd = mean_std(vals)
        agg[metric] = {"mean": mu, "std": sd}

    (root / "per_seed_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (root / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    md = ["| Metric | Mean | Std |", "|---|---:|---:|"]
    for metric in metric_names:
        md.append(f"| {metric} | {agg[metric]['mean']:.4f} | {agg[metric]['std']:.4f} |")
    (root / "aggregate_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ok] multiseed_root={root}")


if __name__ == "__main__":
    main()
