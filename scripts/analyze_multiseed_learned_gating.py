#!/usr/bin/env python3
"""Aggregate multi-seed learned-gating runs into paper-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = ("approach", "grasp", "place")


def infer_channels(*rowsets: list[dict]) -> list[str]:
    seen = []
    for rows in rowsets:
        for row in rows:
            for key in row.get("gates", {}).keys():
                if key not in seen:
                    seen.append(key)
    return seen or ["bbox", "depth", "edge"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mu = mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1)))


def summarize_rows(rows: list[dict], channels: list[str]) -> dict[str, float]:
    out = {
        "n": float(len(rows)),
        "success_rate": mean([1.0 if r["success"] else 0.0 for r in rows]),
        "inference_fps": mean([float(r["fps"]) for r in rows]),
        "robustness_recovery": mean([1.0 if r.get("recovered", False) else 0.0 for r in rows if r.get("disturbed", False)]),
        "avg_selected_channels": mean([float(r.get("selected_channels", 0.0)) for r in rows]),
        "fallback_rate": mean([1.0 if r.get("fallback", False) else 0.0 for r in rows]),
        "avg_budget": mean([float(r.get("budget", 0.0)) for r in rows]),
    }
    for channel in channels:
        out[f"{channel}_keep_rate"] = mean([1.0 if r.get("gates", {}).get(channel, False) else 0.0 for r in rows])
    return out


def summarize_stage(rows: list[dict], channels: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        sub = [r for r in rows if r.get("stage") == stage]
        if sub:
            out[stage] = summarize_rows(sub, channels)
    return out


def metric_triplets(per_seed: list[dict], metric: str) -> tuple[list[float], list[float], list[float]]:
    baseline_vals = [float(row["baseline"][metric]) for row in per_seed]
    learned_vals = [float(row["learned"][metric]) for row in per_seed]
    delta_vals = [l - b for b, l in zip(baseline_vals, learned_vals)]
    return baseline_vals, learned_vals, delta_vals


def write_main_markdown(per_seed: list[dict], channels: list[str], out_path: Path) -> None:
    metrics = [
        "success_rate",
        "inference_fps",
        "robustness_recovery",
        "avg_selected_channels",
        "fallback_rate",
        "avg_budget",
    ]
    metrics.extend([f"{channel}_keep_rate" for channel in channels])
    lines = ["| Metric | Baseline Mean | Learned Mean | Delta Mean | Delta Std |", "|---|---:|---:|---:|---:|"]
    for metric in metrics:
        bvals, lvals, dvals = metric_triplets(per_seed, metric)
        lines.append(
            f"| {metric} | {mean(bvals):.4f} | {mean(lvals):.4f} | {mean(dvals):+.4f} | {std(dvals):.4f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_per_seed_markdown(per_seed: list[dict], channels: list[str], out_path: Path) -> None:
    channel_headers = " | ".join(ch.title() for ch in channels)
    lines = [
        f"| Seed | Base Success | Gate Success | Delta Success | Base FPS | Gate FPS | Delta FPS | Budget | {channel_headers} |",
        f"|---|---:|---:|---:|---:|---:|---:|---:|{'---:|' * len(channels)}",
    ]
    for row in per_seed:
        base = row["baseline"]
        learned = row["learned"]
        lines.append(
            f"| {row['seed']} | {base['success_rate']:.4f} | {learned['success_rate']:.4f} | "
            f"{learned['success_rate']-base['success_rate']:+.4f} | {base['inference_fps']:.4f} | "
            f"{learned['inference_fps']:.4f} | {learned['inference_fps']-base['inference_fps']:+.4f} | "
            f"{learned['avg_budget']:.2f} | "
            + " | ".join(f"{learned[f'{ch}_keep_rate']:.2f}" for ch in channels)
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stage_markdown(stage_agg: dict[str, dict], channels: list[str], out_path: Path) -> None:
    lines = [
        "| Stage | Metric | Baseline Mean | Learned Mean | Delta Mean | Delta Std |",
        "|---|---|---:|---:|---:|---:|",
    ]
    metrics = ["success_rate", "inference_fps", "avg_budget"] + [f"{channel}_keep_rate" for channel in channels]
    for stage in STAGES:
        if stage not in stage_agg:
            continue
        for metric in metrics:
            entry = stage_agg[stage][metric]
            lines.append(
                f"| {stage} | {metric} | {entry['baseline_mean']:.4f} | {entry['learned_mean']:.4f} | "
                f"{entry['delta_mean']:+.4f} | {entry['delta_std']:.4f} |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_per_seed_tradeoff(per_seed: list[dict], out_path: Path) -> None:
    plt.figure(figsize=(6.6, 4.8))
    for row in per_seed:
        seed = row["seed"]
        base = row["baseline"]
        learned = row["learned"]
        plt.scatter(base["inference_fps"], base["success_rate"], marker="o", s=70, label=f"Baseline s{seed}")
        plt.scatter(learned["inference_fps"], learned["success_rate"], marker="^", s=80, label=f"Gate s{seed}")
        plt.plot(
            [base["inference_fps"], learned["inference_fps"]],
            [base["success_rate"], learned["success_rate"]],
            linestyle="--",
            linewidth=1.0,
            color="gray",
            alpha=0.8,
        )
    plt.xlabel("Inference FPS")
    plt.ylabel("Success Rate")
    plt.title("Per-seed Speed/Accuracy Tradeoff")
    plt.legend(fontsize=8, ncols=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_stage_success(stage_agg: dict[str, dict], out_path: Path) -> None:
    labels = [s for s in STAGES if s in stage_agg]
    x = np.arange(len(labels))
    width = 0.35
    bvals = [stage_agg[s]["success_rate"]["baseline_mean"] for s in labels]
    lvals = [stage_agg[s]["success_rate"]["learned_mean"] for s in labels]
    plt.figure(figsize=(7.0, 4.8))
    plt.bar(x - width / 2, bvals, width, label="Baseline")
    plt.bar(x + width / 2, lvals, width, label="LearnedGate")
    plt.xticks(x, labels)
    plt.ylabel("Success Rate")
    plt.title("Stage-wise Success Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_stage_channel_usage(stage_agg: dict[str, dict], channels: list[str], out_path: Path) -> None:
    labels = [s for s in STAGES if s in stage_agg]
    x = np.arange(len(labels))
    width = 0.18 if len(channels) >= 4 else 0.22
    plt.figure(figsize=(7.4, 4.8))
    offsets = np.linspace(-width * (len(channels) - 1) / 2, width * (len(channels) - 1) / 2, len(channels))
    for offset, channel in zip(offsets, channels):
        vals = [stage_agg[s][f"{channel}_keep_rate"]["learned_mean"] for s in labels]
        plt.bar(x + offset, vals, width, label=channel)
    plt.xticks(x, labels)
    plt.ylabel("Keep Rate")
    plt.title("Stage-wise Channel Usage Across Seeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiseed_root", required=True)
    args = parser.parse_args()

    root = Path(args.multiseed_root)
    out_dir = root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted([p for p in root.glob("seed_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[-1]))
    per_seed = []
    stage_raw: dict[str, dict[str, list[float]]] = {}
    channels: list[str] = []
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.split("_")[-1])
        baseline_rows = load_jsonl(seed_dir / "eval"/ "baseline_eval.jsonl")
        learned_rows = load_jsonl(seed_dir / "eval"/ "learned_gating_eval.jsonl")
        if not channels:
            channels = infer_channels(baseline_rows, learned_rows)
        baseline_summary = summarize_rows(baseline_rows, channels)
        learned_summary = summarize_rows(learned_rows, channels)
        per_seed.append({"seed": seed, "baseline": baseline_summary, "learned": learned_summary})

        base_stage = summarize_stage(baseline_rows, channels)
        learned_stage = summarize_stage(learned_rows, channels)
        for stage in STAGES:
            if stage not in base_stage or stage not in learned_stage:
                continue
            stage_raw.setdefault(stage, {})
            for metric in ("success_rate", "inference_fps", "avg_budget", *[f"{channel}_keep_rate" for channel in channels]):
                stage_raw[stage].setdefault(f"baseline::{metric}", []).append(base_stage[stage][metric])
                stage_raw[stage].setdefault(f"learned::{metric}", []).append(learned_stage[stage][metric])

    stage_agg: dict[str, dict] = {}
    for stage, metric_map in stage_raw.items():
        stage_agg[stage] = {}
        for metric in ("success_rate", "inference_fps", "avg_budget", *[f"{channel}_keep_rate" for channel in channels]):
            bvals = metric_map.get(f"baseline::{metric}", [])
            lvals = metric_map.get(f"learned::{metric}", [])
            dvals = [l - b for b, l in zip(bvals, lvals)]
            stage_agg[stage][metric] = {
                "baseline_mean": mean(bvals),
                "learned_mean": mean(lvals),
                "delta_mean": mean(dvals),
                "delta_std": std(dvals),
            }

    (out_dir / "per_seed_summary.json").write_text(json.dumps(per_seed, indent=2), encoding="utf-8")
    (out_dir / "stage_aggregate.json").write_text(json.dumps(stage_agg, indent=2), encoding="utf-8")
    write_main_markdown(per_seed, channels, out_dir / "comparison_summary.md")
    write_per_seed_markdown(per_seed, channels, out_dir / "per_seed_table.md")
    write_stage_markdown(stage_agg, channels, out_dir / "stage_comparison.md")
    plot_per_seed_tradeoff(per_seed, out_dir / "per_seed_tradeoff.png")
    plot_stage_success(stage_agg, out_dir / "stage_success.png")
    plot_stage_channel_usage(stage_agg, channels, out_dir / "stage_channel_usage.png")
    print(f"[ok] analysis_dir={out_dir}")


if __name__ == "__main__":
    main()
