#!/usr/bin/env python3
"""Generate paper-friendly analysis artifacts for learned gating runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


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


def mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def stage_summary(rows: list[dict], channels: list[str]) -> dict[str, dict]:
    out = {}
    for stage in STAGES:
        sub = [r for r in rows if r.get("stage") == stage]
        if not sub:
            continue
        stage_row = {
            "n": len(sub),
            "success_rate": mean([1.0 if r["success"] else 0.0 for r in sub]),
            "fps": mean([float(r["fps"]) for r in sub]),
            "avg_budget": mean([float(r.get("budget", 0.0)) for r in sub]),
            "fallback_rate": mean([1.0 if r.get("fallback", False) else 0.0 for r in sub]),
        }
        for channel in channels:
            stage_row[f"{channel}_keep_rate"] = mean([1.0 if r.get("gates", {}).get(channel, False) else 0.0 for r in sub])
        out[stage] = stage_row
    return out


def write_stage_markdown(baseline: dict, learned: dict, channels: list[str], out_path: Path) -> None:
    channel_headers = " | ".join(ch.title() for ch in channels)
    lines = [
        f"| Stage | Model | N | Success | FPS | AvgBudget | Fallback | {channel_headers} |",
        f"|---|---|---:|---:|---:|---:|---:|{'---:|' * len(channels)}",
    ]
    for stage in STAGES:
        for name, data in [("Baseline", baseline.get(stage)), ("LearnedGate", learned.get(stage))]:
            if data is None:
                continue
            lines.append(
                f"| {stage} | {name} | {data['n']} | {data['success_rate']:.4f} | {data['fps']:.4f} | "
                f"{data['avg_budget']:.4f} | {data['fallback_rate']:.4f} | "
                + " | ".join(f"{data[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_stage_success(baseline: dict, learned: dict, out_path: Path) -> None:
    if plt is None or np is None:
        return
    labels = [s for s in STAGES if s in baseline and s in learned]
    x = np.arange(len(labels))
    w = 0.36
    bvals = [baseline[s]["success_rate"] for s in labels]
    lvals = [learned[s]["success_rate"] for s in labels]
    plt.figure(figsize=(7.2, 4.8))
    plt.bar(x - w / 2, bvals, w, label="Baseline")
    plt.bar(x + w / 2, lvals, w, label="LearnedGate")
    plt.xticks(x, labels)
    plt.ylabel("Success Rate")
    plt.title("Stage-wise Success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_channel_keep(learned: dict, channels: list[str], out_path: Path) -> None:
    if plt is None or np is None:
        return
    labels = [s for s in STAGES if s in learned]
    x = np.arange(len(labels))
    width = 0.18 if len(channels) >= 4 else 0.22
    plt.figure(figsize=(7.6, 4.8))
    offsets = np.linspace(-width * (len(channels) - 1) / 2, width * (len(channels) - 1) / 2, len(channels))
    for offset, channel in zip(offsets, channels):
        vals = [learned[s][f"{channel}_keep_rate"] for s in labels]
        plt.bar(x + offset, vals, width, label=channel)
    plt.xticks(x, labels)
    plt.ylabel("Keep Rate")
    plt.title("Stage-wise Channel Usage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_jsonl", required=True)
    parser.add_argument("--learned_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_rows = load_jsonl(Path(args.baseline_jsonl))
    learned_rows = load_jsonl(Path(args.learned_jsonl))
    channels = infer_channels(baseline_rows, learned_rows)

    baseline_stage = stage_summary(baseline_rows, channels)
    learned_stage = stage_summary(learned_rows, channels)

    (out_dir / "stage_baseline.json").write_text(json.dumps(baseline_stage, indent=2), encoding="utf-8")
    (out_dir / "stage_learned.json").write_text(json.dumps(learned_stage, indent=2), encoding="utf-8")
    write_stage_markdown(baseline_stage, learned_stage, channels, out_dir / "stage_summary.md")
    plot_stage_success(baseline_stage, learned_stage, out_dir / "stage_success.png")
    plot_channel_keep(learned_stage, channels, out_dir / "stage_channel_usage.png")
    if plt is None or np is None:
        print("[warn] plotting deps not installed; skipped png plots")
    print(f"[ok] analysis_dir={out_dir}")


if __name__ == "__main__":
    main()
