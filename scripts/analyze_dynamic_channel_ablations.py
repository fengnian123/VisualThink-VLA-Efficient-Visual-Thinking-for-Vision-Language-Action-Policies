#!/usr/bin/env python3
"""Post-hoc channel ablation diagnostics for DynamicSoft evidence routing.

This script reads completed DynamicSoft evaluation JSONL files and estimates
the route-conditioned contribution of each evidence channel. It does not replace
training-level leave-one-channel ablations, but it provides a fast, reproducible
diagnostic from already completed full benchmark runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


CHANNELS = ("bbox", "edge", "motion", "relation")
STAGES = ("approach", "grasp", "place")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty input: {path}")
    return rows


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset input must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("dataset name is empty")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"missing JSONL file: {path}")
    return name.strip(), path


def avg(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(mean(vals)) if vals else None


def med(values: Iterable[float]) -> float | None:
    vals = list(values)
    return float(median(vals)) if vals else None


def summarize_binary(rows: list[dict]) -> dict:
    if not rows:
        return {
            "n": 0,
            "success_rate": None,
            "avg_l1": None,
            "median_l1": None,
            "avg_latency_s": None,
        }
    return {
        "n": len(rows),
        "success_rate": avg(1.0 if r.get("success", False) else 0.0 for r in rows),
        "avg_l1": avg(float(r.get("l1", 0.0)) for r in rows),
        "median_l1": med(float(r.get("l1", 0.0)) for r in rows),
        "avg_latency_s": avg(float(r.get("inference_time_s", 0.0)) for r in rows),
    }


def analyze_dataset(dataset: str, path: Path) -> tuple[list[dict], list[dict]]:
    rows = load_jsonl(path)
    channel_rows: list[dict] = []
    stage_rows: list[dict] = []

    overall = summarize_binary(rows)
    for channel in CHANNELS:
        kept = [r for r in rows if bool(r.get("gates", {}).get(channel, False))]
        dropped = [r for r in rows if not bool(r.get("gates", {}).get(channel, False))]
        kept_s = summarize_binary(kept)
        dropped_s = summarize_binary(dropped)
        success_delta = None
        l1_delta = None
        if kept_s["success_rate"] is not None and dropped_s["success_rate"] is not None:
            success_delta = kept_s["success_rate"] - dropped_s["success_rate"]
        if kept_s["avg_l1"] is not None and dropped_s["avg_l1"] is not None:
            l1_delta = dropped_s["avg_l1"] - kept_s["avg_l1"]
        channel_rows.append(
            {
                "dataset": dataset,
                "channel": channel,
                "n": overall["n"],
                "keep_rate": len(kept) / len(rows),
                "success_when_kept": kept_s["success_rate"],
                "success_when_dropped": dropped_s["success_rate"],
                "delta_success_kept_minus_dropped": success_delta,
                "avg_l1_when_kept": kept_s["avg_l1"],
                "avg_l1_when_dropped": dropped_s["avg_l1"],
                "delta_l1_dropped_minus_kept": l1_delta,
                "median_l1_when_kept": kept_s["median_l1"],
                "median_l1_when_dropped": dropped_s["median_l1"],
                "n_kept": kept_s["n"],
                "n_dropped": dropped_s["n"],
                "source": str(path),
            }
        )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        stage = str(row.get("stage", "unknown"))
        for channel in CHANNELS:
            grouped[(stage, channel)].append(row)

    for stage in sorted({str(r.get("stage", "unknown")) for r in rows}, key=lambda s: (STAGES.index(s) if s in STAGES else 999, s)):
        stage_all = [r for r in rows if str(r.get("stage", "unknown")) == stage]
        base = summarize_binary(stage_all)
        for channel in CHANNELS:
            sub = grouped[(stage, channel)]
            kept = [r for r in sub if bool(r.get("gates", {}).get(channel, False))]
            dropped = [r for r in sub if not bool(r.get("gates", {}).get(channel, False))]
            kept_s = summarize_binary(kept)
            dropped_s = summarize_binary(dropped)
            stage_rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "channel": channel,
                    "n_stage": base["n"],
                    "stage_success_rate": base["success_rate"],
                    "stage_avg_l1": base["avg_l1"],
                    "keep_rate": len(kept) / len(sub) if sub else 0.0,
                    "success_when_kept": kept_s["success_rate"],
                    "success_when_dropped": dropped_s["success_rate"],
                    "avg_l1_when_kept": kept_s["avg_l1"],
                    "avg_l1_when_dropped": dropped_s["avg_l1"],
                    "n_kept": kept_s["n"],
                    "n_dropped": dropped_s["n"],
                }
            )
    return channel_rows, stage_rows


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, channel_rows: list[dict], stage_rows: list[dict]) -> None:
    lines = [
        "# DynamicSoft Channel Ablation Diagnostics",
        "",
        "This is a post-hoc, route-conditioned diagnostic from completed DynamicSoft evaluations. "
        "Positive `Delta L1` means rows that kept the channel have lower L1 than rows that dropped it. "
        "It should be reported as a fast ablation/audit, not as a replacement for retrained leave-one-channel variants.",
        "",
        "## Overall Channel Contribution",
        "",
        "| Dataset | Channel | N | Keep rate | Success kept | Success dropped | Delta success | L1 kept | L1 dropped | Delta L1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in channel_rows:
        lines.append(
            f"| {row['dataset']} | {row['channel']} | {row['n']} | {fmt(row['keep_rate'])} | "
            f"{fmt(row['success_when_kept'])} | {fmt(row['success_when_dropped'])} | "
            f"{fmt(row['delta_success_kept_minus_dropped'])} | {fmt(row['avg_l1_when_kept'])} | "
            f"{fmt(row['avg_l1_when_dropped'])} | {fmt(row['delta_l1_dropped_minus_kept'])} |"
        )

    lines.extend(
        [
            "",
            "## Stage-Conditioned Keep Rates",
            "",
            "| Dataset | Stage | Channel | N stage | Keep rate | Stage success | Stage avg L1 | Success kept | Success dropped | L1 kept | L1 dropped |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in stage_rows:
        lines.append(
            f"| {row['dataset']} | {row['stage']} | {row['channel']} | {row['n_stage']} | "
            f"{fmt(row['keep_rate'])} | {fmt(row['stage_success_rate'])} | {fmt(row['stage_avg_l1'])} | "
            f"{fmt(row['success_when_kept'])} | {fmt(row['success_when_dropped'])} | "
            f"{fmt(row['avg_l1_when_kept'])} | {fmt(row['avg_l1_when_dropped'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_dataset_arg, required=True, help="NAME=dynamic_soft_eval.jsonl")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_channel_rows: list[dict] = []
    all_stage_rows: list[dict] = []
    for dataset, path in args.input:
        channel_rows, stage_rows = analyze_dataset(dataset, path)
        all_channel_rows.extend(channel_rows)
        all_stage_rows.extend(stage_rows)

    write_csv(output_dir / "channel_ablation_overall.csv", all_channel_rows)
    write_csv(output_dir / "channel_ablation_by_stage.csv", all_stage_rows)
    (output_dir / "channel_ablation_overall.json").write_text(
        json.dumps(all_channel_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "channel_ablation_by_stage.json").write_text(
        json.dumps(all_stage_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir / "channel_ablation_report.md", all_channel_rows, all_stage_rows)
    print(f"[ok] wrote {output_dir}")


if __name__ == "__main__":
    main()
