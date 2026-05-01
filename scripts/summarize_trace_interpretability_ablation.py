#!/usr/bin/env python3
"""Summarize interpretability ablation metrics into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_markdown_table(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0] in {"Metric", "---"}:
            continue
        try:
            metrics[cells[0]] = float(cells[-1])
        except ValueError:
            continue
    return metrics


def parse_threeway_dynamic_success(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| success_rate "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            break
        return float(cells[3])
    raise RuntimeError(f"could not find DynamicSoft success_rate in {path}")


def build_variant_record(name: str, benchmark_path: Path, faithfulness_path: Path) -> dict[str, float | str]:
    faith = parse_markdown_table(faithfulness_path)
    return {
        "variant": name,
        "success_rate": parse_threeway_dynamic_success(benchmark_path),
        "route_rationale_jaccard": faith["route_rationale_jaccard"],
        "top_utility_mentioned_rate": faith["top_utility_mentioned_rate"],
        "top_remove_drop": faith["top_remove_drop"],
        "avg_selected_channels": faith.get("avg_selected_channels", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_benchmark", required=True)
    parser.add_argument("--full_faithfulness", required=True)
    parser.add_argument("--no_route_benchmark", required=True)
    parser.add_argument("--no_route_faithfulness", required=True)
    parser.add_argument("--no_trace_benchmark", required=True)
    parser.add_argument("--no_trace_faithfulness", required=True)
    parser.add_argument("--no_utility_benchmark", required=True)
    parser.add_argument("--no_utility_faithfulness", required=True)
    parser.add_argument("--freeform_benchmark", required=True)
    parser.add_argument("--freeform_faithfulness", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        build_variant_record("full", Path(args.full_benchmark), Path(args.full_faithfulness)),
        build_variant_record("without route supervision", Path(args.no_route_benchmark), Path(args.no_route_faithfulness)),
        build_variant_record("without trace supervision", Path(args.no_trace_benchmark), Path(args.no_trace_faithfulness)),
        build_variant_record("without utility ranking", Path(args.no_utility_benchmark), Path(args.no_utility_faithfulness)),
        build_variant_record("free-form rationale target", Path(args.freeform_benchmark), Path(args.freeform_faithfulness)),
    ]

    (out_dir / "trace_ablation_summary.json").write_text(
        json.dumps(variants, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "| Variant | Success | Route-Rat. | Utility-Mention | TopRemoveDrop |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in variants:
        lines.append(
            f"| {row['variant']} | {100.0 * float(row['success_rate']):.2f}% | "
            f"{float(row['route_rationale_jaccard']):.4f} | "
            f"{float(row['top_utility_mentioned_rate']):.4f} | "
            f"{float(row['top_remove_drop']):+.4f} |"
        )
    (out_dir / "trace_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] summary_json={out_dir / 'trace_ablation_summary.json'}", flush=True)
    print(f"[ok] summary_md={out_dir / 'trace_ablation_table.md'}", flush=True)


if __name__ == "__main__":
    main()
