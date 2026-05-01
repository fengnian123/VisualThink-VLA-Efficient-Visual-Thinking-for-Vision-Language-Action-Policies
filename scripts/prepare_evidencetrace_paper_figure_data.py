#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CHANNELS = ("bbox", "edge", "motion", "relation")
STAGE_ORDER = ("approach", "grasp", "place")
DOMINANT_EVIDENCE_ORDER = (
    "bbox_dominant",
    "edge_dominant",
    "motion_dominant",
    "relation_dominant",
    "balanced",
    "weak_evidence",
)
DIFFICULTY_ORDER = ("L1", "L2", "L3", "L4", "L5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare EvidenceTrace-VLA figure data for the paper. "
            "Outputs publication-oriented CSV/JSON files for routing sparsity "
            "and stage sensitivity figures."
        )
    )
    parser.add_argument(
        "--primary-hq-trace",
        default="runs/EvidenceTrace-VLA/libero10_full_all/governance/release_hq_trace.jsonl",
        help="Primary HQ-Trace JSONL used for the main figure data export.",
    )
    parser.add_argument(
        "--primary-benchmark-label",
        default="LIBERO-Long",
        help="Benchmark label for the primary HQ-Trace input.",
    )
    parser.add_argument(
        "--secondary-hq-trace",
        default="",
        help="Optional secondary HQ-Trace JSONL for comparison figures.",
    )
    parser.add_argument(
        "--secondary-benchmark-label",
        default="",
        help="Benchmark label for the optional secondary HQ-Trace input.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/NeurIPS2026/figure_data/evidencetrace_routing_stage",
        help="Directory for prepared figure data.",
    )
    return parser.parse_args()


def stage_sort_key(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return len(STAGE_ORDER)


def difficulty_sort_key(level: str) -> int:
    try:
        return DIFFICULTY_ORDER.index(level)
    except ValueError:
        return len(DIFFICULTY_ORDER)


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def bool_mask_value(route_mask: dict, channel: str) -> int:
    value = route_mask.get(channel, 0)
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value) > 0.5)
    except (TypeError, ValueError):
        return 0


def extract_core_row(row: dict, benchmark_label: str) -> dict:
    taxonomy = (row.get("governance") or {}).get("taxonomy") or {}
    route_mask = row.get("route_mask") or {}
    return {
        "benchmark": benchmark_label,
        "dataset_name": str(row.get("dataset_name", "")),
        "stage": str(row.get("stage") or taxonomy.get("stage") or "unknown"),
        "dominant_evidence": str(taxonomy.get("dominant_evidence", "unknown")),
        "primary_primitive": str(taxonomy.get("primary_primitive", "unknown")),
        "difficulty_level": str(taxonomy.get("difficulty_level", "unknown")),
        "bbox": bool_mask_value(route_mask, "bbox"),
        "edge": bool_mask_value(route_mask, "edge"),
        "motion": bool_mask_value(route_mask, "motion"),
        "relation": bool_mask_value(route_mask, "relation"),
    }


def aggregate_stage_channel_keep(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["benchmark"], row["stage"])
        entry = grouped.setdefault(
            key,
            {
                "n": 0,
                "bbox_sum": 0.0,
                "edge_sum": 0.0,
                "motion_sum": 0.0,
                "relation_sum": 0.0,
                "active_sum": 0.0,
            },
        )
        entry["n"] += 1
        for ch in CHANNELS:
            entry[f"{ch}_sum"] += float(row[ch])
        entry["active_sum"] += sum(float(row[ch]) for ch in CHANNELS)

    output = []
    for (benchmark, stage), entry in sorted(
        grouped.items(), key=lambda x: (x[0][0], stage_sort_key(x[0][1]), x[0][1])
    ):
        n = int(entry["n"])
        output.append(
            {
                "benchmark": benchmark,
                "stage": stage,
                "stage_label": f"{benchmark}-{stage}",
                "n": n,
                "bbox_keep_rate": entry["bbox_sum"] / n if n else 0.0,
                "edge_keep_rate": entry["edge_sum"] / n if n else 0.0,
                "motion_keep_rate": entry["motion_sum"] / n if n else 0.0,
                "relation_keep_rate": entry["relation_sum"] / n if n else 0.0,
                "avg_selected_channels": entry["active_sum"] / n if n else 0.0,
                "inactive_channel_rate": 1.0 - ((entry["active_sum"] / n) / len(CHANNELS)) if n else 0.0,
            }
        )
    return output


def aggregate_stage_dominant_evidence(rows: list[dict]) -> list[dict]:
    stage_totals: Counter[tuple[str, str]] = Counter()
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        key = (row["benchmark"], row["stage"])
        stage_totals[key] += 1
        counts[(row["benchmark"], row["stage"], row["dominant_evidence"])] += 1

    output = []
    for benchmark, stage in sorted(stage_totals.keys(), key=lambda x: (x[0], stage_sort_key(x[1]), x[1])):
        total = stage_totals[(benchmark, stage)]
        for label in DOMINANT_EVIDENCE_ORDER:
            count = counts.get((benchmark, stage, label), 0)
            output.append(
                {
                    "benchmark": benchmark,
                    "stage": stage,
                    "stage_label": f"{benchmark}-{stage}",
                    "dominant_evidence": label,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                    "n_stage": total,
                }
            )
    return output


def aggregate_primitive_difficulty(rows: list[dict]) -> list[dict]:
    primitive_totals: Counter[str] = Counter()
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        primitive = row["primary_primitive"]
        difficulty = row["difficulty_level"]
        primitive_totals[primitive] += 1
        counts[(row["benchmark"], primitive, difficulty)] += 1

    benchmark_primitive_totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        benchmark_primitive_totals[(row["benchmark"], row["primary_primitive"])] += 1

    output = []
    for benchmark, primitive in sorted(benchmark_primitive_totals.keys()):
        total = benchmark_primitive_totals[(benchmark, primitive)]
        for level in DIFFICULTY_ORDER:
            count = counts.get((benchmark, primitive, level), 0)
            output.append(
                {
                    "benchmark": benchmark,
                    "primary_primitive": primitive,
                    "difficulty_level": level,
                    "count": count,
                    "fraction_within_primitive": count / total if total else 0.0,
                    "n_primitive": total,
                }
            )
    return output


def aggregate_benchmark_overview(rows: list[dict]) -> list[dict]:
    counts = Counter(row["benchmark"] for row in rows)
    output = []
    for benchmark, total in sorted(counts.items()):
        subset = [row for row in rows if row["benchmark"] == benchmark]
        avg_selected_channels = (
            sum(sum(float(r[ch]) for ch in CHANNELS) for r in subset) / total if total else 0.0
        )
        output.append(
            {
                "benchmark": benchmark,
                "num_rows": total,
                "bbox_route_rate": sum(float(r["bbox"]) for r in subset) / total if total else 0.0,
                "edge_route_rate": sum(float(r["edge"]) for r in subset) / total if total else 0.0,
                "motion_route_rate": sum(float(r["motion"]) for r in subset) / total if total else 0.0,
                "relation_route_rate": sum(float(r["relation"]) for r in subset) / total if total else 0.0,
                "avg_selected_channels": avg_selected_channels,
                "inactive_channel_rate": 1.0 - (avg_selected_channels / len(CHANNELS)) if total else 0.0,
            }
        )
    return output


def aggregate_active_channel_histogram(rows: list[dict]) -> list[dict]:
    counts: Counter[tuple[str, str, int]] = Counter()
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        active_channels = int(sum(int(row[ch]) for ch in CHANNELS))
        for stage in ("all", row["stage"]):
            counts[(row["benchmark"], stage, active_channels)] += 1
            totals[(row["benchmark"], stage)] += 1

    output = []
    for benchmark, stage in sorted(
        totals.keys(), key=lambda x: (x[0], -1 if x[1] == "all" else stage_sort_key(x[1]), x[1])
    ):
        total = totals[(benchmark, stage)]
        active_values = sorted({key[2] for key in counts if key[0] == benchmark and key[1] == stage})
        for active_channels in active_values:
            count = counts.get((benchmark, stage, active_channels), 0)
            output.append(
                {
                    "benchmark": benchmark,
                    "stage": stage,
                    "stage_label": f"{benchmark}-{stage}" if stage != "all" else f"{benchmark}-all",
                    "active_channels": active_channels,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                    "n_group": total,
                }
            )
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(path: Path, metadata: dict) -> None:
    main_figure_line = "Use one mixed figure with two panels:"
    if len(metadata["inputs"]) == 1:
        main_figure_line = "Use one two-panel figure:"
    lines = [
        "# EvidenceTrace Figure Data",
        "",
        "This directory contains prepared data for the paper's routing sparsity and stage sensitivity figures.",
        "",
        "## Inputs",
        "",
    ]
    for item in metadata["inputs"]:
        lines.append(f"- `{item['benchmark']}`: `{item['path']}`")
    lines += [
        "",
        "## Outputs",
        "",
        "- `main_stage_channel_keep_rate.csv`: stage x channel keep-rate heatmap input",
        "- `main_stage_dominant_evidence.csv`: stage x dominant-evidence stacked-bar input",
        "- `appendix_primitive_difficulty.csv`: primitive x difficulty appendix input",
        "- `benchmark_overview.csv`: compact benchmark-level route-rate summary",
        "- `active_channel_histogram.csv`: active-channel-count distribution for caption/appendix support",
        "- `plot_manifest.json`: plotting metadata, labels, and recommended orders",
        "",
        "## Main-Text Figure Recommendation",
        "",
        main_figure_line,
        "",
        "- Panel A: `stage x channel keep-rate` heatmap",
        "- Panel B: `stage x dominant_evidence` normalized stacked bar chart",
        "",
        "## Appendix Figure Recommendation",
        "",
        "- `primary_primitive x difficulty_level` heatmap or grouped bar chart",
        "",
        "## Notes",
        "",
        "- The current exports are derived directly from governed `release_hq_trace.jsonl` files.",
        "- `main_stage_channel_keep_rate.csv` also includes `avg_selected_channels` and `inactive_channel_rate` so the same file can support both figure captions and appendix routing tables.",
        "- `active_channel_histogram.csv` is intended for quick checks of whether routing remains effectively sparse at the benchmark or stage level.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_specs: list[tuple[str, Path]] = []
    primary_path = Path(args.primary_hq_trace)
    input_specs.append((args.primary_benchmark_label, primary_path))
    if args.secondary_hq_trace:
        input_specs.append((args.secondary_benchmark_label, Path(args.secondary_hq_trace)))

    rows = []
    for benchmark_label, path in input_specs:
        for row in read_rows(path):
            rows.append(extract_core_row(row, benchmark_label))

    stage_channel_rows = aggregate_stage_channel_keep(rows)
    stage_dominant_rows = aggregate_stage_dominant_evidence(rows)
    primitive_difficulty_rows = aggregate_primitive_difficulty(rows)
    benchmark_overview_rows = aggregate_benchmark_overview(rows)
    active_channel_hist_rows = aggregate_active_channel_histogram(rows)

    write_csv(
        out_dir / "main_stage_channel_keep_rate.csv",
        stage_channel_rows,
        [
            "benchmark",
            "stage",
            "stage_label",
            "n",
            "bbox_keep_rate",
            "edge_keep_rate",
            "motion_keep_rate",
            "relation_keep_rate",
            "avg_selected_channels",
            "inactive_channel_rate",
        ],
    )
    write_csv(
        out_dir / "main_stage_dominant_evidence.csv",
        stage_dominant_rows,
        [
            "benchmark",
            "stage",
            "stage_label",
            "dominant_evidence",
            "count",
            "fraction",
            "n_stage",
        ],
    )
    write_csv(
        out_dir / "appendix_primitive_difficulty.csv",
        primitive_difficulty_rows,
        [
            "benchmark",
            "primary_primitive",
            "difficulty_level",
            "count",
            "fraction_within_primitive",
            "n_primitive",
        ],
    )
    write_csv(
        out_dir / "benchmark_overview.csv",
        benchmark_overview_rows,
        [
            "benchmark",
            "num_rows",
            "bbox_route_rate",
            "edge_route_rate",
            "motion_route_rate",
            "relation_route_rate",
            "avg_selected_channels",
            "inactive_channel_rate",
        ],
    )
    write_csv(
        out_dir / "active_channel_histogram.csv",
        active_channel_hist_rows,
        [
            "benchmark",
            "stage",
            "stage_label",
            "active_channels",
            "count",
            "fraction",
            "n_group",
        ],
    )

    manifest = {
        "inputs": [
            {"benchmark": benchmark_label, "path": str(path)}
            for benchmark_label, path in input_specs
        ],
        "channel_order": list(CHANNELS),
        "stage_order": list(STAGE_ORDER),
        "dominant_evidence_order": list(DOMINANT_EVIDENCE_ORDER),
        "difficulty_order": list(DIFFICULTY_ORDER),
        "recommended_main_figure": {
            "panel_a": {
                "type": "heatmap",
                "path": str(out_dir / "main_stage_channel_keep_rate.csv"),
                "x": list(CHANNELS),
                "y": "stage_label",
                "value_fields": [f"{ch}_keep_rate" for ch in CHANNELS],
                "summary_fields": ["avg_selected_channels", "inactive_channel_rate"],
            },
            "panel_b": {
                "type": "normalized_stacked_bar",
                "path": str(out_dir / "main_stage_dominant_evidence.csv"),
                "x": "stage_label",
                "stack": "dominant_evidence",
                "y": "fraction",
            },
        },
        "recommended_appendix_figure": {
            "type": "heatmap_or_grouped_bar",
            "path": str(out_dir / "appendix_primitive_difficulty.csv"),
            "x": "difficulty_level",
            "y": "primary_primitive",
            "value": "fraction_within_primitive",
            "facet": "benchmark" if len(input_specs) > 1 else None,
        },
        "supporting_summary": {
            "benchmark_overview": str(out_dir / "benchmark_overview.csv"),
            "active_channel_histogram": str(out_dir / "active_channel_histogram.csv"),
        },
    }
    write_json(out_dir / "plot_manifest.json", manifest)
    write_readme(out_dir / "README.md", manifest)

    print(f"[ok] wrote {out_dir}")
    for name in [
        "main_stage_channel_keep_rate.csv",
        "main_stage_dominant_evidence.csv",
        "appendix_primitive_difficulty.csv",
        "benchmark_overview.csv",
        "active_channel_histogram.csv",
        "plot_manifest.json",
        "README.md",
    ]:
        print(f" - {out_dir / name}")


if __name__ == "__main__":
    main()
