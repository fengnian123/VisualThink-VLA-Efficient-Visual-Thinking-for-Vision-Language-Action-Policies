#!/usr/bin/env python3
"""Summarize subset-based internal interface comparison runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET_META = {
    "bridge": {"paper_name": "BridgeData V2"},
    "fractal": {"paper_name": "Fractal"},
    "roboturk": {"paper_name": "RoboTurk"},
    "libero_long": {"paper_name": "LIBERO-Long"},
    "libero_goal": {"paper_name": "LIBERO-Goal"},
    "libero_object": {"paper_name": "LIBERO-Object"},
    "libero_spatial": {"paper_name": "LIBERO-Spatial"},
    "utaustin_mutex": {"paper_name": "UT Austin MUTEX"},
}

VARIANT_META = {
    "openvla": {"label": "OpenVLA-only", "evidence_path": "none"},
    "prompt_text": {"label": "Prompt-text evidence", "evidence_path": "prompt text"},
    "depth_augmented": {"label": "Depth-augmented", "evidence_path": "5ch soft evidence"},
    "full_soft": {"label": "FullSoft", "evidence_path": "4ch dense soft tokens"},
    "dynamic_soft": {"label": "DynamicSoft", "evidence_path": "4ch routed soft tokens"},
}

DATASET_ORDER = (
    "bridge",
    "fractal",
    "roboturk",
    "libero_object",
    "libero_goal",
    "libero_spatial",
    "libero_long",
    "utaustin_mutex",
)

VARIANT_ORDER = ("openvla", "prompt_text", "depth_augmented", "full_soft", "dynamic_soft")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_variant(rows: list[dict], key: str) -> dict:
    for row in rows:
        if str(row.get("variant")) == key:
            return row
    raise KeyError(f"variant not found: {key}")


def fmt_ratio(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{100.0 * x:.2f}%"


def fmt_latency(x: float | None) -> str:
    if x is None:
        return "--"
    return f"{x:.3f}s"


def fmt_selected(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.2f}"


def collect_dataset(run_root: Path, dataset_key: str, prompt_variant: str, depth_variant: str) -> dict:
    run_dir = run_root / dataset_key
    metrics_json = run_dir / "31_interface_metrics"/ "paper_metrics_summary.json"
    prompt_json = run_dir / "30_prompt_screening"/ "summary.json"
    depth_json = run_dir / "20_depth5_eval"/ "summary.json"
    split_json = run_dir / "00_split"/ "split_config.json"

    metrics = load_json(metrics_json)
    prompt_rows = load_json(prompt_json)
    depth_rows = load_json(depth_json)
    split_cfg = load_json(split_json)

    prompt_row = find_variant(prompt_rows, prompt_variant)
    depth_row = find_variant(depth_rows, depth_variant)
    models = metrics["models"]

    records = {
        "openvla": {
            "success_rate": models["openvla"]["success_rate"],
            "avg_step_latency_s": models["openvla"]["avg_step_latency_s"],
            "avg_selected_channels": models["openvla"].get("avg_selected_channels", 0.0),
        },
        "prompt_text": {
            "success_rate": prompt_row["success_rate"],
            "avg_step_latency_s": prompt_row["avg_step_latency_s"],
            "avg_selected_channels": None,
            "prompt_variant": prompt_variant,
        },
        "depth_augmented": {
            "success_rate": depth_row["success_rate"],
            "avg_step_latency_s": depth_row["avg_step_latency_s"],
            "avg_selected_channels": depth_row.get("avg_selected_channels"),
            "depth_variant": depth_variant,
        },
        "full_soft": {
            "success_rate": models["full_soft"]["success_rate"],
            "avg_step_latency_s": models["full_soft"]["avg_step_latency_s"],
            "avg_selected_channels": models["full_soft"].get("avg_selected_channels", 4.0),
        },
        "dynamic_soft": {
            "success_rate": models["dynamic_soft"]["success_rate"],
            "avg_step_latency_s": models["dynamic_soft"]["avg_step_latency_s"],
            "avg_selected_channels": models["dynamic_soft"].get("avg_selected_channels"),
        },
    }
    return {
        "dataset_key": dataset_key,
        "paper_name": DATASET_META[dataset_key]["paper_name"],
        "subset_train_count": split_cfg["train_count"],
        "subset_eval_count": split_cfg["eval_count"],
        "records": records,
    }


def build_long_table(datasets: list[dict]) -> str:
    lines = [
        "| Dataset | Variant | Evidence path | Eval N | Success | Latency | Avg selected |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        eval_n = dataset["subset_eval_count"]
        for variant_key in VARIANT_ORDER:
            rec = dataset["records"][variant_key]
            meta = VARIANT_META[variant_key]
            lines.append(
                f"| {dataset['paper_name']} | {meta['label']} | {meta['evidence_path']} | {eval_n} | "
                f"{fmt_ratio(rec['success_rate'])} | {fmt_latency(rec['avg_step_latency_s'])} | "
                f"{fmt_selected(rec['avg_selected_channels'])} |"
            )
    return "\n".join(lines) + "\n"


def build_paper_table(datasets: list[dict]) -> str:
    lines = [
        "| Dataset | OpenVLA-only | Prompt-text | Depth-augmented | FullSoft | DynamicSoft |",
        "|---|---|---|---|---|---|",
    ]
    for dataset in datasets:
        cells = []
        for variant_key in VARIANT_ORDER:
            rec = dataset["records"][variant_key]
            cell = f"{fmt_ratio(rec['success_rate'])} / {fmt_latency(rec['avg_step_latency_s'])}"
            cells.append(cell)
        lines.append(
            f"| {dataset['paper_name']} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |"
        )
    return "\n".join(lines) + "\n"


def build_notes(datasets: list[dict], prompt_variant: str, depth_variant: str) -> str:
    train_counts = sorted({item["subset_train_count"] for item in datasets})
    eval_counts = sorted({item["subset_eval_count"] for item in datasets})
    lines = [
        "# Internal Interface Comparison Notes",
        "",
        f"- Benchmarks covered: {', '.join(item['paper_name'] for item in datasets)}",
        f"- Prompt-text evidence row is instantiated by prompt variant `{prompt_variant}`.",
        f"- Depth-augmented row is instantiated by depth-enabled soft-evidence variant `{depth_variant}`.",
        f"- Train subset count(s): {', '.join(str(x) for x in train_counts)}",
        f"- Eval subset count(s): {', '.join(str(x) for x in eval_counts)}",
        "- Every dataset is evaluated on its own fixed eval subset; OpenVLA / FullSoft / DynamicSoft are re-benchmarked on that same subset rather than copied from the full-data paper metrics.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_variant", default="full_schema")
    parser.add_argument("--depth_variant", default="depth5_blend_0p35")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = []
    for dataset_key in DATASET_ORDER:
        run_dir = run_root / dataset_key
        if not run_dir.exists():
            continue
        datasets.append(collect_dataset(run_root, dataset_key, args.prompt_variant, args.depth_variant))

    payload = {
        "run_root": str(run_root),
        "prompt_variant": args.prompt_variant,
        "depth_variant": args.depth_variant,
        "datasets": datasets,
    }
    (output_dir / "internal_interface_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary_table.md").write_text(build_long_table(datasets), encoding="utf-8")
    (output_dir / "paper_table.md").write_text(build_paper_table(datasets), encoding="utf-8")
    (output_dir / "paper_notes.md").write_text(
        build_notes(datasets, args.prompt_variant, args.depth_variant),
        encoding="utf-8",
    )
    print(f"[ok] summary_json={output_dir / 'internal_interface_summary.json'}", flush=True)
    print(f"[ok] summary_table={output_dir / 'summary_table.md'}", flush=True)
    print(f"[ok] paper_table={output_dir / 'paper_table.md'}", flush=True)


if __name__ == "__main__":
    main()
