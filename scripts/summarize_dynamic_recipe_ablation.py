#!/usr/bin/env python3
"""Summarize recipe-ablation outputs into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["variant"]): row for row in rows}


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}"


def fmt_delta(a: float, b: float) -> str:
    return f"{(b - a) * 100.0:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge_recipe_summary", required=True)
    parser.add_argument("--bridge_alpha_summary", required=True)
    parser.add_argument("--libero_recipe_summary", required=True)
    parser.add_argument("--libero_alpha_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "BridgeData V2": (
            load_summary(Path(args.bridge_recipe_summary)),
            load_summary(Path(args.bridge_alpha_summary)),
        ),
        "LIBERO-Long": (
            load_summary(Path(args.libero_recipe_summary)),
            load_summary(Path(args.libero_alpha_summary)),
        ),
    }

    table_lines = [
        "| Comparison | Benchmark | Result |",
        "|---|---|---|",
    ]
    summary = {}

    for benchmark, (recipe_rows, alpha_rows) in datasets.items():
        hard = recipe_rows["hard"]
        blend = recipe_rows["blend_0p35"]
        distill = recipe_rows["blend_0p35_distill"]
        gate_old = recipe_rows["blend_0p35_distill_gate_v4"]
        gate_new = recipe_rows["blend_0p35_distill_gate_v8"]
        alpha20 = alpha_rows["blend_alpha_0p20"]
        alpha35 = alpha_rows["blend_alpha_0p35"]
        alpha50 = alpha_rows["blend_alpha_0p50"]

        table_lines.append(
            f"| Hard sparse masks vs. +blend | {benchmark} | "
            f"{fmt_pct(hard['success_rate'])} -> {fmt_pct(blend['success_rate'])} "
            f"({fmt_delta(hard['success_rate'], blend['success_rate'])} pts), "
            f"L1 {hard['avg_l1']:.4f} -> {blend['avg_l1']:.4f} |"
        )
        table_lines.append(
            f"| +blend vs. +blend+distill | {benchmark} | "
            f"{fmt_pct(blend['success_rate'])} -> {fmt_pct(distill['success_rate'])} "
            f"({fmt_delta(blend['success_rate'], distill['success_rate'])} pts), "
            f"L1 {blend['avg_l1']:.4f} -> {distill['avg_l1']:.4f} |"
        )
        table_lines.append(
            f"| Final unified gate vs. earlier gate variant | {benchmark} | "
            f"{fmt_pct(gate_old['success_rate'])} -> {fmt_pct(gate_new['success_rate'])} "
            f"({fmt_delta(gate_old['success_rate'], gate_new['success_rate'])} pts), "
            f"selected {gate_old['avg_selected_channels']:.2f} -> {gate_new['avg_selected_channels']:.2f} |"
        )
        table_lines.append(
            f"| Blend sweep ($\\alpha=0.35$ vs. nearby values) | {benchmark} | "
            f"0.20:{fmt_pct(alpha20['success_rate'])}, 0.35:{fmt_pct(alpha35['success_rate'])}, "
            f"0.50:{fmt_pct(alpha50['success_rate'])}; "
            f"L1=({alpha20['avg_l1']:.4f}, {alpha35['avg_l1']:.4f}, {alpha50['avg_l1']:.4f}) |"
        )

        summary[benchmark] = {
            "hard_vs_blend": {
                "from": hard,
                "to": blend,
                "delta_success_rate": blend["success_rate"] - hard["success_rate"],
            },
            "blend_vs_distill": {
                "from": blend,
                "to": distill,
                "delta_success_rate": distill["success_rate"] - blend["success_rate"],
            },
            "gate_v4_vs_v8": {
                "from": gate_old,
                "to": gate_new,
                "delta_success_rate": gate_new["success_rate"] - gate_old["success_rate"],
            },
            "alpha_sweep": {
                "alpha_0p20": alpha20,
                "alpha_0p35": alpha35,
                "alpha_0p50": alpha50,
            },
        }

    (out_dir / "recipe_ablation_table.md").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    (out_dir / "recipe_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[ok] table={out_dir / 'recipe_ablation_table.md'}", flush=True)


if __name__ == "__main__":
    main()
