#!/usr/bin/env python3
"""Summarize extra channel-screening robustness checks across datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def by_variant(rows: list[dict]) -> dict[str, dict]:
    return {str(row["variant"]): row for row in rows}


def fmt_pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100.0:.2f}%"


def fmt_pts(value: float | None) -> str:
    return "--" if value is None else f"{value * 100.0:+.2f} pts"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", default=[], help="name=prompt_summary.json=depth_summary.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if not args.spec:
      raise RuntimeError("at least one --spec is required")

    rows = []
    for raw in args.spec:
        parts = raw.split("=")
        if len(parts) != 3:
            raise RuntimeError(f"invalid spec: {raw}")
        name, prompt_path, depth_path = parts
        prompt = by_variant(load_json(prompt_path))
        depth = by_variant(load_json(depth_path))
        original = prompt.get("original", {})
        full_schema = prompt.get("full_schema", {})
        depth_only = prompt.get("depth_only", {})
        caption_query = prompt.get("caption_query_only", {})
        depth5 = depth.get("depth5_blend_0p35", {})
        orig_success = original.get("success_rate")
        rows.append(
            {
                "dataset": name,
                "original_success": orig_success,
                "full_schema_success": full_schema.get("success_rate"),
                "full_schema_gain": None if orig_success is None else full_schema.get("success_rate", 0.0) - orig_success,
                "depth_only_gain": None if orig_success is None else depth_only.get("success_rate", 0.0) - orig_success,
                "caption_query_gain": None if orig_success is None else caption_query.get("success_rate", 0.0) - orig_success,
                "depth5_success": depth5.get("success_rate"),
                "depth5_keep_rate": depth5.get("depth_keep_rate"),
                "depth5_selected": depth5.get("avg_selected_channels"),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extra_channel_screening_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "| Dataset | Original | Full schema | Full gain | Depth-only gain | Caption/query gain | Depth5 success | Depth5 keep | Depth5 selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        selected = row["depth5_selected"]
        lines.append(
            f"| {row['dataset']} | {fmt_pct(row['original_success'])} | {fmt_pct(row['full_schema_success'])} | "
            f"{fmt_pts(row['full_schema_gain'])} | {fmt_pts(row['depth_only_gain'])} | {fmt_pts(row['caption_query_gain'])} | "
            f"{fmt_pct(row['depth5_success'])} | {fmt_pct(row['depth5_keep_rate'])} | "
            f"{'--' if selected is None else f'{selected:.2f}'} |"
        )
    (out_dir / "extra_channel_screening_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
