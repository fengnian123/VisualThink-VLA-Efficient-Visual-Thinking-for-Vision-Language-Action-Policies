#!/usr/bin/env python3
"""Summarize three-way OpenVLA soft-evidence benchmark results by instruction.

This is mainly intended for datasets such as LIBERO where the instruction text
aligns well with task categories. The script reads the original feature
manifest, reconstructs the evaluation order used by
`benchmark_openvla_soft_three_way.py`, joins the per-sample benchmark outputs,
and writes aggregated markdown/json summaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty jsonl: {path}")
    return rows


def load_manifest_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def select_candidate_rows(rows: list[dict], limit: int, skip_empty_instruction: bool) -> list[dict]:
    selected = []
    for row in rows:
        instruction = str(row.get("instruction", "") or "").strip()
        if skip_empty_instruction and not instruction:
            continue
        selected.append(row)
    if limit > 0:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError("no candidate rows selected")
    return selected


def summarize(eval_rows: list[dict], channels: list[str]) -> dict[str, float]:
    success = [1.0 if r.get("success", False) else 0.0 for r in eval_rows]
    fps = [float(r.get("fps", 0.0)) for r in eval_rows]
    selected = [float(r.get("selected_channels", 0.0)) for r in eval_rows]
    denom = float(len(eval_rows)) if eval_rows else 1.0
    out: dict[str, float] = {
        "n": int(len(eval_rows)),
        "success_rate": float(sum(success) / denom) if eval_rows else 0.0,
        "fps": float(sum(fps) / denom) if eval_rows else 0.0,
        "avg_selected_channels": float(sum(selected) / denom) if eval_rows else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = (
            float(sum(1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in eval_rows) / denom)
            if eval_rows
            else 0.0
        )
    return out


def attach_groups(candidate_rows: list[dict], eval_rows: list[dict], group_key: str) -> dict[str, list[dict]]:
    if len(candidate_rows) != len(eval_rows):
        raise RuntimeError(
            f"row count mismatch: manifest candidates={len(candidate_rows)} eval_rows={len(eval_rows)}"
        )
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, (manifest_row, eval_row) in enumerate(zip(candidate_rows, eval_rows)):
        if int(eval_row.get("idx", i)) != i:
            raise RuntimeError(f"unexpected eval row order at position {i}: idx={eval_row.get('idx')}")
        group_value = str(manifest_row.get(group_key, "") or "").strip()
        if not group_value:
            group_value = "<empty>"
        groups[group_value].append(eval_row)
    return groups


def write_markdown(
    out_path: Path,
    overall: dict[str, dict[str, float]],
    grouped: dict[str, dict[str, dict[str, float]]],
    group_order: list[str],
    channels: list[str],
) -> None:
    lines = [
        "| Category | Model | N | Success | FPS | AvgSelected | "
        + " | ".join(ch.title() for ch in channels)
        + " |",
        "|---|---|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    model_order = ["OpenVLA", "FullSoft", "DynamicSoft"]
    for model_name in model_order:
        row = overall[model_name]
        lines.append(
            f"| Overall | {model_name} | {row['n']} | {row['success_rate']:.4f} | {row['fps']:.4f} | "
            f"{row['avg_selected_channels']:.4f} | "
            + " | ".join(f"{row[f'{ch}_keep_rate']:.4f}" for ch in channels)
            + " |"
        )
    for group in group_order:
        for model_name in model_order:
            row = grouped[group][model_name]
            lines.append(
                f"| {group} | {model_name} | {row['n']} | {row['success_rate']:.4f} | {row['fps']:.4f} | "
                f"{row['avg_selected_channels']:.4f} | "
                + " | ".join(f"{row[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--openvla_eval", required=True)
    parser.add_argument("--full_soft_eval", required=True)
    parser.add_argument("--dynamic_soft_eval", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_empty_instruction", action="store_true")
    parser.add_argument("--group_key", default="instruction")
    args = parser.parse_args()

    manifest_rows = load_manifest_rows(args.feature_manifest)
    candidate_rows = select_candidate_rows(
        manifest_rows,
        limit=args.limit,
        skip_empty_instruction=args.skip_empty_instruction,
    )
    openvla_rows = load_jsonl(args.openvla_eval)
    full_rows = load_jsonl(args.full_soft_eval)
    dynamic_rows = load_jsonl(args.dynamic_soft_eval)

    if not (len(candidate_rows) == len(openvla_rows) == len(full_rows) == len(dynamic_rows)):
        raise RuntimeError(
            "candidate/eval row count mismatch: "
            f"{len(candidate_rows)} / {len(openvla_rows)} / {len(full_rows)} / {len(dynamic_rows)}"
        )

    # Derive channels from the gate dicts that actually appear in the benchmark outputs.
    channel_counter: Counter[str] = Counter()
    for row in dynamic_rows:
        channel_counter.update(row.get("gates", {}).keys())
    channels = sorted(channel_counter.keys())
    if not channels:
        raise RuntimeError("could not infer channels from eval outputs")

    openvla_grouped = attach_groups(candidate_rows, openvla_rows, args.group_key)
    full_grouped = attach_groups(candidate_rows, full_rows, args.group_key)
    dynamic_grouped = attach_groups(candidate_rows, dynamic_rows, args.group_key)

    # Order categories by sample count descending.
    counts = Counter()
    for key, rows in openvla_grouped.items():
        counts[key] = len(rows)
    group_order = [key for key, _ in counts.most_common()]

    overall = {
        "OpenVLA": summarize(openvla_rows, channels),
        "FullSoft": summarize(full_rows, channels),
        "DynamicSoft": summarize(dynamic_rows, channels),
    }
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for key in group_order:
        grouped[key] = {
            "OpenVLA": summarize(openvla_grouped[key], channels),
            "FullSoft": summarize(full_grouped[key], channels),
            "DynamicSoft": summarize(dynamic_grouped[key], channels),
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_markdown(out_dir / "instruction_summary.md", overall, grouped, group_order, channels)
    (out_dir / "instruction_summary.json").write_text(
        json.dumps(
            {
                "group_key": args.group_key,
                "channels": channels,
                "overall": overall,
                "grouped": grouped,
                "group_order": group_order,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[ok] wrote {out_dir / 'instruction_summary.md'}")


if __name__ == "__main__":
    main()
