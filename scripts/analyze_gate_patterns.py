#!/usr/bin/env python3
"""Analyze route patterns for learned gating runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pattern_from_row(row: dict) -> tuple[str, ...]:
    return tuple(sorted(k for k, v in row.get("gates", {}).items() if v))


def parse_target_phrase(instruction: str) -> str:
    text = instruction.lower().strip()
    patterns = [
        r"^(?:move|place|put|push|pick up|grab|lift)\s+the\s+(.+?)\s+(?:to|into|onto|in|on|at|toward|towards)\b",
        r"^(?:move|place|put|push|pick up|grab|lift)\s+(.+?)\s+(?:to|into|onto|in|on|at|toward|towards)\b",
    ]
    for pattern in patterns:
        m = re.match(pattern, text)
        if m:
            phrase = m.group(1).strip()
            phrase = re.sub(r"\s+", " ", phrase)
            return phrase
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(tokens[:4]) if tokens else "unknown"


def top_counter_md(counter: Counter, total: int, top_k: int = 12) -> list[str]:
    lines = ["| Pattern | Count | Ratio |", "|---|---:|---:|"]
    for pattern, count in counter.most_common(top_k):
        label = ", ".join(pattern) if pattern else "none"
        lines.append(f"| `{label}` | {count} | {count / max(1, total):.4f} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--eval_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=12)
    args = parser.parse_args()

    manifest_rows = load_jsonl(Path(args.feature_manifest))
    eval_rows = load_jsonl(Path(args.eval_jsonl))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joined = []
    for row in eval_rows:
        idx = int(row["idx"])
        if idx < 0 or idx >= len(manifest_rows):
            continue
        feat = manifest_rows[idx]
        joined.append(
            {
                **row,
                "instruction": feat.get("instruction", ""),
                "query_words": feat.get("query_words", []),
                "target_phrase": parse_target_phrase(feat.get("instruction", "")),
                "pattern": pattern_from_row(row),
            }
        )

    global_patterns = Counter(item["pattern"] for item in joined)
    stage_patterns: dict[str, Counter] = defaultdict(Counter)
    instruction_patterns: dict[str, Counter] = defaultdict(Counter)
    target_patterns: dict[str, Counter] = defaultdict(Counter)

    for item in joined:
        stage_patterns[item.get("stage", "unknown")][item["pattern"]] += 1
        instruction_patterns[item["instruction"]][item["pattern"]] += 1
        target_patterns[item["target_phrase"]][item["pattern"]] += 1

    # Global pattern summary.
    global_lines = [f"# Global Patterns", "", f"Total samples: `{len(joined)}`", ""]
    global_lines.extend(top_counter_md(global_patterns, len(joined), top_k=args.top_k))
    (out_dir / "global_patterns.md").write_text("\n".join(global_lines) + "\n", encoding="utf-8")

    # Stage pattern summary.
    stage_lines = ["# Stage-wise Patterns", ""]
    for stage in sorted(stage_patterns.keys()):
        counter = stage_patterns[stage]
        stage_lines.append(f"## {stage}")
        stage_lines.append(f"- unique_patterns: `{len(counter)}`")
        stage_lines.append(f"- samples: `{sum(counter.values())}`")
        stage_lines.append("")
        stage_lines.extend(top_counter_md(counter, sum(counter.values()), top_k=args.top_k))
        stage_lines.append("")
    (out_dir / "stage_patterns.md").write_text("\n".join(stage_lines) + "\n", encoding="utf-8")

    # Group by exact instruction.
    instr_lines = ["# Instruction-Grouped Patterns", ""]
    for instruction, counter in sorted(instruction_patterns.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))[: args.top_k]:
        instr_lines.append(f"## {instruction}")
        instr_lines.append(f"- unique_patterns: `{len(counter)}`")
        instr_lines.append(f"- samples: `{sum(counter.values())}`")
        instr_lines.append("")
        instr_lines.extend(top_counter_md(counter, sum(counter.values()), top_k=min(args.top_k, 8)))
        instr_lines.append("")
    (out_dir / "instruction_patterns.md").write_text("\n".join(instr_lines) + "\n", encoding="utf-8")

    # Group by parsed target phrase.
    target_lines = ["# Target-Phrase Patterns", ""]
    for target, counter in sorted(target_patterns.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))[: args.top_k]:
        target_lines.append(f"## {target}")
        target_lines.append(f"- unique_patterns: `{len(counter)}`")
        target_lines.append(f"- samples: `{sum(counter.values())}`")
        target_lines.append("")
        target_lines.extend(top_counter_md(counter, sum(counter.values()), top_k=min(args.top_k, 8)))
        target_lines.append("")
    (out_dir / "target_patterns.md").write_text("\n".join(target_lines) + "\n", encoding="utf-8")

    (out_dir / "patterns.json").write_text(
        json.dumps(
            {
                "global_patterns": {"|".join(k): v for k, v in global_patterns.items()},
                "stage_patterns": {stage: {"|".join(k): v for k, v in counter.items()} for stage, counter in stage_patterns.items()},
                "instruction_patterns": {key: {"|".join(k): v for k, v in counter.items()} for key, counter in instruction_patterns.items()},
                "target_patterns": {key: {"|".join(k): v for k, v in counter.items()} for key, counter in target_patterns.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ok] pattern_analysis_dir={out_dir}")


if __name__ == "__main__":
    main()
