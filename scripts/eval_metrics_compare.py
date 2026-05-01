#!/usr/bin/env python3
"""Compare baseline OpenVLA vs dynamic-thinking metrics."""

import argparse
import json

import numpy as np


def load_records(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict]) -> dict:
    success = np.array([1.0 if r.get("success", False) else 0.0 for r in rows], dtype=np.float32)
    fps = np.array([float(r.get("fps", 0.0)) for r in rows], dtype=np.float32)
    recovered = np.array([1.0 if r.get("recovered", False) else 0.0 for r in rows], dtype=np.float32)
    disturbed = np.array([1.0 if r.get("disturbed", False) else 0.0 for r in rows], dtype=np.float32)
    disturbed_count = max(1.0, disturbed.sum())
    return {
        "n": int(len(rows)),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "inference_fps": float(fps.mean()) if len(rows) else 0.0,
        "robustness_recovery": float(recovered.sum() / disturbed_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_jsonl", required=True)
    parser.add_argument("--dynamic_jsonl", required=True)
    args = parser.parse_args()

    b = summarize(load_records(args.baseline_jsonl))
    d = summarize(load_records(args.dynamic_jsonl))

    print("| Metric | Baseline OpenVLA | Dynamic Thinking | Delta |")
    print("|---|---:|---:|---:|")
    for k in ["success_rate", "inference_fps", "robustness_recovery"]:
        delta = d[k] - b[k]
        print(f"| {k} | {b[k]:.4f} | {d[k]:.4f} | {delta:+.4f} |")


if __name__ == "__main__":
    main()
