#!/usr/bin/env python3
"""Build OpenVLA/Fast-ECoT-compatible dataset_statistics.json from an extracted manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compute_stats(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to 01_extract/manifest.jsonl")
    parser.add_argument("--dataset-key", required=True, help="Top-level key to write in dataset_statistics.json")
    parser.add_argument("--output", required=True, help="Path to output dataset_statistics.json")
    parser.add_argument(
        "--absolute-last-action",
        action="store_true",
        help="Mark the last action dimension as absolute / not normalized.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    actions = []
    episode_ids = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "episode_idx" in row:
                episode_ids.add(int(row["episode_idx"]))
            action = row.get("action")
            if action is None:
                continue
            actions.append(action)

    if not actions:
        raise RuntimeError(f"No `action` rows found in {manifest_path}")

    action_arr = np.asarray(actions, dtype=np.float32)
    if action_arr.ndim != 2:
        raise RuntimeError(f"Expected 2D action array, got shape={action_arr.shape}")

    action_stats = compute_stats(action_arr)
    if args.absolute_last_action:
        mask = [True] * action_arr.shape[1]
        mask[-1] = False
        action_stats["mask"] = mask

    # Manifests do not currently carry proprio. We keep a compatible placeholder.
    proprio_arr = np.zeros_like(action_arr, dtype=np.float32)
    proprio_stats = compute_stats(proprio_arr)

    payload = {
        args.dataset_key: {
            "action": action_stats,
            "proprio": proprio_stats,
            "num_transitions": int(action_arr.shape[0]),
            "num_trajectories": int(len(episode_ids)),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output_path}")


if __name__ == "__main__":
    main()
