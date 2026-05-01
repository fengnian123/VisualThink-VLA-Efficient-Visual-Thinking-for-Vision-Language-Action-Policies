#!/usr/bin/env python3
"""Build a trainable RLDS train subset from an official dataset directory."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path


def resolve_train_split(info: dict) -> dict:
    splits = info.get("splits", [])
    if not splits:
        raise RuntimeError("dataset_info.json has no splits")
    for split in splits:
        if split.get("name") == "train":
            return split
    return splits[0]


def parse_shard_index(name: str) -> int:
    m = re.match(r"^.*-(\d+)-of-(\d+)$", name)
    if m is None:
        raise RuntimeError(f"unexpected shard filename: {name}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--shard-prefix", required=True)
    args = parser.parse_args()

    src_dir = Path(args.dataset_dir)
    dst_dir = Path(args.output_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"dataset dir not found: {src_dir}")

    info_path = src_dir / "dataset_info.json"
    features_path = src_dir / "features.json"
    if not info_path.exists() or not features_path.exists():
        raise FileNotFoundError(f"missing dataset_info/features under {src_dir}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    train_split = resolve_train_split(info)
    total_shards = len(train_split["shardLengths"])
    total_shards_pad = f"{total_shards:05d}"

    shard_pattern = f"{args.shard_prefix}-*-of-{total_shards_pad}"
    shards = sorted(src_dir.glob(shard_pattern))
    effective_shards = total_shards
    effective_lengths = list(train_split["shardLengths"])
    if len(shards) != total_shards:
        indices = [parse_shard_index(path.name) for path in shards]
        expected = list(range(len(shards)))
        if indices != expected:
            raise RuntimeError(
                f"shard mismatch for pattern {shard_pattern}: "
                f"dataset_info={total_shards}, files={len(shards)}, "
                "and local shards are not a contiguous prefix"
            )
        effective_shards = len(shards)
        effective_lengths = effective_lengths[:effective_shards]
        print(
            f"[warn] partial local dataset detected: dataset_info={total_shards}, "
            f"local_files={len(shards)}; using contiguous prefix 0..{effective_shards - 1}"
        )

    n = max(1, math.ceil(effective_shards * args.fraction))
    selected = shards[:n]

    dst_dir.mkdir(parents=True, exist_ok=True)
    m = re.match(r"^(.*)-(\d+)-of-(\d+)$", selected[0].name)
    if m is None:
        raise RuntimeError(f"unexpected shard filename: {selected[0].name}")
    prefix = m.group(1)
    idx_width = len(m.group(2))
    total_width = len(m.group(3))

    for i, src in enumerate(selected):
        dst_name = f"{prefix}-{i:0{idx_width}d}-of-{n:0{total_width}d}"
        dst = dst_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    for sidecar in src_dir.glob("*.json"):
        if sidecar.name in {"dataset_info.json", "features.json"}:
            continue
        dst = dst_dir / sidecar.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(sidecar, dst)

    dst_features = dst_dir / "features.json"
    if dst_features.exists() or dst_features.is_symlink():
        dst_features.unlink()
    os.symlink(features_path, dst_features)

    new_train = dict(train_split)
    new_train["name"] = "train"
    new_train["shardLengths"] = effective_lengths[:n]
    new_train["numBytes"] = str(sum(p.stat().st_size for p in selected))
    info["splits"] = [new_train]
    (dst_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[ok] built subset: {dst_dir} | total_shards={effective_shards} | "
        f"selected_shards={n} | total_episodes={sum(int(x) for x in new_train['shardLengths'])}"
    )


if __name__ == "__main__":
    main()
