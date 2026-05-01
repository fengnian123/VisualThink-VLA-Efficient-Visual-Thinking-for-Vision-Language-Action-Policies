#!/usr/bin/env python3
"""Build trainable mini RLDS datasets by trimming shard metadata to existing shards."""

import argparse
import json
import re
import shutil
from pathlib import Path

import tensorflow as tf


def count_records(tfrecord_path: Path) -> int:
    n = 0
    for _ in tf.data.TFRecordDataset([str(tfrecord_path)]):
        n += 1
    return n


def build_one(src_dir: Path, dst_dir: Path, shard_glob: str) -> None:
    src_info = src_dir / "dataset_info.json"
    src_feat = src_dir / "features.json"
    if not src_info.exists() or not src_feat.exists():
        raise FileNotFoundError(f"Missing dataset_info/features under {src_dir}")

    shards = sorted(src_dir.glob(shard_glob))
    if len(shards) == 0:
        raise RuntimeError(f"No shard files matched: {src_dir}/{shard_glob}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_feat, dst_dir / "features.json")

    shard_lengths = []
    num_bytes = 0
    m = re.match(r"^(.*)-(\d+)-of-(\d+)$", shards[0].name)
    if m is None:
        raise RuntimeError(f"Unrecognized shard filename format: {shards[0].name}")
    prefix = m.group(1)
    shard_w = len(m.group(2))
    total_w = len(m.group(3))
    new_total = len(shards)

    for i, s in enumerate(shards):
        dst_name = f"{prefix}-{i:0{shard_w}d}-of-{new_total:0{total_w}d}"
        dst_s = dst_dir / dst_name
        shutil.copy2(s, dst_s)
        shard_lengths.append(str(count_records(dst_s)))
        num_bytes += dst_s.stat().st_size

    info = json.loads(src_info.read_text(encoding="utf-8"))
    if not info.get("splits"):
        raise RuntimeError(f"dataset_info has empty splits: {src_info}")
    train_split = None
    for s in info["splits"]:
        if s.get("name") == "train":
            train_split = s
            break
    if train_split is None:
        train_split = info["splits"][0]
        train_split["name"] = "train"
    train_split["shardLengths"] = shard_lengths
    train_split["numBytes"] = str(num_bytes)
    # Keep only train split for mini subset to prevent missing shard lookups on val/test.
    info["splits"] = [train_split]
    (dst_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[ok] built mini dataset: {dst_dir} | shards={len(shards)} | "
        f"records={sum(int(x) for x in shard_lengths)} | bytes={num_bytes}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge_src", required=True)
    parser.add_argument("--libero_src", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()

    tf.config.set_visible_devices([], "GPU")

    out_root = Path(args.output_root)

    # Bridge mini: use available first two shards.
    build_one(
        src_dir=Path(args.bridge_src),
        dst_dir=out_root / "bridge_orig"/ "1.0.0",
        shard_glob="bridge_dataset-train.tfrecord-0000[0-1]-of-01024",
    )

    # LIBERO mini: use two real shards we pulled.
    build_one(
        src_dir=Path(args.libero_src),
        dst_dir=out_root / "libero_10_no_noops"/ "1.0.0",
        shard_glob="liber_o10-train.tfrecord-0000[0-1]-of-00032",
    )


if __name__ == "__main__":
    main()
