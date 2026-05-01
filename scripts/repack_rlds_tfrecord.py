#!/usr/bin/env python3
"""Repack action, instruction, full/partial evidence into RLDS-like TFRecord."""

import argparse
import json
from pathlib import Path

import tensorflow as tf


def b(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def bl(vs: list[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=vs))


def fl(vs: list[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=vs))


def il(vs: list[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=vs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_manifest", required=True)
    parser.add_argument("--output_tfrecord", required=True)
    args = parser.parse_args()

    out_path = Path(args.output_tfrecord)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with tf.io.TFRecordWriter(str(out_path)) as writer, open(args.dropout_manifest, "r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            image_bytes = Path(rec["image_path"]).read_bytes()
            full_schema = rec["full_schema_text"].encode("utf-8")
            partial_schema = rec["partial_schema_text"].encode("utf-8")
            instruction = rec["instruction"].encode("utf-8")
            action = [float(x) for x in rec["action"]]

            # RLDS-like one-step episode sample.
            feat = {
                "steps/observation/image_0": bl([image_bytes]),
                "steps/language_instruction": bl([instruction]),
                "steps/full_schema_text": bl([full_schema]),
                "steps/partial_schema_text": bl([partial_schema]),
                "steps/action": fl(action),
                "steps/is_first": il([1]),
                "steps/is_last": il([1]),
                "steps/is_terminal": il([1]),
                "steps/reward": fl([0.0]),
                "steps/discount": fl([1.0]),
                "episode_metadata/episode_id": b(
                    f"ep{rec['episode_idx']:06d}_s{rec['step_idx']:04d}".encode("utf-8")
                ),
                "episode_metadata/stage": b(str(rec["stage"]).encode("utf-8")),
            }
            ex = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(ex.SerializeToString())
            n += 1

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps({"num_records": n, "format": "rlds_like"}, indent=2), encoding="utf-8")
    print(f"[ok] tfrecord: {out_path}")
    print(f"[ok] records: {n}")
    print(f"[ok] meta: {meta_path}")


if __name__ == "__main__":
    main()
