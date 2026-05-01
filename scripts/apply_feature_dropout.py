#!/usr/bin/env python3
"""Construct full/partial feature views with stage-aware dropout policy."""

import argparse
import json
import random
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.visual_evidence_pipeline import VisualEvidenceSchema


DEFAULT_POLICY = {
    "approach": {"bbox": 0.1, "depth": 0.2, "edge": 0.2},
    "grasp": {"bbox": 0.7, "depth": 0.2, "edge": 0.3},
    "place": {"bbox": 0.3, "depth": 0.4, "edge": 0.4},
}


def infer_stage(instruction: str) -> str:
    s = instruction.lower()
    if any(k in s for k in ["grasp", "pick", "grab"]):
        return "grasp"
    if any(k in s for k in ["place", "put", "drop"]):
        return "place"
    return "approach"


def maybe_drop(prob: float) -> bool:
    return random.random() < prob


def unpack_edge(edge_packed: np.ndarray, edge_shape: np.ndarray) -> np.ndarray:
    if edge_shape.size == 0 or int(edge_shape[0]) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    h, w = int(edge_shape[0]), int(edge_shape[1])
    bits = np.unpackbits(edge_packed)[: h * w]
    return (bits.reshape(h, w).astype(np.uint8) * 255)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policy_yaml", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    random.seed(args.seed)
    policy = DEFAULT_POLICY
    if args.policy_yaml:
        with open(args.policy_yaml, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    full_dir = out_dir / "full_npz"
    part_dir = out_dir / "partial_npz"
    full_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = out_dir / "dropout_manifest.jsonl"
    total = sum(1 for _ in open(args.feature_manifest, "r", encoding="utf-8"))
    if total == 0:
        raise RuntimeError(f"Empty feature manifest: {args.feature_manifest}")
    start_ts = time.time()
    print(f"[info] total_samples={total} log_every={args.log_every}", flush=True)

    with open(args.feature_manifest, "r", encoding="utf-8") as fin, out_manifest.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)
            stage = infer_stage(rec["instruction"])
            p = policy.get(stage, {})

            src = np.load(rec["npz_path"], allow_pickle=True)
            bboxes = src["bboxes"]
            scores = src["scores"]
            labels = src["labels"]
            packed_masks = src["packed_masks"]
            mask_shape = src["mask_shape"]
            depth_u16 = src["depth_u16"]
            edge_packed = src["edge_packed"]
            edge_shape = src["edge_shape"]
            # Full features save
            stem = Path(rec["npz_path"]).name
            full_path = full_dir / stem
            np.savez_compressed(
                full_path,
                bboxes=bboxes,
                scores=scores,
                labels=labels,
                packed_masks=packed_masks,
                mask_shape=mask_shape,
                depth_u16=depth_u16,
                edge_packed=edge_packed,
                edge_shape=edge_shape,
            )

            # Partial features with dropout
            use_bbox = not maybe_drop(float(p.get("bbox", 0.0)))
            use_depth = not maybe_drop(float(p.get("depth", 0.0)))
            use_edge = not maybe_drop(float(p.get("edge", 0.0)))

            pbboxes = bboxes if use_bbox else np.zeros((0, 4), dtype=np.float32)
            pscores = scores if use_bbox else np.zeros((0,), dtype=np.float32)
            plabels = labels if use_bbox else np.array([], dtype=object)
            pmasks = packed_masks if use_bbox else np.zeros((0,), dtype=np.uint8)
            pmask_shape = mask_shape if use_bbox else np.array([0, 0, 0], dtype=np.int32)
            pdepth = depth_u16 if use_depth else np.zeros_like(depth_u16)
            pedge = edge_packed if use_edge else np.zeros_like(edge_packed)

            part_path = part_dir / stem
            np.savez_compressed(
                part_path,
                bboxes=pbboxes,
                scores=pscores,
                labels=plabels,
                packed_masks=pmasks,
                mask_shape=pmask_shape,
                depth_u16=pdepth,
                edge_packed=pedge,
                edge_shape=edge_shape,
            )

            full_schema = rec["schema_text"]
            partial_schema = VisualEvidenceSchema.build(
                instruction=rec["instruction"],
                caption=rec["caption"],
                query_words=rec["query_words"],
                detections=[] if not use_bbox else rec["detections"],
                depth=(None if not use_depth else (depth_u16.astype(np.float32) / 65535.0)),
                edge=(None if not use_edge else unpack_edge(edge_packed, edge_shape)),
            )

            out = {
                "episode_idx": rec["episode_idx"],
                "step_idx": rec["step_idx"],
                "stage": stage,
                "instruction": rec["instruction"],
                "image_path": rec["image_path"],
                "action": rec["action"],
                "full_npz_path": str(full_path),
                "partial_npz_path": str(part_path),
                "full_schema_text": full_schema,
                "partial_schema_text": partial_schema,
                "drop_flags": {
                    "bbox": not use_bbox,
                    "depth": not use_depth,
                    "edge": not use_edge,
                },
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            if idx % args.log_every == 0 or idx == total:
                elapsed = max(1e-6, time.time() - start_ts)
                speed = idx / elapsed
                remain = max(0, total - idx)
                eta_sec = int(remain / max(1e-6, speed))
                print(
                    f"[progress] {idx}/{total} ({idx * 100.0 / total:.1f}%) "
                    f"speed={speed:.1f} samples/s eta={eta_sec}s",
                    flush=True,
                )

    print(f"[ok] dropout manifest: {out_manifest}")


if __name__ == "__main__":
    main()
