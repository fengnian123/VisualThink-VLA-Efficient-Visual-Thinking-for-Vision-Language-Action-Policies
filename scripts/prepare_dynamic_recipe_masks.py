#!/usr/bin/env python3
"""Prepare sampled manifests and hard/soft/blended routing masks for recipe ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_openvla_soft_evidence import compute_dynamic_masks  # noqa: E402


def load_rows(path: Path) -> list[str]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"empty manifest: {path}")
    return lines


def choose_indices(n: int, sample_count: int, seed: int, strategy: str) -> list[int]:
    if sample_count <= 0 or sample_count >= n:
        return list(range(n))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), sample_count))
    if strategy == "stride":
        return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, num=sample_count)))
    raise ValueError(f"unknown sample strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--dynamic_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=4000)
    parser.add_argument("--sample_strategy", choices=("stride", "random"), default="stride")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alphas", default="0.20,0.35,0.50")
    parser.add_argument("--gate_batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    feature_manifest = Path(args.feature_manifest)
    ckpt_dir = Path(args.dynamic_checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(resolved["channels"])
    gate_checkpoint_dir = resolved.get("gate_checkpoint_dir")
    gate_config = resolved.get("gate_config") or resolved.get("config_path") or ""
    if not gate_checkpoint_dir:
        raise RuntimeError(f"missing gate_checkpoint_dir in {ckpt_dir / 'resolved_config.json'}")

    lines = load_rows(feature_manifest)
    indices = choose_indices(len(lines), args.sample_count, args.seed, args.sample_strategy)
    sample_manifest = out_dir / "sample_manifest.jsonl"
    sample_manifest.write_text("\n".join(lines[i] for i in indices) + "\n", encoding="utf-8")

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[info] sample_manifest={sample_manifest} rows={len(indices)} device={device}", flush=True)
    print("[stage] compute hard sparse masks", flush=True)
    hard = compute_dynamic_masks(
        str(sample_manifest),
        channels=channels,
        gate_checkpoint_dir=str(gate_checkpoint_dir),
        gate_config=str(gate_config),
        device=device,
        batch_size=args.gate_batch_size,
        mask_mode="hard",
    )
    print("[stage] compute soft masks", flush=True)
    soft = compute_dynamic_masks(
        str(sample_manifest),
        channels=channels,
        gate_checkpoint_dir=str(gate_checkpoint_dir),
        gate_config=str(gate_config),
        device=device,
        batch_size=args.gate_batch_size,
        mask_mode="soft",
    )

    mask_paths: dict[str, str] = {}
    hard_path = out_dir / "hard_sparse.npy"
    soft_path = out_dir / "soft_gate.npy"
    np.save(hard_path, hard.astype(np.float32))
    np.save(soft_path, soft.astype(np.float32))
    mask_paths["hard_sparse"] = str(hard_path)
    mask_paths["soft_gate"] = str(soft_path)

    for raw_alpha in args.alphas.split(","):
        raw_alpha = raw_alpha.strip()
        if not raw_alpha:
            continue
        alpha = float(raw_alpha)
        blend = np.clip((1.0 - alpha) * hard + alpha * soft, 0.0, 1.0).astype(np.float32)
        name = f"blend_alpha_{alpha:.2f}".replace(".", "p")
        path = out_dir / f"{name}.npy"
        np.save(path, blend)
        mask_paths[name] = str(path)

    (out_dir / "mask_paths.json").write_text(json.dumps(mask_paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "feature_manifest": str(feature_manifest),
        "sample_manifest": str(sample_manifest),
        "dynamic_checkpoint_dir": str(ckpt_dir),
        "gate_checkpoint_dir": str(gate_checkpoint_dir),
        "gate_config": str(gate_config),
        "channels": channels,
        "sample_count": len(indices),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "alphas": args.alphas,
    }
    (out_dir / "recipe_mask_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] mask_paths={out_dir / 'mask_paths.json'}", flush=True)


if __name__ == "__main__":
    main()
