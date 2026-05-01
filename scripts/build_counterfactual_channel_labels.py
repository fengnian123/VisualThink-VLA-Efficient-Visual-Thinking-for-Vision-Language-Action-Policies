#!/usr/bin/env python3
"""Build per-sample counterfactual channel-utility labels for learned evidence gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    LearnedEvidencePolicy,
    LearnedGatingDataset,
    build_counterfactual_utility_map,
    load_yaml,
    resolve_budget_values,
    resolve_channels,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    channels = resolve_channels(cfg)
    budget_values = resolve_budget_values(cfg, channels)
    utility_cfg = cfg.get("utility", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        limit=int(args.limit),
        channels=channels,
    )
    model = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        channels=channels,
        budget_values=budget_values,
    ).to(device)

    ckpt_dir = Path(args.checkpoint_dir)
    model.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    for ch in channels:
        getattr(model, f"{ch}_encoder").load_state_dict(torch.load(ckpt_dir / f"{ch}_encoder.pt", map_location=device))
    model.teacher.load_state_dict(torch.load(ckpt_dir / "teacher.pt", map_location=device))
    model.eval()

    utility_map = build_counterfactual_utility_map(
        model,
        dataset,
        device=device,
        channels=channels,
        channel_costs=[cfg["channels"][ch]["cost"] for ch in channels],
        budget_values=budget_values,
        batch_size=int(utility_cfg.get("batch_size", 64)),
        relative_keep_thresh=float(utility_cfg.get("relative_keep_thresh", 0.35)),
        absolute_score_thresh=float(utility_cfg.get("absolute_score_thresh", 1e-4)),
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in dataset.records:
            key = f"{rec.episode_idx}:{rec.step_idx}:{rec.npz_path}"
            if key in utility_map:
                f.write(json.dumps(utility_map[key], ensure_ascii=False) + "\n")
    print(f"[ok] wrote={out_path}")
    print(f"[ok] rows={len(utility_map)}")


if __name__ == "__main__":
    main()
