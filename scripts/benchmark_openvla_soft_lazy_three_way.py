#!/usr/bin/env python3
"""Benchmark OpenVLA/FullSoft/DynamicSoft with lazy channel loading.

This is a first-step implementation of "gate first, then lazy evidence
extraction". Instead of materializing every channel for every sample, the
DynamicSoft branch only loads channel vectors selected by its route mask.

The route itself is read from a trained DynamicSoft checkpoint
(`channel_masks.npy`) so that we can validate the lazy extraction pipeline
without retraining a new gate yet. A side-path evidence trace is written for
interpretability, but the action decoder does not need to autoregressively
emit explanation tokens before predicting the action.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    FEATURE_DIMS,
    STAGES,
    bbox_vector,
    edge_vector,
    infer_stage,
    motion_vector,
    relation_vector,
)
from models.openvla_soft_evidence import (
    SoftEvidenceAdapter,
    SoftEvidenceBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)


def load_rows(manifest_path: str) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def load_adapter(
    checkpoint_dir: str,
    channels: tuple[str, ...],
    hidden_size: int,
    device: torch.device,
) -> tuple[SoftEvidenceAdapter, np.ndarray, dict]:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    adapter_cfg = resolved["config"]["adapter"]
    adapter = SoftEvidenceAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(adapter_cfg["num_global_tokens"]),
        proj_dim=int(adapter_cfg["proj_dim"]),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(ckpt_dir / "adapter.pt", map_location=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing adapter keys when loading {ckpt_dir}: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected adapter keys when loading {ckpt_dir}: {unexpected}", flush=True)
    adapter.eval()
    masks = np.load(ckpt_dir / "channel_masks.npy")
    return adapter, masks, resolved


def action_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32))))


def predict_action_original(vla, processor, image: Image.Image, instruction: str, model_path: str, device: str, dtype, unnorm_key: str):
    prompt = make_openvla_prompt(instruction, model_path)
    inputs = processor(prompt, image).to(device, dtype=dtype)
    start = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32), time.time() - start


def stage_one_hot(stage: str, device: torch.device) -> torch.Tensor:
    vec = torch.zeros((1, len(STAGES)), dtype=torch.float32, device=device)
    vec[0, STAGES.index(stage)] = 1.0
    return vec


def build_step_ratio_map(rows: list[dict]) -> dict[tuple[int, int, str], float]:
    step_max = {}
    for row in rows:
        ep = int(row["episode_idx"])
        step_max[ep] = max(step_max.get(ep, 0), int(row["step_idx"]))
    out = {}
    for row in rows:
        ep = int(row["episode_idx"])
        step = int(row["step_idx"])
        ratio = float(step) / float(max(1, step_max[ep]))
        out[(ep, step, str(row["npz_path"]))] = ratio
    return out


def expand_backend_dependencies(route_mask: dict[str, bool]) -> dict[str, bool]:
    plan = dict(route_mask)
    if plan.get("relation", False):
        plan["bbox"] = True
    return plan


def load_channel_features_lazy(
    row: dict,
    channels: tuple[str, ...],
    route_mask: dict[str, bool],
) -> tuple[dict[str, np.ndarray], float]:
    start = time.time()
    plan = expand_backend_dependencies(route_mask)
    features = {ch: np.zeros((FEATURE_DIMS[ch],), dtype=np.float32) for ch in channels}
    with np.load(row["npz_path"], allow_pickle=True) as npz:
        if plan.get("bbox", False) and "bbox" in channels:
            features["bbox"] = bbox_vector(npz)
        if plan.get("edge", False) and "edge" in channels:
            features["edge"] = edge_vector(npz)
        if plan.get("motion", False) and "motion" in channels:
            features["motion"] = motion_vector(npz)
        if plan.get("relation", False) and "relation" in channels:
            features["relation"] = relation_vector(npz, row=row)
    return features, time.time() - start


def load_channel_features_full(row: dict, channels: tuple[str, ...]) -> tuple[dict[str, np.ndarray], float]:
    return load_channel_features_lazy(row=row, channels=channels, route_mask={ch: True for ch in channels})


def build_evidence_batch(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    channels: tuple[str, ...],
    stage: str,
    step_ratio: float,
    device: torch.device,
) -> SoftEvidenceBatch:
    channel_features = {
        ch: torch.tensor(features[ch], dtype=torch.float32, device=device).unsqueeze(0)
        for ch in channels
    }
    return SoftEvidenceBatch(
        channel_features=channel_features,
        channel_mask=torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
        stage_one_hot=stage_one_hot(stage, device=device),
        step_ratio=torch.tensor([[float(step_ratio)]], dtype=torch.float32, device=device),
    )


def route_trace(
    row: dict,
    route_mask: dict[str, bool],
    features: dict[str, np.ndarray],
    stage: str,
) -> dict:
    selected = [ch for ch, on in route_mask.items() if on]
    if not selected:
        selected = ["edge"]
    snippets = {}
    if "bbox" in features:
        b = features["bbox"]
        snippets["bbox"] = f"count={b[0]:.2f}, mean_score={b[1]:.2f}, top_center=({b[8]:.2f},{b[9]:.2f})"
    if "edge" in features:
        e = features["edge"]
        snippets["edge"] = f"density={e[0]:.3f}, left={e[3]:.3f}, right={e[4]:.3f}"
    if "motion" in features:
        m = features["motion"]
        snippets["motion"] = f"mean={m[0]:.3f}, density={m[2]:.3f}, center={m[3]:.3f}"
    if "relation" in features:
        r = features["relation"]
        snippets["relation"] = f"target_center=({r[4]:.2f},{r[5]:.2f}), goal_dist={r[17]:.3f}, nearest_other={r[8]:.3f}"
    rationale_bits = [f"{ch} [{snippets.get(ch, 'n/a')}]" for ch in selected]
    return {
        "selected_evidence": selected,
        "visual_rationale": f"At the {stage} stage, selected channels are {', '.join(selected)}: " + "; ".join(rationale_bits),
        "action_intent": row.get("instruction", ""),
    }


@torch.inference_mode()
def predict_soft(
    vla,
    processor,
    tokenizer,
    adapter: SoftEvidenceAdapter,
    image: Image.Image,
    row: dict,
    evidence_batch: SoftEvidenceBatch,
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    unnorm_key: str,
) -> tuple[np.ndarray, float]:
    start = time.time()
    action = predict_action_with_soft_evidence(
        vla=vla,
        processor=processor,
        tokenizer=tokenizer,
        adapter=adapter,
        image=image,
        instruction=row.get("instruction", ""),
        evidence_batch=evidence_batch,
        model_path=model_path,
        device=device,
        dtype=dtype,
        unnorm_key=unnorm_key,
    )
    return np.asarray(action, dtype=np.float32), time.time() - start


def summarize(rows: list[dict], channels: tuple[str, ...]) -> dict:
    out = {
        "n": int(len(rows)),
        "success_rate": float(np.mean([1.0 if row.get("success") else 0.0 for row in rows])) if rows else 0.0,
        "action_fps": float(np.mean([float(row.get("action_fps", 0.0)) for row in rows])) if rows else 0.0,
        "end2end_fps": float(np.mean([float(row.get("end2end_fps", 0.0)) for row in rows])) if rows else 0.0,
        "avg_selected_channels": float(np.mean([float(row.get("selected_channels", 0.0)) for row in rows])) if rows else 0.0,
        "avg_extract_time_ms": 1000.0 * float(np.mean([float(row.get("extract_time", 0.0)) for row in rows])) if rows else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(np.mean([1.0 if row.get("gates", {}).get(ch, False) else 0.0 for row in rows])) if rows else 0.0
    return out


def stage_summary(rows: list[dict], channels: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for stage in STAGES:
        sub = [row for row in rows if row.get("stage") == stage]
        if sub:
            out[stage] = summarize(sub, channels)
    return out


def write_reports(
    orig_rows: list[dict],
    full_rows: list[dict],
    lazy_rows: list[dict],
    out_dir: Path,
    channels: tuple[str, ...],
) -> None:
    orig = summarize(orig_rows, channels)
    full = summarize(full_rows, channels)
    lazy = summarize(lazy_rows, channels)
    metrics = [
        "success_rate",
        "action_fps",
        "end2end_fps",
        "avg_extract_time_ms",
        "avg_selected_channels",
    ] + [f"{ch}_keep_rate" for ch in channels]
    lines = [
        "| Metric | OpenVLA | FullSoft | LazyDynamicSoft | Full-Orig | Lazy-Orig |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in metrics:
        ov = float(orig.get(key, 0.0))
        fv = float(full.get(key, 0.0))
        lv = float(lazy.get(key, 0.0))
        lines.append(f"| {key} | {ov:.4f} | {fv:.4f} | {lv:.4f} | {fv - ov:+.4f} | {lv - ov:+.4f} |")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    stage_lines = [
        "| Stage | Model | N | Success | ActionFPS | End2EndFPS | ExtractMS | AvgSelected | "
        + " | ".join(ch.title() for ch in channels)
        + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    stage_maps = {
        "OpenVLA": stage_summary(orig_rows, channels),
        "FullSoft": stage_summary(full_rows, channels),
        "LazyDynamicSoft": stage_summary(lazy_rows, channels),
    }
    for stage in STAGES:
        for model_name in ["OpenVLA", "FullSoft", "LazyDynamicSoft"]:
            row = stage_maps[model_name].get(stage)
            if not row:
                continue
            stage_lines.append(
                f"| {stage} | {model_name} | {row['n']} | {row['success_rate']:.4f} | "
                f"{row['action_fps']:.4f} | {row['end2end_fps']:.4f} | {row['avg_extract_time_ms']:.4f} | "
                f"{row['avg_selected_channels']:.4f} | "
                + " | ".join(f"{row[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "stage_summary.md").write_text("\n".join(stage_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--full_checkpoint_dir", required=True)
    parser.add_argument("--dynamic_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.feature_manifest)
    full_resolved = json.loads((Path(args.full_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(full_resolved["channels"])
    step_ratio_map = build_step_ratio_map(rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    hidden_size = int(vla.get_input_embeddings().weight.shape[1])

    full_adapter, full_masks, _ = load_adapter(args.full_checkpoint_dir, channels, hidden_size, device)
    lazy_adapter, dyn_masks, _ = load_adapter(args.dynamic_checkpoint_dir, channels, hidden_size, device)

    candidate_indices = list(range(len(rows)))
    if args.limit > 0:
        candidate_indices = candidate_indices[: args.limit]

    orig_rows = []
    full_rows = []
    lazy_rows = []

    for out_idx, idx in enumerate(candidate_indices):
        row = rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(row["action"], dtype=np.float32)
        key = (int(row["episode_idx"]), int(row["step_idx"]), str(row["npz_path"]))
        step_ratio = float(step_ratio_map[key])
        stage = infer_stage(step_ratio)

        orig_action, orig_t = predict_action_original(
            vla=vla,
            processor=processor,
            image=image,
            instruction=row.get("instruction", ""),
            model_path=args.model_path,
            device=str(device),
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        orig_l1 = action_l1(orig_action, gt)
        orig_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": orig_l1 <= float(args.success_l1_thresh),
                "l1": orig_l1,
                "action_fps": 1.0 / max(1e-6, orig_t),
                "end2end_fps": 1.0 / max(1e-6, orig_t),
                "extract_time": 0.0,
                "selected_channels": 0.0,
                "gates": {ch: False for ch in channels},
            }
        )

        full_features, full_extract_t = load_channel_features_full(row, channels)
        full_batch = build_evidence_batch(
            features=full_features,
            mask=np.asarray(full_masks[idx], dtype=np.float32),
            channels=channels,
            stage=stage,
            step_ratio=step_ratio,
            device=device,
        )
        full_action, full_action_t = predict_soft(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=full_adapter,
            image=image,
            row=row,
            evidence_batch=full_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        full_total_t = full_extract_t + full_action_t
        full_l1 = action_l1(full_action, gt)
        full_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": full_l1 <= float(args.success_l1_thresh),
                "l1": full_l1,
                "action_fps": 1.0 / max(1e-6, full_action_t),
                "end2end_fps": 1.0 / max(1e-6, full_total_t),
                "extract_time": full_extract_t,
                "selected_channels": float(np.sum(full_masks[idx] > args.mask_threshold)),
                "gates": {ch: bool(full_masks[idx][i] > args.mask_threshold) for i, ch in enumerate(channels)},
            }
        )

        lazy_mask = np.asarray(dyn_masks[idx], dtype=np.float32)
        lazy_gates = {ch: bool(lazy_mask[i] > args.mask_threshold) for i, ch in enumerate(channels)}
        lazy_features, lazy_extract_t = load_channel_features_lazy(
            row=row,
            channels=channels,
            route_mask=lazy_gates,
        )
        lazy_batch = build_evidence_batch(
            features=lazy_features,
            mask=lazy_mask,
            channels=channels,
            stage=stage,
            step_ratio=step_ratio,
            device=device,
        )
        lazy_action, lazy_action_t = predict_soft(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=lazy_adapter,
            image=image,
            row=row,
            evidence_batch=lazy_batch,
            model_path=args.model_path,
            device=device,
            dtype=dtype,
            unnorm_key=args.unnorm_key,
        )
        lazy_total_t = lazy_extract_t + lazy_action_t
        lazy_l1 = action_l1(lazy_action, gt)
        lazy_trace = route_trace(row=row, route_mask=lazy_gates, features=lazy_features, stage=stage)
        lazy_rows.append(
            {
                "idx": out_idx,
                "stage": stage,
                "success": lazy_l1 <= float(args.success_l1_thresh),
                "l1": lazy_l1,
                "action_fps": 1.0 / max(1e-6, lazy_action_t),
                "end2end_fps": 1.0 / max(1e-6, lazy_total_t),
                "extract_time": lazy_extract_t,
                "selected_channels": float(np.sum(lazy_mask > args.mask_threshold)),
                "gates": lazy_gates,
                **lazy_trace,
            }
        )

        if (out_idx + 1) % 10 == 0 or (out_idx + 1) == len(candidate_indices):
            print(f"[progress] {out_idx + 1}/{len(candidate_indices)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openvla_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in orig_rows),
        encoding="utf-8",
    )
    (out_dir / "full_soft_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in full_rows),
        encoding="utf-8",
    )
    (out_dir / "lazy_dynamic_soft_eval.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lazy_rows),
        encoding="utf-8",
    )
    write_reports(orig_rows, full_rows, lazy_rows, out_dir, channels)
    print("[ok] openvla:", summarize(orig_rows, channels))
    print("[ok] full_soft:", summarize(full_rows, channels))
    print("[ok] lazy_dynamic_soft:", summarize(lazy_rows, channels))
    print(f"[ok] report dir: {out_dir}")


if __name__ == "__main__":
    main()
