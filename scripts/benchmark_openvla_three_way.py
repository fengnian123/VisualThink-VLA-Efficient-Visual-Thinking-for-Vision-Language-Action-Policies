#!/usr/bin/env python3
"""Benchmark three actual OpenVLA inference modes on the same feature manifest.

Modes:
1. OpenVLA-only: original instruction prompt, no external evidence.
2. FullEvidence: same OpenVLA model, but with all configured evidence channels serialized into prompt text.
3. DynamicEvidence: same OpenVLA model, but with a learned gate selecting a subset of evidence channels.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    LearnedEvidencePolicy,
    LearnedGatingDataset,
    budget_entropy,
    gate_entropy,
    load_yaml,
    move_batch_to_device,
    resolve_budget_values,
    resolve_channels,
    sample_budget_topk_gates,
    unpack_edge,
)
from utils.relation_features import relation_text


STAGES = ("approach", "grasp", "place")


def make_original_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        sys_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{sys_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def prompt_from_schema(instruction: str, schema_text: str) -> str:
    q = f"{instruction}\nVisual evidence:\n{schema_text}\nReturn robot action."
    return f"In: What action should the robot take to {q}?\nOut:"


def add_disturbance(image: Image.Image, sigma: float = 22.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def action_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def predict_action(vla, processor, image: Image.Image, prompt: str, device: str, dtype, unnorm_key: str):
    inputs = processor(prompt, image).to(device, dtype=dtype)
    start = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    elapsed = time.time() - start
    return np.asarray(action, dtype=np.float32), elapsed


def load_dense_features(npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as npz:
        depth = npz["depth_u16"].astype(np.float32) / 65535.0 if "depth_u16" in npz.files else None
        edge = unpack_edge(npz["edge_packed"], npz["edge_shape"]) if "edge_packed" in npz.files else None
        motion = npz["motion_u8"].astype(np.uint8) if "motion_u8" in npz.files else None
    return depth, edge, motion


def build_schema_from_row(row: dict, channels: tuple[str, ...], gates: dict[str, bool]) -> str:
    detections = row.get("detections", []) if ("bbox" in channels and gates.get("bbox", False)) else []
    depth, edge, motion = load_dense_features(row["npz_path"])
    depth_in = depth if ("depth" in channels and gates.get("depth", False)) else None
    edge_in = edge if ("edge" in channels and gates.get("edge", False)) else None
    motion_in = motion if ("motion" in channels and gates.get("motion", False)) else None
    relation_in = row.get("relation_stats") if ("relation" in channels and gates.get("relation", False)) else None

    det_text = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["bbox"]
        det_text.append(
            f"obj{i}:{d['label']} score={d['score']:.3f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )
    depth_text = "none" if depth_in is None else f"mean={float(depth_in.mean()):.4f},std={float(depth_in.std()):.4f}"
    edge_text = "none" if edge_in is None else f"density={float((edge_in > 0).mean()):.4f}"
    if motion_in is None:
        motion_text = "none"
    else:
        motion_f = motion_in.astype(np.float32) / 255.0
        motion_text = f"mean={float(motion_f.mean()):.4f},std={float(motion_f.std()):.4f},density={float((motion_f > 0.08).mean()):.4f}"
    relation_line = relation_text(relation_in)

    return (
        "<visual_evidence>\n"
        f"instruction: {row['instruction']}\n"
        f"caption: {row['caption']}\n"
        f"queries: {', '.join(row.get('query_words', []))}\n"
        f"detections: {'; '.join(det_text) if det_text else 'none'}\n"
        f"depth: {depth_text}\n"
        f"edge: {edge_text}\n"
        f"motion: {motion_text}\n"
        f"relation: {relation_line}\n"
        "</visual_evidence>"
    )


def summarize(rows: list[dict], channels: tuple[str, ...]) -> dict:
    success = np.array([1.0 if r.get("success", False) else 0.0 for r in rows], dtype=np.float32)
    fps = np.array([float(r.get("fps", 0.0)) for r in rows], dtype=np.float32)
    disturbed = np.array([1.0 if r.get("disturbed", False) else 0.0 for r in rows], dtype=np.float32)
    recovered = np.array([1.0 if r.get("recovered", False) else 0.0 for r in rows], dtype=np.float32)
    selected = np.array([float(r.get("selected_channels", 0.0)) for r in rows], dtype=np.float32)
    fallback = np.array([1.0 if r.get("fallback", False) else 0.0 for r in rows], dtype=np.float32)
    out = {
        "n": int(len(rows)),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "inference_fps": float(fps.mean()) if len(rows) else 0.0,
        "robustness_recovery": float(recovered.sum() / max(1.0, disturbed.sum())),
        "avg_selected_channels": float(selected.mean()) if len(rows) else 0.0,
        "fallback_rate": float(fallback.mean()) if len(rows) else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(np.mean([1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in rows])) if rows else 0.0
    return out


def stage_summary(rows: list[dict], channels: tuple[str, ...]) -> dict[str, dict]:
    out = {}
    for stage in STAGES:
        sub = [r for r in rows if r.get("stage") == stage]
        if not sub:
            continue
        row = {
            "n": len(sub),
            "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in sub])),
            "fps": float(np.mean([float(r["fps"]) for r in sub])),
            "avg_selected_channels": float(np.mean([float(r.get("selected_channels", 0.0)) for r in sub])),
            "fallback_rate": float(np.mean([1.0 if r.get("fallback", False) else 0.0 for r in sub])),
        }
        for ch in channels:
            row[f"{ch}_keep_rate"] = float(np.mean([1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in sub]))
        out[stage] = row
    return out


def write_stage_summary(orig_rows: list[dict], full_rows: list[dict], dyn_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    orig = stage_summary(orig_rows, channels)
    full = stage_summary(full_rows, channels)
    dyn = stage_summary(dyn_rows, channels)
    lines = [
        "| Stage | Model | N | Success | FPS | AvgSelected | Fallback | " + " | ".join(ch.title() for ch in channels) + " |",
        "|---|---|---:|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    for stage in STAGES:
        for name, data in [("OpenVLA", orig.get(stage)), ("FullEvidence", full.get(stage)), ("Dynamic", dyn.get(stage))]:
            if data is None:
                continue
            lines.append(
                f"| {stage} | {name} | {data['n']} | {data['success_rate']:.4f} | {data['fps']:.4f} | "
                f"{data['avg_selected_channels']:.4f} | {data['fallback_rate']:.4f} | "
                + " | ".join(f"{data[f'{ch}_keep_rate']:.4f}" for ch in channels)
                + " |"
            )
    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "stage_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report(orig_rows: list[dict], full_rows: list[dict], dyn_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    orig = summarize(orig_rows, channels)
    full = summarize(full_rows, channels)
    dyn = summarize(dyn_rows, channels)
    metrics = [
        "success_rate",
        "inference_fps",
        "robustness_recovery",
        "avg_selected_channels",
        "fallback_rate",
    ] + [f"{ch}_keep_rate" for ch in channels]

    lines = [
        "| Metric | OpenVLA | FullEvidence | Dynamic | Full-Orig | Dyn-Orig |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in metrics:
        ov = orig.get(key, 0.0)
        fv = full.get(key, 0.0)
        dv = dyn.get(key, 0.0)
        lines.append(f"| {key} | {ov:.4f} | {fv:.4f} | {dv:.4f} | {fv-ov:+.4f} | {dv-ov:+.4f} |")
    (out_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = ["success_rate", "inference_fps", "robustness_recovery"]
    ovals = [orig[k] for k in labels]
    fvals = [full[k] for k in labels]
    dvals = [dyn[k] for k in labels]
    x = np.arange(len(labels))
    w = 0.25
    plt.figure(figsize=(8.4, 4.8))
    plt.bar(x - w, ovals, w, label="OpenVLA")
    plt.bar(x, fvals, w, label="FullEvidence")
    plt.bar(x + w, dvals, w, label="Dynamic")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Value")
    plt.title("Three-Way OpenVLA Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()

    write_stage_summary(orig_rows, full_rows, dyn_rows, out_dir, channels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--disturb_ratio", type=float, default=0.0)
    parser.add_argument("--disturb_scale", type=float, default=22.0)
    parser.add_argument("--fallback_policy", choices=["none", "entropy", "disturbed_if_entropy"], default="none")
    parser.add_argument("--skip_empty_instruction", action="store_true")
    parser.add_argument("--unnorm_key", default="bridge_orig")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_yaml(args.config)
    channels = resolve_channels(cfg)
    budget_values = resolve_budget_values(cfg, channels)
    gate_cfg = cfg["gating"]
    model_cfg = cfg["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    manifest_rows = []
    with open(args.feature_manifest, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                manifest_rows.append(json.loads(line))

    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        limit=0,
        channels=channels,
    )

    policy = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        channels=channels,
        budget_values=budget_values,
    ).to(device)
    ckpt_dir = Path(args.checkpoint_dir)
    policy.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    for ch in channels:
        getattr(policy, f"{ch}_encoder").load_state_dict(torch.load(ckpt_dir / f"{ch}_encoder.pt", map_location=device))
    policy.gate.load_state_dict(torch.load(ckpt_dir / "gate.pt", map_location=device))
    policy.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    candidate_indices = []
    for i, row in enumerate(manifest_rows):
        if args.skip_empty_instruction and not (row.get("instruction") or "").strip():
            continue
        candidate_indices.append(i)
    if args.limit > 0:
        candidate_indices = candidate_indices[: args.limit]
    if not candidate_indices:
        raise RuntimeError("No samples selected for three-way benchmark.")

    orig_rows = []
    full_rows = []
    dyn_rows = []

    for out_idx, idx in enumerate(candidate_indices):
        row = manifest_rows[idx]
        batch = dataset[idx]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        batch = move_batch_to_device(batch, device)
        image = Image.open(row["image_path"]).convert("RGB")
        disturbed = random.random() < args.disturb_ratio
        image_in = add_disturbance(image, sigma=args.disturb_scale) if disturbed else image
        gt = np.asarray(row["action"], dtype=np.float32)

        with torch.inference_mode():
            ctx = policy.encode_context(batch)
            channel_embeds = policy.encode_channels(batch)
            channel_logits, budget_logits = policy.gate(policy.gate_inputs(ctx, batch), batch["stage_one_hot"])
            hard_gates, gate_probs, hard_budget, budget_probs = sample_budget_topk_gates(
                channel_logits,
                budget_logits,
                budget_values=budget_values,
                temperature=float(gate_cfg["temperature_end"]),
                training=False,
            )
            entropy = float((gate_entropy(gate_probs) + budget_entropy(budget_probs)).mean().item())
            gate_map = {ch: bool(hard_gates[0, i].item()) for i, ch in enumerate(channels)}
            missing_channels = len(channels) - int(hard_gates.sum().item())

        orig_prompt = make_original_prompt(row["instruction"], args.model_path)
        orig_action, orig_t = predict_action(vla, processor, image_in, orig_prompt, device, dtype, args.unnorm_key)
        orig_l1 = action_l1(orig_action, gt)
        orig_rows.append(
            {
                "idx": out_idx,
                "stage": dataset.records[idx].stage,
                "success": orig_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, orig_t),
                "disturbed": disturbed,
                "recovered": False,
                "l1": orig_l1,
                "selected_channels": 0.0,
                "fallback": False,
                "gates": {ch: False for ch in channels},
            }
        )

        full_schema = build_schema_from_row(row, channels, {ch: True for ch in channels})
        full_prompt = prompt_from_schema(row["instruction"], full_schema)
        full_action, full_t = predict_action(vla, processor, image_in, full_prompt, device, dtype, args.unnorm_key)
        full_l1 = action_l1(full_action, gt)
        full_rows.append(
            {
                "idx": out_idx,
                "stage": dataset.records[idx].stage,
                "success": full_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, full_t),
                "disturbed": disturbed,
                "recovered": False,
                "l1": full_l1,
                "selected_channels": float(len(channels)),
                "fallback": False,
                "gates": {ch: True for ch in channels},
            }
        )

        dynamic_schema = build_schema_from_row(row, channels, gate_map)
        dynamic_prompt = prompt_from_schema(row["instruction"], dynamic_schema)
        dyn_action, dyn_t = predict_action(vla, processor, image_in, dynamic_prompt, device, dtype, args.unnorm_key)
        dyn_l1 = action_l1(dyn_action, gt)
        pre_fallback_l1 = dyn_l1

        use_fallback = False
        if args.fallback_policy == "entropy":
            use_fallback = entropy >= float(gate_cfg["fallback_entropy_thresh"]) and missing_channels >= int(gate_cfg.get("fallback_min_missing_channels", 1))
        elif args.fallback_policy == "disturbed_if_entropy":
            use_fallback = disturbed and entropy >= float(gate_cfg["fallback_entropy_thresh"]) and missing_channels >= int(gate_cfg.get("fallback_min_missing_channels", 1))

        if use_fallback:
            fb_action, fb_t = predict_action(vla, processor, image_in, full_prompt, device, dtype, args.unnorm_key)
            dyn_action = fb_action
            dyn_t += fb_t
            dyn_l1 = action_l1(dyn_action, gt)

        recovered = disturbed and use_fallback and (pre_fallback_l1 > args.success_l1_thresh) and (dyn_l1 <= args.success_l1_thresh)
        dyn_rows.append(
            {
                "idx": out_idx,
                "stage": dataset.records[idx].stage,
                "success": dyn_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, dyn_t),
                "disturbed": disturbed,
                "recovered": bool(recovered),
                "l1": dyn_l1,
                "fallback": bool(use_fallback),
                "selected_channels": float(hard_gates.sum().item()),
                "gate_entropy": entropy,
                "budget": float(hard_gates.sum().item()),
                "gates": gate_map,
                "gate_probs": {ch: float(gate_probs[0, i].item()) for i, ch in enumerate(channels)},
                "budget_probs": {str(v): float(budget_probs[0, i].item()) for i, v in enumerate(budget_values)},
            }
        )

        if (out_idx + 1) % 10 == 0 or out_idx + 1 == len(candidate_indices):
            print(f"[progress] {out_idx + 1}/{len(candidate_indices)}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "openvla_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in orig_rows), encoding="utf-8")
    (out_dir / "full_evidence_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in full_rows), encoding="utf-8")
    (out_dir / "dynamic_eval.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dyn_rows), encoding="utf-8")
    save_report(orig_rows, full_rows, dyn_rows, out_dir, channels)

    print("[ok] openvla:", summarize(orig_rows, channels))
    print("[ok] full_evidence:", summarize(full_rows, channels))
    print("[ok] dynamic:", summarize(dyn_rows, channels))
    print(f"[ok] report dir: {out_dir}")


if __name__ == "__main__":
    main()
