#!/usr/bin/env python3
"""Benchmark learnable evidence gating against a full-evidence teacher baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    LearnedEvidencePolicy,
    LearnedGatingDataset,
    budget_entropy,
    channel_cost_tensor,
    gate_entropy,
    load_yaml,
    mean_action_l1,
    move_batch_to_device,
    resolve_budget_values,
    resolve_channels,
    resolve_route_bank,
    route_bank_tensor,
    sample_route_mixture_gates,
    sample_budget_topk_gates,
)


def resolved_stage_input(stage_one_hot: torch.Tensor, enabled: bool) -> torch.Tensor:
    return stage_one_hot if enabled else torch.zeros_like(stage_one_hot)


def summarize(rows: list[dict], channels: tuple[str, ...]) -> dict:
    success = np.array([1.0 if r["success"] else 0.0 for r in rows], dtype=np.float32)
    fps = np.array([float(r["fps"]) for r in rows], dtype=np.float32)
    disturbed = np.array([1.0 if r["disturbed"] else 0.0 for r in rows], dtype=np.float32)
    recovered = np.array([1.0 if r.get("recovered", False) else 0.0 for r in rows], dtype=np.float32)
    selected = np.array([r.get("selected_channels", 0.0) for r in rows], dtype=np.float32)
    fallback = np.array([1.0 if r.get("fallback", False) else 0.0 for r in rows], dtype=np.float32)
    out = {
        "n": int(len(rows)),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "inference_fps": float(fps.mean()) if len(rows) else 0.0,
        "robustness_recovery": float(recovered.sum() / max(1.0, disturbed.sum())),
        "avg_selected_channels": float(selected.mean()) if len(rows) else 0.0,
        "fallback_rate": float(fallback.mean()) if len(rows) else 0.0,
        "avg_budget": float(np.mean([r.get("budget", 0.0) for r in rows])) if len(rows) else 0.0,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(np.mean([1.0 if r.get("gates", {}).get(ch, False) else 0.0 for r in rows]))
    return out


def save_report(baseline_rows: list[dict], dynamic_rows: list[dict], out_dir: Path, channels: tuple[str, ...]) -> None:
    b = summarize(baseline_rows, channels)
    d = summarize(dynamic_rows, channels)
    metrics = [
        ("success_rate", b["success_rate"], d["success_rate"]),
        ("inference_fps", b["inference_fps"], d["inference_fps"]),
        ("robustness_recovery", b["robustness_recovery"], d["robustness_recovery"]),
        ("avg_selected_channels", b["avg_selected_channels"], d["avg_selected_channels"]),
        ("fallback_rate", b["fallback_rate"], d["fallback_rate"]),
        ("avg_budget", b["avg_budget"], d["avg_budget"]),
    ]
    for ch in channels:
        metrics.append((f"{ch}_keep_rate", b.get(f"{ch}_keep_rate", 1.0), d.get(f"{ch}_keep_rate", 0.0)))

    md = ["| Metric | Baseline | LearnedGate | Delta |", "|---|---:|---:|---:|"]
    for key, bv, dv in metrics:
        md.append(f"| {key} | {bv:.4f} | {dv:.4f} | {dv-bv:+.4f} |")
    (out_dir / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    chart_metrics = metrics[:4]
    labels = [m[0] for m in chart_metrics]
    bvals = [m[1] for m in chart_metrics]
    dvals = [m[2] for m in chart_metrics]
    x = np.arange(len(labels))
    w = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w / 2, bvals, w, label="Baseline")
    plt.bar(x + w / 2, dvals, w, label="LearnedGate")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Value")
    plt.title("Teacher Baseline vs Learned Evidence Gating")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--config", default="configs/gating_policy_v1.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--disturb_ratio", type=float, default=0.4)
    parser.add_argument("--disturb_scale", type=float, default=0.25)
    parser.add_argument("--fallback_policy", choices=["none", "entropy", "disturbed_if_entropy"], default="disturbed_if_entropy")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    text_cfg = model_cfg.get("text_encoder", {})
    text_encoder_type = str(text_cfg.get("type", "bow"))
    phase_proxy_cfg = cfg.get("phase_proxy", {})
    stage_conditioning_cfg = cfg.get("stage_conditioning", {})
    stage_conditioning_enabled = bool(stage_conditioning_cfg.get("enabled", True))
    latent_phase_cfg = cfg.get("latent_phase", {})
    latent_phase_enabled = bool(latent_phase_cfg.get("enabled", False))
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))
    gate_cfg = cfg["gating"]
    channels = resolve_channels(cfg)
    budget_values = resolve_budget_values(cfg, channels)
    route_bank = resolve_route_bank(cfg, channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel_costs = channel_cost_tensor({ch: cfg["channels"][ch]["cost"] for ch in channels}, channels, device)
    route_bank_masks = route_bank_tensor(route_bank, channels, device) if route_mixture_enabled else None
    rng = np.random.default_rng(args.seed)

    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        limit=int(args.limit),
        channels=channels,
        text_encoder_type=text_encoder_type,
        tokenizer_path=text_cfg.get("tokenizer_path"),
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        phase_proxy_cfg=phase_proxy_cfg,
    )
    model = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        action_dim=int(dataset.action_dim),
        channels=channels,
        budget_values=budget_values,
        text_encoder_type=text_encoder_type,
        text_vocab_size=int(getattr(dataset, "text_vocab_size", 0)),
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type="route_mixture" if route_mixture_enabled else ("latent_phase" if latent_phase_enabled else "stage_conditioned"),
        latent_phase_slots=int(latent_phase_cfg.get("max_phase_slots", 8)),
        route_bank=route_bank,
    ).to(device)
    ckpt_dir = Path(args.checkpoint_dir)
    model.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    if getattr(model, "instruction_text_encoder", None) is not None and (ckpt_dir / "instruction_text_encoder.pt").exists():
        model.instruction_text_encoder.load_state_dict(
            torch.load(ckpt_dir / "instruction_text_encoder.pt", map_location=device)
        )
    if getattr(model, "query_text_encoder", None) is not None and (ckpt_dir / "query_text_encoder.pt").exists():
        model.query_text_encoder.load_state_dict(
            torch.load(ckpt_dir / "query_text_encoder.pt", map_location=device)
        )
    for ch in channels:
        getattr(model, f"{ch}_encoder").load_state_dict(torch.load(ckpt_dir / f"{ch}_encoder.pt", map_location=device))
    model.teacher.load_state_dict(torch.load(ckpt_dir / "teacher.pt", map_location=device))
    model.student.load_state_dict(torch.load(ckpt_dir / "student.pt", map_location=device))
    model.gate.load_state_dict(torch.load(ckpt_dir / "gate.pt", map_location=device))
    model.eval()

    baseline_rows = []
    dynamic_rows = []

    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        batch = dataset[idx]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        batch = move_batch_to_device(batch, device)
        gt = batch["action"]
        disturbed = rng.random() < args.disturb_ratio
        all_channel_cost = float(channel_costs.sum().item())

        with torch.inference_mode():
            t0 = time.time()
            ctx = model.encode_context(batch)
            channel_embeds = model.encode_channels(batch)
            baseline_pred = model.teacher(ctx, channel_embeds)
            base_elapsed = time.time() - t0 + all_channel_cost

            t1 = time.time()
            dyn_ctx = model.encode_context(batch)
            dyn_embeds = model.encode_channels(batch)
            gate_stage = resolved_stage_input(batch["stage_one_hot"], stage_conditioning_enabled)
            gate_out = model.forward_gate(
                dyn_ctx,
                batch,
                gate_stage,
                phase_temperature=float(latent_phase_cfg.get("temperature_end", gate_cfg["temperature_end"])),
                hard_phase=bool(latent_phase_cfg.get("hard_assignment", False)),
                training=False,
            )
            utility_pred = gate_out["utility_pred"]
            phase_probs = gate_out["phase_probs"]
            route_probs = None
            selected_route_idx = None
            if route_bank_masks is not None:
                hard_gates, gate_probs, hard_route, route_probs, hard_budget, budget_probs = sample_route_mixture_gates(
                    gate_out["route_logits"],
                    route_bank_masks,
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )
                selected_route_idx = int(torch.argmax(route_probs, dim=1)[0].item())
            else:
                hard_gates, gate_probs, hard_budget, budget_probs = sample_budget_topk_gates(
                    gate_out["channel_logits"],
                    gate_out["budget_logits"],
                    budget_values=budget_values,
                    temperature=float(gate_cfg["temperature_end"]),
                    training=False,
                )

            noisy_embeds = {}
            for ch in channels:
                emb = dyn_embeds[ch]
                if disturbed:
                    noise = torch.tensor(
                        rng.normal(0.0, args.disturb_scale, size=tuple(emb.shape)).astype(np.float32).tolist(),
                        dtype=torch.float32,
                        device=device,
                    )
                    noisy_embeds[ch] = emb + noise
                else:
                    noisy_embeds[ch] = emb

            student_pred = model.student(dyn_ctx, noisy_embeds, hard_gates, gate_probs)
            selected_cost = float((hard_gates[0] * channel_costs).sum().item())
            dyn_elapsed = time.time() - t1 + selected_cost

            entropy = float((gate_entropy(gate_probs) + budget_entropy(budget_probs)).mean().item())
            missing_channels = int((hard_gates.shape[1] - hard_gates.sum()).item())
            if args.fallback_policy == "entropy":
                use_fallback = (
                    entropy >= float(gate_cfg["fallback_entropy_thresh"])
                    and missing_channels >= int(gate_cfg.get("fallback_min_missing_channels", 1))
                )
            elif args.fallback_policy == "disturbed_if_entropy":
                use_fallback = (
                    disturbed
                    and entropy >= float(gate_cfg["fallback_entropy_thresh"])
                    and missing_channels >= int(gate_cfg.get("fallback_min_missing_channels", 1))
                )
            else:
                use_fallback = False

            if use_fallback:
                t2 = time.time()
                dynamic_pred = model.teacher(dyn_ctx, dyn_embeds)
                missing_cost = float(((1.0 - hard_gates[0]) * channel_costs).sum().item())
                dyn_elapsed += time.time() - t2 + missing_cost
            else:
                dynamic_pred = student_pred

        base_l1 = float(mean_action_l1(baseline_pred, gt).item())
        student_l1 = float(mean_action_l1(student_pred, gt).item())
        dyn_l1 = float(mean_action_l1(dynamic_pred, gt).item())
        recovered = disturbed and use_fallback and (student_l1 > args.success_l1_thresh) and (dyn_l1 <= args.success_l1_thresh)

        baseline_rows.append(
            {
                "idx": idx,
                "stage": rec.stage,
                "success": base_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, base_elapsed),
                "disturbed": bool(disturbed),
                "recovered": False,
                "l1": base_l1,
                "selected_channels": float(len(channels)),
                "budget": float(len(channels)),
                "gates": {ch: True for ch in channels},
            }
        )
        dynamic_rows.append(
            {
                "idx": idx,
                "stage": rec.stage,
                "success": dyn_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, dyn_elapsed),
                "disturbed": bool(disturbed),
                "recovered": bool(recovered),
                "l1": dyn_l1,
                "fallback": bool(use_fallback),
                "selected_channels": float(hard_gates.sum().item()),
                "budget": float(hard_gates.sum().item()),
                "missing_channels": float(missing_channels),
                "gate_entropy": entropy,
                "budget_probs": {str(v): float(budget_probs[0, i].item()) for i, v in enumerate(budget_values)},
                "gates": {ch: bool(hard_gates[0, i].item()) for i, ch in enumerate(channels)},
                "gate_probs": {ch: float(gate_probs[0, i].item()) for i, ch in enumerate(channels)},
                "utility_pred": {ch: float(utility_pred[0, i].item()) for i, ch in enumerate(channels)},
                "phase_probs": {str(i): float(phase_probs[0, i].item()) for i in range(phase_probs.shape[1])} if phase_probs is not None else None,
                "route_idx": selected_route_idx,
                "route_name": ("+".join(route_bank[selected_route_idx]) if (selected_route_idx is not None and selected_route_idx < len(route_bank)) else None),
                "route_probs": {("+".join(route_bank[i])): float(route_probs[0, i].item()) for i in range(route_probs.shape[1])} if route_probs is not None else None,
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in baseline_rows),
        encoding="utf-8",
    )
    (out_dir / "learned_gating_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dynamic_rows),
        encoding="utf-8",
    )
    save_report(baseline_rows, dynamic_rows, out_dir, channels)

    print("[ok] baseline:", summarize(baseline_rows, channels))
    print("[ok] learned_gating:", summarize(dynamic_rows, channels))
    print(f"[ok] report dir: {out_dir}")


if __name__ == "__main__":
    main()
