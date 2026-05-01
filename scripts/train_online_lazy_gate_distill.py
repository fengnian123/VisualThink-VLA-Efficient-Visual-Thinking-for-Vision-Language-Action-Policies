#!/usr/bin/env python3
"""Distill an online LazyGate from an existing DynamicSoft route checkpoint.

The student gate only consumes cheap context:
- low-resolution RGB frame
- instruction/query text
- step ratio
- instruction metadata

It does NOT consume bbox/edge/motion/relation vectors at routing time. The
target route mask comes from a trained DynamicSoft checkpoint's
`channel_masks.npy`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    INSTANCE_AMBIGUITY_DIM,
    LearnedEvidencePolicy,
    hashed_bow,
    infer_stage,
    instruction_meta_vector,
    load_local_tokenizer,
    load_yaml,
    move_batch_to_device,
    resolve_budget_values,
    resolve_channels,
    route_bank_tensor,
    sample_budget_topk_gates,
    sample_route_mixture_gates,
    stage_to_one_hot,
    tokenize_to_arrays,
)
from scripts.train_learned_gating import set_seed, split_indices


def load_rows(manifest_path: str, limit: int = 0) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def load_image_tensor(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr.tolist(), dtype=torch.float32)


class OnlineLazyGateDistillDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        teacher_masks: np.ndarray,
        image_size: int,
        bow_dim: int,
        query_bow_dim: int,
        channels: tuple[str, ...],
        text_encoder_type: str,
        tokenizer_path: str | None,
        instruction_max_len: int,
        query_max_len: int,
        limit: int = 0,
    ) -> None:
        self.rows = load_rows(manifest_path, limit=limit)
        if teacher_masks.shape[0] < len(self.rows):
            raise RuntimeError(f"teacher mask rows {teacher_masks.shape[0]} < data rows {len(self.rows)}")
        self.teacher_masks = np.asarray(teacher_masks[: len(self.rows)], dtype=np.float32)
        self.image_size = int(image_size)
        self.bow_dim = int(bow_dim)
        self.query_bow_dim = int(query_bow_dim)
        self.channels = tuple(channels)
        self.text_encoder_type = str(text_encoder_type)
        self.instruction_max_len = int(instruction_max_len)
        self.query_max_len = int(query_max_len)
        self.tokenizer = None
        if self.text_encoder_type == "sequence":
            if not tokenizer_path:
                raise RuntimeError("text_encoder_type=sequence requires tokenizer_path")
            self.tokenizer = load_local_tokenizer(tokenizer_path)

        step_max = {}
        for row in self.rows:
            ep = int(row["episode_idx"])
            step_max[ep] = max(step_max.get(ep, 0), int(row["step_idx"]))

        self.meta = []
        for row in self.rows:
            step_ratio = float(row["step_idx"]) / float(max(1, step_max[int(row["episode_idx"])]))
            stage = infer_stage(step_ratio)
            instruction = row.get("instruction", "")
            query_text = " ".join(row.get("query_words") or [])
            if self.tokenizer is not None:
                instruction_ids, instruction_mask = tokenize_to_arrays(
                    self.tokenizer,
                    instruction,
                    self.instruction_max_len,
                )
                query_ids, query_mask = tokenize_to_arrays(
                    self.tokenizer,
                    query_text,
                    self.query_max_len,
                )
            else:
                instruction_ids = np.zeros((self.instruction_max_len,), dtype=np.int64)
                instruction_mask = np.zeros((self.instruction_max_len,), dtype=np.float32)
                query_ids = np.zeros((self.query_max_len,), dtype=np.int64)
                query_mask = np.zeros((self.query_max_len,), dtype=np.float32)
            self.meta.append(
                {
                    "image_path": row["image_path"],
                    "bow": hashed_bow(instruction, self.bow_dim),
                    "query_bow": hashed_bow(query_text, self.query_bow_dim),
                    "instruction_meta": instruction_meta_vector(instruction, row.get("query_words")),
                    "ambiguity_vec": np.zeros((INSTANCE_AMBIGUITY_DIM,), dtype=np.float32),
                    "stage_one_hot": stage_to_one_hot(stage),
                    "step_ratio": np.array([step_ratio], dtype=np.float32),
                    "instruction_ids": instruction_ids,
                    "instruction_mask": instruction_mask,
                    "query_ids": query_ids,
                    "query_mask": query_mask,
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.meta[idx]
        return {
            "image": load_image_tensor(item["image_path"], self.image_size),
            "bow": torch.tensor(item["bow"].tolist(), dtype=torch.float32),
            "query_bow": torch.tensor(item["query_bow"].tolist(), dtype=torch.float32),
            "instruction_meta": torch.tensor(item["instruction_meta"].tolist(), dtype=torch.float32),
            "ambiguity_vec": torch.tensor(item["ambiguity_vec"].tolist(), dtype=torch.float32),
            "stage_one_hot": torch.tensor(item["stage_one_hot"].tolist(), dtype=torch.float32),
            "step_ratio": torch.tensor(item["step_ratio"].tolist(), dtype=torch.float32),
            "instruction_ids": torch.tensor(item["instruction_ids"].tolist(), dtype=torch.long),
            "instruction_mask": torch.tensor(item["instruction_mask"].tolist(), dtype=torch.float32),
            "query_ids": torch.tensor(item["query_ids"].tolist(), dtype=torch.long),
            "query_mask": torch.tensor(item["query_mask"].tolist(), dtype=torch.float32),
            "target_mask": torch.tensor(self.teacher_masks[idx].tolist(), dtype=torch.float32),
        }


def build_loaders(dataset: Dataset, batch_size: int, val_ratio: float, seed: int) -> tuple[DataLoader, DataLoader]:
    train_idx, val_idx = split_indices(len(dataset), val_ratio=val_ratio, seed=seed)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def resolve_gate_type(cfg: dict) -> str:
    model_cfg = cfg["model"]
    gate_type = str(model_cfg.get("gate_type", "stage_conditioned"))
    if gate_type == "stage_conditioned":
        if bool(cfg.get("route_mixture", {}).get("enabled", False)):
            return "route_mixture"
        if bool(cfg.get("latent_phase", {}).get("enabled", False)):
            return "latent_phase"
    return gate_type


def build_policy(cfg: dict, channels: tuple[str, ...], budget_values: tuple[int, ...], device: torch.device) -> LearnedEvidencePolicy:
    model_cfg = cfg["model"]
    text_cfg = dict(model_cfg.get("text_encoder", {}))
    tokenizer_vocab = 0
    if str(text_cfg.get("type", "bow")) == "sequence":
        tokenizer = load_local_tokenizer(str(text_cfg["tokenizer_path"]))
        tokenizer_vocab = int(len(tokenizer))
    return LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        action_dim=7,
        channels=channels,
        budget_values=budget_values,
        text_encoder_type=str(text_cfg.get("type", "bow")),
        text_vocab_size=tokenizer_vocab,
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type=resolve_gate_type(cfg),
        latent_phase_slots=int(model_cfg.get("latent_phase_slots", 8)),
        route_bank=tuple(tuple(r) for r in cfg.get("route_mixture", {}).get("route_bank", [])),
    ).to(device)


def infer_route_probs(
    policy: LearnedEvidencePolicy,
    batch: dict[str, torch.Tensor],
    cfg: dict,
    channels: tuple[str, ...],
    budget_values: tuple[int, ...],
    route_bank_masks: torch.Tensor | None,
    stage_conditioning_enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_cfg = cfg["gating"]
    latent_phase_cfg = cfg.get("latent_phase", {})
    ctx = policy.encode_context(batch)
    stage_one_hot = batch["stage_one_hot"] if stage_conditioning_enabled else torch.zeros_like(batch["stage_one_hot"])
    gate_out = policy.forward_gate(
        ctx,
        batch,
        stage_one_hot,
        phase_temperature=float(latent_phase_cfg.get("temperature_end", gate_cfg["temperature_end"])),
        hard_phase=bool(latent_phase_cfg.get("hard_assignment", False)),
        training=True,
    )
    if route_bank_masks is not None:
        hard_gates, gate_probs, _, _, _, _ = sample_route_mixture_gates(
            gate_out["route_logits"],
            route_bank_masks,
            budget_values=budget_values,
            temperature=float(gate_cfg["temperature_end"]),
            training=True,
        )
        return hard_gates, gate_probs
    hard_gates, channel_probs, _, budget_probs = sample_budget_topk_gates(
        gate_out["channel_logits"],
        gate_out["budget_logits"],
        budget_values=budget_values,
        temperature=float(gate_cfg["temperature_end"]),
        training=True,
    )
    budget_tensor = torch.tensor(budget_values, dtype=budget_probs.dtype, device=budget_probs.device).unsqueeze(0)
    soft_budget = torch.sum(budget_probs * budget_tensor, dim=1)
    denom = torch.clamp(channel_probs.sum(dim=1, keepdim=True), min=1e-6)
    soft_gates = torch.clamp(channel_probs * (soft_budget.unsqueeze(1) / denom), 0.0, 1.0)
    return hard_gates, soft_gates


def evaluate(
    policy: LearnedEvidencePolicy,
    loader: DataLoader,
    cfg: dict,
    channels: tuple[str, ...],
    budget_values: tuple[int, ...],
    route_bank_masks: torch.Tensor | None,
    stage_conditioning_enabled: bool,
    device: torch.device,
) -> dict:
    policy.eval()
    total_loss = 0.0
    total_jaccard = 0.0
    total_exact = 0.0
    total_selected = torch.zeros((len(channels),), dtype=torch.float64)
    n = 0
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            hard_gates, soft_gates = infer_route_probs(
                policy,
                batch,
                cfg,
                channels,
                budget_values,
                route_bank_masks,
                stage_conditioning_enabled,
            )
            target = batch["target_mask"]
            loss = F.mse_loss(soft_gates, target)
            pred_bin = hard_gates > 0.5
            target_bin = target > 0.5
            inter = (pred_bin & target_bin).sum(dim=1).to(torch.float32)
            union = (pred_bin | target_bin).sum(dim=1).clamp(min=1).to(torch.float32)
            total_loss += float(loss.item()) * target.shape[0]
            total_jaccard += float((inter / union).sum().item())
            total_exact += float(torch.all(pred_bin == target_bin, dim=1).to(torch.float32).sum().item())
            total_selected += hard_gates.sum(dim=0).detach().cpu().to(torch.float64)
            n += target.shape[0]
    out = {
        "val_loss": total_loss / max(1, n),
        "route_jaccard": total_jaccard / max(1, n),
        "route_exact": total_exact / max(1, n),
        "avg_selected_channels": float(total_selected.sum().item() / max(1, n)),
    }
    for i, ch in enumerate(channels):
        out[f"{ch}_keep_rate"] = float(total_selected[i].item() / max(1, n))
    return out


def save_checkpoint(
    out_dir: Path,
    policy: LearnedEvidencePolicy,
    cfg: dict,
    channels: tuple[str, ...],
    budget_values: tuple[int, ...],
    teacher_dynamic_checkpoint_dir: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.context.state_dict(), out_dir / "context.pt")
    torch.save(policy.gate.state_dict(), out_dir / "gate.pt")
    if getattr(policy, "instruction_text_encoder", None) is not None:
        torch.save(policy.instruction_text_encoder.state_dict(), out_dir / "instruction_text_encoder.pt")
    if getattr(policy, "query_text_encoder", None) is not None:
        torch.save(policy.query_text_encoder.state_dict(), out_dir / "query_text_encoder.pt")
    resolved = {
        "config": cfg,
        "channels": list(channels),
        "budget_values": list(budget_values),
        "teacher_dynamic_checkpoint_dir": teacher_dynamic_checkpoint_dir,
        "student_type": "online_lazy_gate",
        "routing_inputs": ["image", "instruction", "query_words", "step_ratio", "instruction_meta"],
    }
    (out_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--teacher_dynamic_checkpoint_dir", required=True)
    parser.add_argument("--gate_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)

    cfg = load_yaml(args.gate_config)
    channels = resolve_channels(cfg)
    budget_values = resolve_budget_values(cfg, channels)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    stage_conditioning_enabled = bool(cfg.get("stage_conditioning", {}).get("enabled", True))
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))

    teacher_dir = Path(args.teacher_dynamic_checkpoint_dir)
    teacher_resolved = json.loads((teacher_dir / "resolved_config.json").read_text(encoding="utf-8"))
    teacher_channels = tuple(teacher_resolved["channels"])
    if tuple(channels) != teacher_channels:
        raise RuntimeError(f"channel mismatch: gate cfg {channels} vs teacher ckpt {teacher_channels}")
    teacher_masks = np.load(teacher_dir / "channel_masks.npy")

    dataset = OnlineLazyGateDistillDataset(
        manifest_path=args.feature_manifest,
        teacher_masks=teacher_masks,
        image_size=int(model_cfg["image_size"]),
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        channels=channels,
        text_encoder_type=str(model_cfg.get("text_encoder", {}).get("type", "bow")),
        tokenizer_path=model_cfg.get("text_encoder", {}).get("tokenizer_path"),
        instruction_max_len=int(model_cfg.get("text_encoder", {}).get("instruction_max_len", 24)),
        query_max_len=int(model_cfg.get("text_encoder", {}).get("query_max_len", 12)),
        limit=int(args.limit),
    )
    train_loader, val_loader = build_loaders(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        val_ratio=float(train_cfg["val_ratio"]),
        seed=int(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = build_policy(cfg, channels, budget_values, device=device)
    route_bank_masks = None
    if route_mixture_enabled:
        route_bank = tuple(tuple(route) for route in route_mixture_cfg.get("route_bank", []))
        route_bank_masks = route_bank_tensor(route_bank, channels, device)

    train_params = list(policy.context.parameters()) + list(policy.gate.parameters())
    if getattr(policy, "instruction_text_encoder", None) is not None:
        train_params += list(policy.instruction_text_encoder.parameters())
    if getattr(policy, "query_text_encoder", None) is not None:
        train_params += list(policy.query_text_encoder.parameters())
    optimizer = torch.optim.AdamW(
        train_params,
        lr=float(train_cfg["lr_gate"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "train_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    epochs = int(train_cfg.get("joint_epochs", 6))
    best_val = float("inf")
    global_step = 0
    for epoch in range(1, epochs + 1):
        policy.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            global_step += 1
            batch = move_batch_to_device(batch, device)
            _, soft_gates = infer_route_probs(
                policy,
                batch,
                cfg,
                channels,
                budget_values,
                route_bank_masks,
                stage_conditioning_enabled,
            )
            target = batch["target_mask"]
            mask_loss = F.mse_loss(soft_gates, target)
            budget_loss = F.mse_loss(
                soft_gates.sum(dim=1),
                target.sum(dim=1),
            )
            loss = mask_loss + 0.2 * budget_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            running += float(loss.item()) * target.shape[0]
            seen += target.shape[0]
            if global_step % max(1, int(args.log_every)) == 0:
                print(
                    f"[progress] epoch={epoch}/{epochs} step={global_step} loss={running / max(1, seen):.6f}",
                    flush=True,
                )

        val_metrics = evaluate(
            policy,
            val_loader,
            cfg,
            channels,
            budget_values,
            route_bank_masks,
            stage_conditioning_enabled,
            device,
        )
        train_loss = running / max(1, seen)
        row = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[epoch] {json.dumps(row)}", flush=True)

        if val_metrics["val_loss"] <= best_val:
            best_val = val_metrics["val_loss"]
            save_checkpoint(
                out_dir=out_dir,
                policy=policy,
                cfg=cfg,
                channels=channels,
                budget_values=budget_values,
                teacher_dynamic_checkpoint_dir=str(teacher_dir),
            )

    print(f"[ok] saved online lazy gate checkpoint to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
