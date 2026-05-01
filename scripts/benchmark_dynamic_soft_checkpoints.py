#!/usr/bin/env python3
"""Benchmark multiple DynamicSoft checkpoints on the same fixed manifest."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import FEATURE_DIMS, LearnedGatingDataset  # noqa: E402
from models.openvla_soft_evidence import (  # noqa: E402
    SoftEvidenceAdapter,
    SoftEvidenceBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)
from scripts.train_openvla_soft_evidence import compute_dynamic_masks  # noqa: E402


def load_rows(manifest_path: str) -> list[dict]:
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {manifest_path}")
    return rows


def action_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def load_norm_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_adapter(checkpoint_dir: Path, channels: tuple[str, ...], hidden_size: int, device: torch.device) -> SoftEvidenceAdapter:
    resolved = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    adapter_cfg = resolved["config"]["adapter"]
    adapter = SoftEvidenceAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(adapter_cfg["num_global_tokens"]),
        proj_dim=int(adapter_cfg["proj_dim"]),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(checkpoint_dir / "adapter.pt", map_location=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing adapter keys when loading {checkpoint_dir}: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected adapter keys when loading {checkpoint_dir}: {unexpected}", flush=True)
    adapter.eval()
    return adapter


def masks_for_manifest(checkpoint_dir: Path, channels: tuple[str, ...], feature_manifest: str, device: torch.device) -> np.ndarray:
    resolved = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved.get("config", {})
    dynamic_mask_cfg = cfg.get("dynamic_mask", {})
    gate_checkpoint_dir = str(resolved.get("gate_checkpoint_dir") or "")
    gate_config = str(resolved.get("gate_config") or resolved.get("config_path") or "")
    if not gate_checkpoint_dir:
        raise RuntimeError(f"missing gate_checkpoint_dir in {checkpoint_dir / 'resolved_config.json'}")
    return compute_dynamic_masks(
        manifest_path=feature_manifest,
        channels=channels,
        gate_checkpoint_dir=gate_checkpoint_dir,
        gate_config=gate_config,
        device=device,
        limit=0,
        batch_size=64,
        mask_mode=str(dynamic_mask_cfg.get("mode", "hard")).lower(),
        soft_mask_blend=float(dynamic_mask_cfg.get("soft_mask_blend", 0.0)),
        min_mask_floor=float(dynamic_mask_cfg.get("min_mask_floor", 0.0)),
    ).astype(np.float32)


def summarize(rows: list[dict], channels: tuple[str, ...]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, items in grouped.items():
        success = np.asarray([1.0 if item["success"] else 0.0 for item in items], dtype=np.float32)
        l1 = np.asarray([float(item["l1"]) for item in items], dtype=np.float32)
        latency = np.asarray([float(item["inference_time_s"]) for item in items], dtype=np.float32)
        selected = np.asarray([float(item["selected_channels"]) for item in items], dtype=np.float32)
        rec = {
            "variant": variant,
            "n": len(items),
            "success_rate": float(success.mean()),
            "avg_l1": float(l1.mean()),
            "median_l1": float(np.median(l1)),
            "avg_step_latency_s": float(latency.mean()),
            "inference_fps": float(1.0 / max(1e-6, latency.mean())),
            "avg_selected_channels": float(selected.mean()),
        }
        for channel in channels:
            rec[f"{channel}_keep_rate"] = float(
                np.mean([1.0 if item["gates"].get(channel, False) else 0.0 for item in items])
            )
        out.append(rec)
    return out


def write_summary(path: Path, rows: list[dict], channels: tuple[str, ...], order: tuple[str, ...]) -> None:
    sort_key = {name: idx for idx, name in enumerate(order)}
    rows = sorted(rows, key=lambda item: sort_key.get(item["variant"], 999))
    (path / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md = [
        "| Variant | N | Success | Avg L1 | Median L1 | Latency(s) | Avg selected | "
        + " | ".join(ch.title() for ch in channels)
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    for row in rows:
        md.append(
            f"| {row['variant']} | {row['n']} | {row['success_rate']:.4f} | {row['avg_l1']:.4f} | "
            f"{row['median_l1']:.4f} | {row['avg_step_latency_s']:.4f} | {row['avg_selected_channels']:.4f} | "
            + " | ".join(f"{row[f'{ch}_keep_rate']:.4f}" for ch in channels)
            + " |"
        )
    (path / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_variant(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"expected name=checkpoint_dir, got: {raw}")
    name, checkpoint_dir = raw.split("=", 1)
    name = name.strip()
    checkpoint_dir = checkpoint_dir.strip()
    if not name or not checkpoint_dir:
        raise ValueError(f"invalid variant spec: {raw}")
    return name, checkpoint_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--norm_stats", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--variant", action="append", default=[])
    args = parser.parse_args()

    if not args.variant:
        raise RuntimeError("at least one --variant name=checkpoint_dir is required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.feature_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = [parse_variant(item) for item in args.variant]
    order = tuple(name for name, _ in variant_specs)
    first_resolved = json.loads((Path(variant_specs[0][1]) / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(first_resolved["channels"])

    dataset = LearnedGatingDataset(
        args.feature_manifest,
        image_size=64,
        bow_dim=256,
        limit=0,
        channels=channels,
    )
    if len(rows) != len(dataset):
        raise RuntimeError(f"row/dataset mismatch: {len(rows)} vs {len(dataset)}")

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
    if args.norm_stats:
        extra_norm_stats = load_norm_stats(args.norm_stats)
        base_norm_stats = dict(getattr(vla, "norm_stats", {}) or {})
        base_norm_stats.update(extra_norm_stats)
        vla.norm_stats = base_norm_stats
    vla.eval()

    hidden_size = int(vla.get_input_embeddings().weight.shape[1])
    variant_models: dict[str, tuple[SoftEvidenceAdapter, np.ndarray]] = {}
    for name, ckpt_str in variant_specs:
        ckpt_dir = Path(ckpt_str)
        adapter = load_adapter(ckpt_dir, channels, hidden_size, device)
        masks = masks_for_manifest(ckpt_dir, channels, args.feature_manifest, device)
        if len(masks) != len(rows):
            raise RuntimeError(f"mask rows ({len(masks)}) != manifest rows ({len(rows)}) for {name}")
        variant_models[name] = (adapter, masks)

    result_rows: list[dict] = []
    start_all = time.perf_counter()
    total_preds = len(rows) * len(variant_models)
    for sample_i, row in enumerate(rows, start=1):
        image = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(row["action"], dtype=np.float32)
        for name, (adapter, masks) in variant_models.items():
            mask = np.asarray(masks[sample_i - 1], dtype=np.float32)
            batch = SoftEvidenceBatch(
                channel_features={ch: dataset[sample_i - 1][ch].unsqueeze(0).to(device=device, dtype=torch.float32) for ch in channels},
                channel_mask=torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0),
                stage_one_hot=dataset[sample_i - 1]["stage_one_hot"].unsqueeze(0).to(device=device, dtype=torch.float32),
                step_ratio=dataset[sample_i - 1]["step_ratio"].unsqueeze(0).to(device=device, dtype=torch.float32),
            )
            start = time.perf_counter()
            action = predict_action_with_soft_evidence(
                vla=vla,
                processor=processor,
                tokenizer=tokenizer,
                adapter=adapter,
                image=image,
                instruction=row["instruction"],
                evidence_batch=batch,
                model_path=args.model_path,
                device=device,
                dtype=dtype,
                unnorm_key=args.unnorm_key,
            )
            elapsed = time.perf_counter() - start
            l1 = action_l1(np.asarray(action, dtype=np.float32), gt)
            result_rows.append(
                {
                    "dataset": args.dataset,
                    "variant": name,
                    "source_idx": sample_i - 1,
                    "episode_idx": int(row["episode_idx"]),
                    "step_idx": int(row["step_idx"]),
                    "success": bool(l1 <= args.success_l1_thresh),
                    "l1": l1,
                    "inference_time_s": elapsed,
                    "selected_channels": float(np.sum(mask)),
                    "gates": {ch: bool(mask[i] > 0.5) for i, ch in enumerate(channels)},
                }
            )
        if sample_i == 1 or sample_i % 100 == 0 or sample_i == len(rows):
            elapsed_all = time.perf_counter() - start_all
            done_preds = sample_i * len(variant_models)
            speed = done_preds / max(1e-6, elapsed_all)
            eta = (total_preds - done_preds) / max(1e-6, speed)
            print(
                f"[progress] {sample_i}/{len(rows)} samples {done_preds}/{total_preds} preds "
                f"speed={speed:.2f} pred/s eta={eta/3600:.2f}h",
                flush=True,
            )

    (output_dir / "eval.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in result_rows),
        encoding="utf-8",
    )
    summary_rows = summarize(result_rows, channels)
    write_summary(output_dir, summary_rows, channels, order=order)
    meta = {
        "dataset": args.dataset,
        "feature_manifest": args.feature_manifest,
        "variant_order": list(order),
        "seed": args.seed,
        "success_l1_thresh": args.success_l1_thresh,
    }
    (output_dir / "run_config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
