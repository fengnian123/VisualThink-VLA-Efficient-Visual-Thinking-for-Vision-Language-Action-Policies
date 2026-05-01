#!/usr/bin/env python3
"""Controlled channel-mask ablations for DynamicSoft evidence inference.

The benchmark fixes the trained DynamicSoft adapter and evaluates several mask
variants on the same sampled rows. This is the causal/control-variable ablation
for the paper's feature-ablation table; it is distinct from post-hoc
route-conditioned filtering.
"""

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

from models.evidence_gating import (  # noqa: E402
    FEATURE_DIMS,
    bbox_vector,
    edge_vector,
    infer_stage_with_proxy,
    motion_vector,
    relation_vector,
    stage_to_one_hot,
)
from models.openvla_soft_evidence import (  # noqa: E402
    SoftEvidenceAdapter,
    SoftEvidenceBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)


CHANNELS = ("bbox", "edge", "motion", "relation")
DEFAULT_VARIANTS = (
    "dynamic_full",
    "without_bbox",
    "without_edge",
    "without_motion",
    "without_relation",
    "only_bbox_relation",
    "only_edge_motion",
)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def load_norm_stats(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_indices(n: int, sample_count: int, seed: int, strategy: str) -> list[int]:
    if sample_count <= 0 or sample_count >= n:
        return list(range(n))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(n), sample_count))
    if strategy == "stride":
        return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, num=sample_count)))
    raise ValueError(f"unknown sample strategy: {strategy}")


def load_adapter(checkpoint_dir: Path, channels: tuple[str, ...], hidden_size: int, device: torch.device) -> SoftEvidenceAdapter:
    resolved = json.loads((checkpoint_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]["adapter"]
    adapter = SoftEvidenceAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(cfg["num_global_tokens"]),
        proj_dim=int(cfg["proj_dim"]),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(checkpoint_dir / "adapter.pt", map_location=device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing adapter keys: {missing}", flush=True)
    if unexpected:
        print(f"[warn] unexpected adapter keys: {unexpected}", flush=True)
    adapter.eval()
    return adapter


def action_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def apply_variant(base_mask: np.ndarray, channels: tuple[str, ...], variant: str) -> np.ndarray:
    mask = np.asarray(base_mask, dtype=np.float32).copy()
    if variant == "dynamic_full":
        return mask
    if variant.startswith("without_"):
        channel = variant.removeprefix("without_")
        if channel not in channels:
            raise ValueError(f"unknown channel in variant {variant}")
        mask[channels.index(channel)] = 0.0
        return mask
    if variant == "only_bbox_relation":
        keep = {"bbox", "relation"}
    elif variant == "only_edge_motion":
        keep = {"edge", "motion"}
    else:
        raise ValueError(f"unknown variant: {variant}")
    return np.asarray([1.0 if ch in keep else 0.0 for ch in channels], dtype=np.float32)


def feature_batch_from_row(row: dict, step_max: dict[int, int], channels: tuple[str, ...], mask: np.ndarray) -> SoftEvidenceBatch:
    episode_idx = int(row["episode_idx"])
    step_idx = int(row["step_idx"])
    step_ratio = float(step_idx) / float(max(1, step_max[episode_idx]))
    with np.load(row["npz_path"], allow_pickle=True) as npz:
        bbox = bbox_vector(npz)
        edge = edge_vector(npz)
        motion = motion_vector(npz)
        relation = relation_vector(npz, row=row)
    stage = infer_stage_with_proxy(
        step_ratio,
        row=row,
        bbox_vec=bbox,
        edge_vec=edge,
        motion_vec=motion,
        relation_vec=relation,
        phase_proxy_cfg={},
    )
    vectors = {
        "bbox": bbox,
        "edge": edge,
        "motion": motion,
        "relation": relation,
    }
    return SoftEvidenceBatch(
        channel_features={ch: torch.tensor(vectors[ch], dtype=torch.float32).unsqueeze(0) for ch in channels},
        channel_mask=torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
        stage_one_hot=torch.tensor(stage_to_one_hot(stage), dtype=torch.float32).unsqueeze(0),
        step_ratio=torch.tensor([step_ratio], dtype=torch.float32).unsqueeze(0),
    )


def summarize(rows: list[dict], channels: tuple[str, ...]) -> list[dict]:
    by_variant: dict[str, list[dict]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    out = []
    for variant, items in by_variant.items():
        success = np.asarray([1.0 if r["success"] else 0.0 for r in items], dtype=np.float32)
        l1 = np.asarray([float(r["l1"]) for r in items], dtype=np.float32)
        latency = np.asarray([float(r["inference_time_s"]) for r in items], dtype=np.float32)
        selected = np.asarray([float(r["selected_channels"]) for r in items], dtype=np.float32)
        record = {
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
            record[f"{channel}_keep_rate"] = float(
                np.mean([1.0 if r["gates"].get(channel, False) else 0.0 for r in items])
            )
        out.append(record)
    return sorted(out, key=lambda r: DEFAULT_VARIANTS.index(r["variant"]) if r["variant"] in DEFAULT_VARIANTS else 999)


def write_summary(output_dir: Path, summary_rows: list[dict], channels: tuple[str, ...]) -> None:
    (output_dir / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "| Variant | N | Success | Avg L1 | Median L1 | Latency(s) | Avg selected | "
        + " | ".join(ch.title() for ch in channels)
        + " |",
        "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(channels),
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['success_rate']:.4f} | {row['avg_l1']:.4f} | "
            f"{row['median_l1']:.4f} | {row['avg_step_latency_s']:.4f} | {row['avg_selected_channels']:.4f} | "
            + " | ".join(f"{row[f'{ch}_keep_rate']:.4f}" for ch in channels)
            + " |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature_manifest", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dynamic_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sample_count", type=int, default=4000)
    parser.add_argument("--sample_strategy", choices=("stride", "random"), default="stride")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--norm_stats", default="")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--mask_paths_json", default="", help="Optional JSON mapping variant name to .npy mask array.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    manifest_path = Path(args.feature_manifest)
    ckpt_dir = Path(args.dynamic_checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    channels = tuple(resolved["channels"])
    if channels != CHANNELS:
        print(f"[warn] channel order differs from default: {channels}", flush=True)
    masks = np.load(ckpt_dir / "channel_masks.npy")
    external_masks: dict[str, np.ndarray] = {}
    if args.mask_paths_json:
        mask_spec = json.loads(Path(args.mask_paths_json).read_text(encoding="utf-8"))
        for variant, raw_path in mask_spec.items():
            external_masks[str(variant)] = np.load(raw_path).astype(np.float32)

    print(f"[stage] load manifest {manifest_path}", flush=True)
    rows = load_rows(manifest_path)
    if len(masks) < len(rows):
        raise RuntimeError(f"mask rows ({len(masks)}) fewer than manifest rows ({len(rows)})")
    step_max: dict[int, int] = {}
    for row in rows:
        ep = int(row["episode_idx"])
        step_max[ep] = max(step_max.get(ep, 0), int(row["step_idx"]))
    indices = choose_indices(len(rows), args.sample_count, args.seed, args.sample_strategy)
    if external_masks and args.variants == ",".join(DEFAULT_VARIANTS):
        variants = tuple(external_masks.keys())
    else:
        variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for variant in variants:
        if variant in external_masks and len(external_masks[variant]) < len(rows):
            raise RuntimeError(
                f"external mask rows for {variant} ({len(external_masks[variant])}) fewer than manifest rows ({len(rows)})"
            )
    print(
        f"[info] dataset={args.dataset} total_rows={len(rows)} sampled={len(indices)} variants={len(variants)}",
        flush=True,
    )

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
    adapter = load_adapter(ckpt_dir, channels, int(vla.get_input_embeddings().weight.shape[1]), device)

    result_rows: list[dict] = []
    start_all = time.perf_counter()
    for sample_i, idx in enumerate(indices, start=1):
        row = rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        gt = np.asarray(row["action"], dtype=np.float32)
        prompt = make_openvla_prompt(row["instruction"], args.model_path)
        base_mask = np.asarray(masks[idx], dtype=np.float32)

        for variant in variants:
            if variant in external_masks:
                variant_mask = np.asarray(external_masks[variant][idx], dtype=np.float32)
            else:
                variant_mask = apply_variant(base_mask, channels, variant)
            batch = feature_batch_from_row(row, step_max, channels, variant_mask)
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
                    "variant": variant,
                    "source_idx": int(idx),
                    "episode_idx": int(row["episode_idx"]),
                    "step_idx": int(row["step_idx"]),
                    "success": bool(l1 <= args.success_l1_thresh),
                    "l1": l1,
                    "inference_time_s": elapsed,
                    "selected_channels": float(np.sum(variant_mask)),
                    "gates": {ch: bool(variant_mask[i] > 0.5) for i, ch in enumerate(channels)},
                    "mask_values": {ch: float(variant_mask[i]) for i, ch in enumerate(channels)},
                }
            )
        if sample_i == 1 or sample_i % 100 == 0 or sample_i == len(indices):
            elapsed_all = time.perf_counter() - start_all
            done_preds = sample_i * len(variants)
            total_preds = len(indices) * len(variants)
            speed = done_preds / max(1e-6, elapsed_all)
            eta = (total_preds - done_preds) / max(1e-6, speed)
            print(
                f"[progress] {sample_i}/{len(indices)} samples {done_preds}/{total_preds} preds "
                f"speed={speed:.2f} pred/s eta={eta/3600:.2f}h",
                flush=True,
            )

    (output_dir / "eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result_rows),
        encoding="utf-8",
    )
    summary_rows = summarize(result_rows, channels)
    write_summary(output_dir, summary_rows, channels)
    run_meta = {
        "dataset": args.dataset,
        "feature_manifest": str(manifest_path),
        "dynamic_checkpoint_dir": str(ckpt_dir),
        "sample_count": len(indices),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "variants": variants,
        "unnorm_key": args.unnorm_key,
        "success_l1_thresh": args.success_l1_thresh,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
