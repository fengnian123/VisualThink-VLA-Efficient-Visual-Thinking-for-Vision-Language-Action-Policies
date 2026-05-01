#!/usr/bin/env python3
"""Benchmark teacher baseline vs student+fallback dynamic policy and export report artifacts."""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_dynamic_distill import DistillDataset, PolicyMLP


def summarize(rows: list[dict]) -> dict:
    success = np.array([1.0 if r["success"] else 0.0 for r in rows], dtype=np.float32)
    fps = np.array([float(r["fps"]) for r in rows], dtype=np.float32)
    disturbed = np.array([1.0 if r["disturbed"] else 0.0 for r in rows], dtype=np.float32)
    recovered = np.array([1.0 if r.get("recovered", False) else 0.0 for r in rows], dtype=np.float32)
    disturbed_count = max(1.0, disturbed.sum())
    return {
        "n": int(len(rows)),
        "success_rate": float(success.mean()) if len(rows) else 0.0,
        "inference_fps": float(fps.mean()) if len(rows) else 0.0,
        "robustness_recovery": float(recovered.sum() / disturbed_count),
    }


def l1(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - gt)).item())


def save_report(baseline_rows: list[dict], dynamic_rows: list[dict], out_dir: Path) -> None:
    b = summarize(baseline_rows)
    d = summarize(dynamic_rows)
    rows = [
        ("success_rate", b["success_rate"], d["success_rate"]),
        ("inference_fps", b["inference_fps"], d["inference_fps"]),
        ("robustness_recovery", b["robustness_recovery"], d["robustness_recovery"]),
    ]

    md = ["| Metric | Baseline | Dynamic | Delta |", "|---|---:|---:|---:|"]
    for k, bv, dv in rows:
        md.append(f"| {k} | {bv:.4f} | {dv:.4f} | {dv-bv:+.4f} |")
    (out_dir / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    labels = [r[0] for r in rows]
    bvals = [r[1] for r in rows]
    dvals = [r[2] for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - w / 2, bvals, w, label="Baseline")
    plt.bar(x + w / 2, dvals, w, label="Dynamic")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Value")
    plt.title("Distill Baseline vs Dynamic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_manifest", required=True)
    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--student_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--teacher_hidden", type=int, default=1024)
    parser.add_argument("--student_hidden", type=int, default=256)
    parser.add_argument("--success_l1_thresh", type=float, default=0.08)
    parser.add_argument("--disturb_ratio", type=float, default=0.4)
    parser.add_argument("--disturb_scale", type=float, default=0.25)
    parser.add_argument("--fallback_norm_thresh", type=float, default=1.2)
    parser.add_argument("--fallback_policy", choices=["norm", "disturbed_always"], default="norm")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = DistillDataset(args.dropout_manifest, limit=args.limit)
    in_dim = len(ds[0][0])

    teacher = PolicyMLP(in_dim, hidden=args.teacher_hidden).to(device)
    student = PolicyMLP(in_dim, hidden=args.student_hidden).to(device)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
    student.load_state_dict(torch.load(args.student_ckpt, map_location=device))
    teacher.eval()
    student.eval()

    baseline_rows = []
    dynamic_rows = []

    for i in range(len(ds)):
        full_x, part_x, gt = ds[i]
        disturbed = rng.random() < args.disturb_ratio
        full_x = full_x.to(device)
        part_x = part_x.to(device)
        gt = gt.to(device)

        if disturbed:
            noise = torch.from_numpy(rng.normal(0.0, args.disturb_scale, size=part_x.shape).astype(np.float32)).to(device)
            part_x_in = part_x + noise
        else:
            part_x_in = part_x

        # Baseline: teacher on full features
        t0 = time.time()
        with torch.inference_mode():
            b_pred = teacher(full_x.unsqueeze(0)).squeeze(0)
        b_t = time.time() - t0
        b_l1 = l1(b_pred, gt)
        baseline_rows.append(
            {
                "idx": i,
                "success": b_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, b_t),
                "disturbed": bool(disturbed),
                "recovered": False,
                "l1": b_l1,
            }
        )

        # Dynamic: student first, fallback to teacher on disturbed frames when student confidence proxy is poor.
        t1 = time.time()
        with torch.inference_mode():
            s_pred = student(part_x_in.unsqueeze(0)).squeeze(0)
        p_t = time.time() - t1
        confidence_proxy = float(torch.linalg.norm(s_pred).item())
        if args.fallback_policy == "disturbed_always":
            use_fallback = disturbed
        else:
            use_fallback = disturbed and confidence_proxy > args.fallback_norm_thresh

        if use_fallback:
            t2 = time.time()
            with torch.inference_mode():
                d_pred = teacher(full_x.unsqueeze(0)).squeeze(0)
            d_t = p_t + (time.time() - t2)
        else:
            d_pred = s_pred
            d_t = p_t

        s_l1 = l1(s_pred, gt)
        d_l1 = l1(d_pred, gt)
        recovered = disturbed and use_fallback and (s_l1 > args.success_l1_thresh) and (d_l1 <= args.success_l1_thresh)
        dynamic_rows.append(
            {
                "idx": i,
                "success": d_l1 <= args.success_l1_thresh,
                "fps": 1.0 / max(1e-6, d_t),
                "disturbed": bool(disturbed),
                "recovered": bool(recovered),
                "l1": d_l1,
                "fallback": bool(use_fallback),
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in baseline_rows),
        encoding="utf-8",
    )
    (out_dir / "dynamic_eval.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in dynamic_rows),
        encoding="utf-8",
    )
    save_report(baseline_rows, dynamic_rows, out_dir)

    print("[ok] baseline:", summarize(baseline_rows))
    print("[ok] dynamic:", summarize(dynamic_rows))
    print(f"[ok] report dir: {out_dir}")


if __name__ == "__main__":
    main()
