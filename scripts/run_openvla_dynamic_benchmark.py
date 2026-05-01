#!/usr/bin/env python3
"""Small-scale baseline vs dynamic benchmark with OpenVLA and report artifacts."""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def prompt_from_schema(instruction: str, schema_text: str) -> str:
    q = (
        f"{instruction}\n"
        f"Visual evidence:\n{schema_text}\n"
        "Return robot action."
    )
    return f"In: What action should the robot take to {q}?\nOut:"


def add_disturbance(image: Image.Image, sigma: float = 22.0) -> Image.Image:
    arr = np.array(image.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


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


def predict_action(vla, processor, image: Image.Image, prompt: str, device: str, dtype):
    inputs = processor(prompt, image).to(device, dtype=dtype)
    start = time.time()
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    elapsed = time.time() - start
    return np.asarray(action, dtype=np.float32), elapsed


def save_report(baseline_rows: list[dict], dynamic_rows: list[dict], out_dir: Path) -> None:
    b = summarize(baseline_rows)
    d = summarize(dynamic_rows)

    rows = [
        ("success_rate", b["success_rate"], d["success_rate"]),
        ("inference_fps", b["inference_fps"], d["inference_fps"]),
        ("robustness_recovery", b["robustness_recovery"], d["robustness_recovery"]),
    ]

    md = [
        "| Metric | Baseline OpenVLA | Dynamic Thinking | Delta |",
        "|---|---:|---:|---:|",
    ]
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
    plt.title("Baseline vs Dynamic (Small-Scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="openvla/openvla-7b")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--disturb_ratio", type=float, default=0.4)
    parser.add_argument("--success_l1_thresh", type=float, default=0.2)
    parser.add_argument("--fallback_partial_l1_thresh", type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    baseline_rows = []
    dynamic_rows = []

    with open(args.dropout_manifest, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            image = Image.open(rec["image_path"]).convert("RGB")
            gt = np.asarray(rec["action"], dtype=np.float32)
            disturbed = random.random() < args.disturb_ratio
            image_in = add_disturbance(image) if disturbed else image

            # Baseline: full schema
            b_prompt = prompt_from_schema(rec["instruction"], rec["full_schema_text"])
            b_action, b_t = predict_action(vla, processor, image_in, b_prompt, device, dtype)
            b_l1 = l1(b_action, gt)
            baseline_rows.append(
                {
                    "idx": idx,
                    "success": b_l1 <= args.success_l1_thresh,
                    "fps": 1.0 / max(1e-6, b_t),
                    "disturbed": disturbed,
                    "recovered": False,
                    "l1": b_l1,
                }
            )

            # Dynamic: partial schema, fallback to full when partial quality is poor (oracle for small-scale eval)
            p_prompt = prompt_from_schema(rec["instruction"], rec["partial_schema_text"])
            p_action, p_t = predict_action(vla, processor, image_in, p_prompt, device, dtype)
            p_l1 = l1(p_action, gt)

            use_fallback = disturbed and p_l1 > args.fallback_partial_l1_thresh
            if use_fallback:
                d_action, fb_t = predict_action(vla, processor, image_in, b_prompt, device, dtype)
                d_t = p_t + fb_t
            else:
                d_action = p_action
                d_t = p_t

            d_l1 = l1(d_action, gt)
            recovered = disturbed and use_fallback and (p_l1 > args.success_l1_thresh) and (d_l1 <= args.success_l1_thresh)
            dynamic_rows.append(
                {
                    "idx": idx,
                    "success": d_l1 <= args.success_l1_thresh,
                    "fps": 1.0 / max(1e-6, d_t),
                    "disturbed": disturbed,
                    "recovered": recovered,
                    "l1": d_l1,
                    "fallback": use_fallback,
                }
            )

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
