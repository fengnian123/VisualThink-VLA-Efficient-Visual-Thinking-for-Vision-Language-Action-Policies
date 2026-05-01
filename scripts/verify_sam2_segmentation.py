#!/usr/bin/env python3
"""Run SAM2 real segmentation on a single image + bbox and save mask."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(args.image_path).convert("RGB"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = SAM2ImagePredictor(build_sam2(args.model_cfg, args.checkpoint, device=device))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        predictor.set_image(img)
        box = np.array(args.bbox, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)

    mask = masks[0].astype(np.uint8)
    score = float(scores[0])
    mask_png = (mask * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "sam2_mask.png"), mask_png)

    # Overlay visualization
    vis = img.copy()
    vis[mask > 0] = (0.6 * vis[mask > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    x1, y1, x2, y2 = [int(v) for v in args.bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite(str(out_dir / "sam2_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    rect_area = max(0, x2 - x1) * max(0, y2 - y1)
    mask_area = int(mask.sum())
    print(f"[ok] score={score:.6f}")
    print(f"[ok] mask_area={mask_area}, bbox_area={rect_area}, ratio={mask_area / max(1, rect_area):.4f}")
    print(f"[ok] saved={out_dir}")


if __name__ == "__main__":
    main()
