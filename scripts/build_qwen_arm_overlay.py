#!/usr/bin/env python3
"""Extract a robot arm with Qwen-Image-Edit, then overlay a grayscale ghost on the next frame."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image


BASE_URL = "https://api-inference.modelscope.cn/"


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def run_modelscope_edit(
    *,
    api_key: str,
    image_path: Path,
    prompt: str,
    model: str,
    poll_interval: float,
    timeout: float,
) -> Image.Image:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": prompt,
        "image_url": [image_to_data_url(image_path)],
    }
    response = requests.post(
        f"{BASE_URL}v1/images/generations",
        headers={**headers, "X-ModelScope-Async-Mode": "true"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=180,
    )
    response.raise_for_status()
    task_id = response.json()["task_id"]
    deadline = time.time() + timeout
    while True:
        result = requests.get(
            f"{BASE_URL}v1/tasks/{task_id}",
            headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
            timeout=60,
        )
        result.raise_for_status()
        data = result.json()
        status = data.get("task_status", "")
        print(f"[info] task_id={task_id} status={status}", flush=True)
        if status == "SUCCEED":
            output_url = data["output_images"][0]
            return Image.open(BytesIO(requests.get(output_url, timeout=120).content)).convert("RGB")
        if status == "FAILED":
            raise RuntimeError(f"generation failed: {json.dumps(data, ensure_ascii=False)}")
        if time.time() > deadline:
            raise TimeoutError(f"timed out waiting for task {task_id}")
        time.sleep(poll_interval)


def _postprocess_mask(mask: np.ndarray, blur: int) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    filtered = np.zeros_like(mask)
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        if area >= 300:
            filtered[labels == idx] = 255
    if blur > 0:
        k = max(1, int(blur))
        if k % 2 == 0:
            k += 1
        filtered = cv2.GaussianBlur(filtered, (k, k), 0)
    return filtered


def build_mask(extracted_rgb: np.ndarray, min_value: int, blur: int) -> np.ndarray:
    gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY)
    black_bg = _postprocess_mask((gray > min_value).astype(np.uint8) * 255, blur)
    white_bg = _postprocess_mask((gray < (255 - min_value)).astype(np.uint8) * 255, blur)
    black_area = int((black_bg > 0).sum())
    white_area = int((white_bg > 0).sum())
    candidates = []
    if black_area > 0:
        candidates.append((black_area, black_bg))
    if white_area > 0:
        candidates.append((white_area, white_bg))
    if not candidates:
        return black_bg
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def compose_overlay(
    current_rgb: np.ndarray,
    extracted_rgb: np.ndarray,
    mask_u8: np.ndarray,
    opacity: float,
    gray_scale: float,
) -> np.ndarray:
    gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = np.clip(gray * gray_scale, 0.0, 255.0)
    ghost = np.stack([gray, gray, gray], axis=-1)
    alpha = (mask_u8.astype(np.float32) / 255.0) * opacity
    base = current_rgb.astype(np.float32)
    comp = base * (1.0 - alpha[..., None]) + ghost * alpha[..., None]
    return np.clip(comp, 0.0, 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prev_image", required=True)
    parser.add_argument("--curr_image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extract_output", default=None)
    parser.add_argument("--mask_output", default=None)
    parser.add_argument("--gray_output", default=None)
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mask-threshold", type=int, default=18)
    parser.add_argument("--mask-blur", type=int, default=7)
    parser.add_argument("--opacity", type=float, default=0.42)
    parser.add_argument("--gray-scale", type=float, default=0.75)
    parser.add_argument("--extract-prompt", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("MODELSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MODELSCOPE_API_KEY is required")

    prev_path = Path(args.prev_image).expanduser().resolve()
    curr_path = Path(args.curr_image).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()
    extract_path = Path(args.extract_output).expanduser().resolve() if args.extract_output else out_path.with_name(out_path.stem + "_extract.png")
    mask_path = Path(args.mask_output).expanduser().resolve() if args.mask_output else out_path.with_name(out_path.stem + "_mask.png")
    gray_path = Path(args.gray_output).expanduser().resolve() if args.gray_output else out_path.with_name(out_path.stem + "_gray.png")

    prompt = args.extract_prompt or (
        "Keep only the robot arm and gripper visible in this image. "
        "Replace all other regions, including table, bowl, food, cloth, stove, wall, and background, with pure white. "
        "Preserve the exact geometry, pose, and realistic photo appearance and brightness of the robot arm. "
        "Do not add any new robot, object, or background."
    )
    extracted = run_modelscope_edit(
        api_key=api_key,
        image_path=prev_path,
        prompt=prompt,
        model=args.model,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    extract_path.parent.mkdir(parents=True, exist_ok=True)
    extracted.save(extract_path)

    extracted_rgb = np.array(extracted)
    curr_rgb = np.array(Image.open(curr_path).convert("RGB"))
    mask_u8 = build_mask(extracted_rgb, min_value=args.mask_threshold, blur=args.mask_blur)
    gray = cv2.cvtColor(extracted_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = np.clip(gray * args.gray_scale, 0.0, 255.0).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    gray_rgb[mask_u8 == 0] = 0
    overlay = compose_overlay(
        current_rgb=curr_rgb,
        extracted_rgb=extracted_rgb,
        mask_u8=mask_u8,
        opacity=args.opacity,
        gray_scale=args.gray_scale,
    )

    Image.fromarray(mask_u8).save(mask_path)
    Image.fromarray(gray_rgb).save(gray_path)
    Image.fromarray(overlay).save(out_path)
    print(f"[ok] extract={extract_path}")
    print(f"[ok] mask={mask_path}")
    print(f"[ok] gray={gray_path}")
    print(f"[ok] overlay={out_path}")


if __name__ == "__main__":
    main()
