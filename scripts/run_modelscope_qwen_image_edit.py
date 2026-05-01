#!/usr/bin/env python3
"""Call ModelScope async image-generation API for Qwen image edit."""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


BASE_URL = "https://api-inference.modelscope.cn/"


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--image", action="append", required=True, dest="images", help="One or more input image paths")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("MODELSCOPE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("MODELSCOPE_API_KEY is required")

    image_paths = [Path(p).expanduser().resolve() for p in args.images]
    for path in image_paths:
        if not path.exists():
            raise SystemExit(f"image not found: {path}")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "image_url": [image_to_data_url(path) for path in image_paths],
    }

    response = requests.post(
        f"{BASE_URL}v1/images/generations",
        headers={**headers, "X-ModelScope-Async-Mode": "true"},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=180,
    )
    response.raise_for_status()
    task_id = response.json()["task_id"]
    deadline = time.time() + args.timeout

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
            image = Image.open(BytesIO(requests.get(output_url, timeout=120).content)).convert("RGB")
            out_path = Path(args.output).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
            print(f"[ok] saved={out_path}")
            return
        if status == "FAILED":
            raise SystemExit(f"generation failed: {json.dumps(data, ensure_ascii=False)}")
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for task {task_id}")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
