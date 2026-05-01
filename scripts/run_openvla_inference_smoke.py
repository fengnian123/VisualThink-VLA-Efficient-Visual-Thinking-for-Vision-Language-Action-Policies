#!/usr/bin/env python3
"""OpenVLA inference smoke test without flash-attn."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def make_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        sys_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{sys_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def load_image(image_path: str | None, image_size: int) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    arr = np.random.randint(0, 255, size=(image_size, image_size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="openvla/openvla-7b")
    parser.add_argument("--instruction", default="pick up the cup")
    parser.add_argument("--unnorm_key", default="bridge_orig")
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"[info] device={device} dtype={dtype} model={args.model_path} attn_impl={args.attn_impl}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_impl,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    prompt = make_prompt(args.instruction, args.model_path)
    image = load_image(args.image_path, args.image_size)
    inputs = processor(prompt, image).to(device, dtype=dtype)

    start = time.time()
    with torch.inference_mode():
        action = model.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
    elapsed = time.time() - start

    print(f"[ok] action={action}")
    print(f"[ok] elapsed_sec={elapsed:.4f}")
    if args.image_path:
        print(f"[ok] image={Path(args.image_path).resolve()}")


if __name__ == "__main__":
    main()
