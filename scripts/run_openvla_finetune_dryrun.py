#!/usr/bin/env python3
"""OpenVLA LoRA fine-tuning dry-run on DummyDataset."""

import argparse
import os
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import DummyDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="openvla/openvla-7b")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", default="artifacts/checkpoints/finetune_dryrun")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for OpenVLA dry-run fine-tuning.")

    # Register OpenVLA custom classes for local loading compatibility.
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_config)
    model.train()

    prompt_builder = PurePromptBuilder if "v01" not in args.model_path else VicunaV15ChatPromptBuilder
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = DummyDataset(
        action_tokenizer=action_tokenizer,
        base_tokenizer=processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=prompt_builder,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    total = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
        input_ids = batch["input_ids"].to(device=device)
        attention_mask = batch["attention_mask"].to(device=device)
        labels = batch["labels"].to(device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
            loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += 1
        print(f"[step {total}] loss={loss.item():.6f}")
        if total >= args.steps:
            break

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"[ok] dry-run done. saved={out_dir}")


if __name__ == "__main__":
    main()
