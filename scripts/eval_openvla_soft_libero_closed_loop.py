#!/usr/bin/env python3
"""Closed-loop LIBERO evaluation for OpenVLA / FullSoft / DynamicSoft.

This script is the paper-metrics path for our own methods. It reuses the
actual LIBERO simulator loop and reports:
  - success_rate
  - avg_completion_time_s (proxy: per-episode sum of step inference latency)
  - timeout_penalized_completion_time_s (same proxy over all episodes)
  - avg_step_latency_s

Unlike the existing offline action-prediction benchmarks, this script still
runs a real LIBERO closed loop. Raw wall-clock completion time is retained as
an auxiliary field, but the summary's completion-time columns follow the
current paper proxy definition so they are comparable to offline datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
import torch
import tqdm
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from torch import nn
from transformers import AutoModelForVision2Seq, AutoProcessor, pipeline as hf_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.evidence_gating import (
    FEATURE_DIMS,
    LearnedEvidencePolicy,
    hashed_bow,
    instruction_meta_vector,
    route_bank_tensor,
    sample_budget_topk_gates,
    sample_route_mixture_gates,
    stage_to_one_hot,
    tokenize_to_arrays,
)
from models.openvla_soft_evidence import (
    SoftEvidenceAdapter,
    SoftEvidenceBatch,
    make_openvla_prompt,
    predict_action_with_soft_evidence,
)
from utils.motion_features import compute_motion_map, motion_vector_from_map
from utils.relation_features import build_relation_stats, relation_vector_from_stats


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
STAGES = ("approach", "grasp", "place")
SUITE_TO_UNNORM_KEY = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
SUITE_TO_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "in",
    "on",
    "and",
    "or",
    "of",
    "for",
    "with",
    "is",
    "are",
    "be",
    "robot",
    "what",
    "should",
    "take",
    "action",
    "task",
    "move",
    "pick",
    "place",
    "put",
    "open",
    "close",
    "stack",
    "turn",
    "push",
}
OBJECT_HINTS = {
    "bowl",
    "plate",
    "cup",
    "mug",
    "can",
    "bottle",
    "box",
    "block",
    "spoon",
    "fork",
    "knife",
    "pan",
    "pot",
    "lid",
    "drawer",
    "microwave",
    "cabinet",
    "door",
    "stove",
    "burner",
    "cloth",
    "towel",
}


class TeeStream:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, data):
        self.primary.write(data)
        self.secondary.write(data)
        self.secondary.flush()
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


@dataclass
class ModelBundle:
    name: str
    adapter: SoftEvidenceAdapter | None
    gate_policy: LearnedEvidencePolicy | None
    gate_resolved: dict[str, Any] | None
    gate_tokenizer: Any
    fixed_mask: np.ndarray | None


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resize_image(img: np.ndarray, resize_size: tuple[int, int]) -> np.ndarray:
    encoded = tf.image.encode_jpeg(img)
    decoded = tf.io.decode_image(encoded, expand_animations=False, dtype=tf.uint8)
    resized = tf.image.resize(decoded, resize_size, method="lanczos3", antialias=True)
    resized = tf.cast(tf.clip_by_value(tf.round(resized), 0, 255), tf.uint8)
    return resized.numpy()


def get_libero_env(task, resolution: int = 256):
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_libero_image(obs: dict, resize_size: int) -> np.ndarray:
    img = obs["agentview_image"]
    img = img[::-1, ::-1]
    return resize_image(img, (resize_size, resize_size))


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(1e-12, 1.0 - quat[3] * quat[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[..., -1] = 2.0 * out[..., -1] - 1.0
    if binarize:
        out[..., -1] = np.sign(out[..., -1])
    return out


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    out[..., -1] *= -1.0
    return out


def infer_stage_from_ratio(step_ratio: float) -> str:
    if step_ratio < 0.60:
        return "approach"
    if step_ratio < 0.85:
        return "grasp"
    return "place"


def load_norm_stats(norm_stats_path: str) -> dict[str, Any]:
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_unnorm_key(vla, suite_name: str) -> str:
    key = SUITE_TO_UNNORM_KEY.get(suite_name, suite_name)
    if key in getattr(vla, "norm_stats", {}):
        return key
    if suite_name in getattr(vla, "norm_stats", {}):
        return suite_name
    raise KeyError(f"Action un-normalization key not found for suite={suite_name}")


def load_vla_and_processor(model_path: str, norm_stats_path: str, device: torch.device, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = processor.tokenizer
    vla = AutoModelForVision2Seq.from_pretrained(
        model_path,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    vla.norm_stats = load_norm_stats(norm_stats_path)
    return vla, processor, tokenizer


def load_adapter(checkpoint_dir: str, channels: tuple[str, ...], hidden_size: int, device: torch.device) -> SoftEvidenceAdapter:
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    adapter_cfg = resolved["config"]["adapter"]
    adapter = SoftEvidenceAdapter(
        channel_dims={ch: int(FEATURE_DIMS[ch]) for ch in channels},
        channels=channels,
        hidden_size=hidden_size,
        num_global_tokens=int(adapter_cfg["num_global_tokens"]),
        proj_dim=int(adapter_cfg["proj_dim"]),
        dropout=float(adapter_cfg.get("dropout", 0.1)),
    ).to(device)
    state = torch.load(ckpt_dir / "adapter.pt", map_location=device)
    adapter.load_state_dict(state, strict=False)
    adapter.eval()
    return adapter


def load_full_mask(checkpoint_dir: str) -> np.ndarray:
    masks = np.load(Path(checkpoint_dir) / "channel_masks.npy")
    return np.asarray(masks[0], dtype=np.float32)


def load_local_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def load_gate_policy(checkpoint_dir: str, device: torch.device):
    ckpt_dir = Path(checkpoint_dir)
    resolved = json.loads((ckpt_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = resolved["config"]
    model_cfg = cfg["model"]
    channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    text_cfg = dict(model_cfg.get("text_encoder", {}))
    text_encoder_type = str(text_cfg.get("type", "bow"))
    text_vocab_size = 0
    tokenizer = None
    if text_encoder_type == "sequence":
        tokenizer = load_local_tokenizer(str(text_cfg["tokenizer_path"]))
        text_vocab_size = int(len(tokenizer))
    gate_type = str(model_cfg.get("gate_type", "stage_conditioned"))
    if gate_type == "stage_conditioned":
        if bool(cfg.get("route_mixture", {}).get("enabled", False)):
            gate_type = "route_mixture"
        elif bool(cfg.get("latent_phase", {}).get("enabled", False)):
            gate_type = "latent_phase"
    policy = LearnedEvidencePolicy(
        bow_dim=int(model_cfg["bow_dim"]),
        query_bow_dim=int(model_cfg.get("query_bow_dim", 64)),
        ctx_dim=int(model_cfg["ctx_dim"]),
        channel_dim=int(model_cfg["channel_dim"]),
        teacher_hidden=int(model_cfg["teacher_hidden"]),
        student_hidden=int(model_cfg["student_hidden"]),
        channels=channels,
        budget_values=budget_values,
        text_encoder_type=text_encoder_type,
        text_vocab_size=text_vocab_size,
        instruction_max_len=int(text_cfg.get("instruction_max_len", 24)),
        query_max_len=int(text_cfg.get("query_max_len", 12)),
        text_embed_dim=int(text_cfg.get("embed_dim", 96)),
        text_hidden_dim=int(text_cfg.get("hidden_dim", 128)),
        gate_type=gate_type,
        latent_phase_slots=int(model_cfg.get("latent_phase_slots", 8)),
        route_bank=tuple(tuple(r) for r in cfg.get("route_mixture", {}).get("route_bank", [])),
    ).to(device)
    policy.context.load_state_dict(torch.load(ckpt_dir / "context.pt", map_location=device))
    if getattr(policy, "instruction_text_encoder", None) is not None and (ckpt_dir / "instruction_text_encoder.pt").exists():
        policy.instruction_text_encoder.load_state_dict(torch.load(ckpt_dir / "instruction_text_encoder.pt", map_location=device))
    if getattr(policy, "query_text_encoder", None) is not None and (ckpt_dir / "query_text_encoder.pt").exists():
        policy.query_text_encoder.load_state_dict(torch.load(ckpt_dir / "query_text_encoder.pt", map_location=device))
    policy.gate.load_state_dict(torch.load(ckpt_dir / "gate.pt", map_location=device))
    policy.eval()
    return policy, resolved, tokenizer


def make_sequence_ids(tokenizer, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = tokenize_to_arrays(tokenizer, text, max_len)
    return torch.tensor(input_ids.tolist(), dtype=torch.long), torch.tensor(attention_mask.tolist(), dtype=torch.float32)


def pil_to_chw_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    arr = np.asarray(image.resize((image_size, image_size)).convert("RGB"), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr.tolist(), dtype=torch.float32, device=device).unsqueeze(0)


def build_gate_batch(
    image: Image.Image,
    instruction: str,
    query_words: list[str],
    stage: str,
    step_ratio: float,
    policy: LearnedEvidencePolicy,
    resolved: dict[str, Any],
    lazy_tokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model_cfg = resolved["config"]["model"]
    query_text = " ".join(query_words)
    batch = {
        "image": pil_to_chw_tensor(image, int(model_cfg["image_size"]), device),
        "bow": torch.tensor(
            hashed_bow(instruction, int(model_cfg["bow_dim"])).tolist(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "query_bow": torch.tensor(
            hashed_bow(query_text, int(model_cfg.get("query_bow_dim", 64))).tolist(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "instruction_meta": torch.tensor(
            instruction_meta_vector(instruction, query_words).tolist(),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        "ambiguity_vec": torch.zeros((1, 12), dtype=torch.float32, device=device),
        "stage_one_hot": torch.tensor(stage_to_one_hot(stage).tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        "step_ratio": torch.tensor([[float(step_ratio)]], dtype=torch.float32, device=device),
    }
    if policy.use_text_sequence:
        text_cfg = dict(model_cfg.get("text_encoder", {}))
        instruction_ids, instruction_mask = make_sequence_ids(
            lazy_tokenizer,
            instruction,
            int(text_cfg.get("instruction_max_len", 24)),
        )
        query_ids, query_mask = make_sequence_ids(
            lazy_tokenizer,
            query_text,
            int(text_cfg.get("query_max_len", 12)),
        )
        batch["instruction_ids"] = instruction_ids.unsqueeze(0).to(device)
        batch["instruction_mask"] = instruction_mask.unsqueeze(0).to(device)
        batch["query_ids"] = query_ids.unsqueeze(0).to(device)
        batch["query_mask"] = query_mask.unsqueeze(0).to(device)
    else:
        batch["instruction_ids"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        batch["instruction_mask"] = torch.zeros((1, 1), dtype=torch.float32, device=device)
        batch["query_ids"] = torch.zeros((1, 1), dtype=torch.long, device=device)
        batch["query_mask"] = torch.zeros((1, 1), dtype=torch.float32, device=device)
    return batch


def infer_online_mask(
    policy: LearnedEvidencePolicy,
    resolved: dict[str, Any],
    batch: dict[str, torch.Tensor],
    soft_mask_blend: float,
) -> tuple[np.ndarray, dict[str, bool]]:
    cfg = resolved["config"]
    channels = tuple(resolved["channels"])
    budget_values = tuple(int(x) for x in resolved["budget_values"])
    gate_cfg = cfg["gating"]
    stage_conditioning_enabled = bool(cfg.get("stage_conditioning", {}).get("enabled", True))
    route_mixture_cfg = cfg.get("route_mixture", {})
    route_mixture_enabled = bool(route_mixture_cfg.get("enabled", False))
    route_bank_masks = None
    if route_mixture_enabled:
        route_bank = tuple(tuple(route) for route in route_mixture_cfg.get("route_bank", []))
        route_bank_masks = route_bank_tensor(route_bank, channels, batch["image"].device)

    with torch.inference_mode():
        ctx = policy.encode_context(batch)
        stage_one_hot = batch["stage_one_hot"] if stage_conditioning_enabled else torch.zeros_like(batch["stage_one_hot"])
        gate_out = policy.forward_gate(
            ctx,
            batch,
            stage_one_hot,
            phase_temperature=float(cfg.get("latent_phase", {}).get("temperature_end", gate_cfg["temperature_end"])),
            hard_phase=bool(cfg.get("latent_phase", {}).get("hard_assignment", False)),
            training=False,
        )
        if route_bank_masks is not None:
            hard_gates, soft_gates, _, _, _, _ = sample_route_mixture_gates(
                gate_out["route_logits"],
                route_bank_masks,
                budget_values=budget_values,
                temperature=float(gate_cfg["temperature_end"]),
                training=False,
            )
        else:
            hard_gates, channel_probs, _, budget_probs = sample_budget_topk_gates(
                gate_out["channel_logits"],
                gate_out["budget_logits"],
                budget_values=budget_values,
                temperature=float(gate_cfg["temperature_end"]),
                training=False,
            )
            budget_tensor = torch.tensor(budget_values, dtype=budget_probs.dtype, device=budget_probs.device).unsqueeze(0)
            soft_budget = torch.sum(budget_probs * budget_tensor, dim=1)
            denom = torch.clamp(channel_probs.sum(dim=1, keepdim=True), min=1e-6)
            soft_gates = torch.clamp(channel_probs * (soft_budget.unsqueeze(1) / denom), 0.0, 1.0)
        blend = float(np.clip(soft_mask_blend, 0.0, 1.0))
        effective = torch.clamp((1.0 - blend) * hard_gates + blend * soft_gates, 0.0, 1.0)[0].cpu().numpy()
        hard = hard_gates[0].cpu().numpy() > 0.5
    return effective.astype(np.float32), {ch: bool(hard[i]) for i, ch in enumerate(channels)}


def extract_query_words(instruction: str, max_words: int = 8) -> list[str]:
    tokens = [tok for tok in instruction.lower().replace("-", " ").split() if tok and tok not in STOPWORDS]
    phrases: list[str] = []
    seen: set[str] = set()
    for idx, token in enumerate(tokens):
        if token in OBJECT_HINTS:
            if idx > 0 and tokens[idx - 1] not in STOPWORDS:
                phrase = f"{tokens[idx - 1]} {token}"
                if phrase not in seen:
                    seen.add(phrase)
                    phrases.append(phrase)
            if token not in seen:
                seen.add(token)
                phrases.append(token)
        elif len(token) >= 4 and token not in seen:
            seen.add(token)
            phrases.append(token)
        if len(phrases) >= max_words:
            break
    return phrases[:max_words] or ["object"]


class OwlDetector:
    def __init__(self, model_id: str, device: str, score_thresh: float = 0.1, max_total: int = 8) -> None:
        self.score_thresh = float(score_thresh)
        self.max_total = int(max_total)
        device_idx = 0 if device == "cuda" else -1
        model_dtype = torch.float16 if device == "cuda" else torch.float32
        self.pipe = hf_pipeline(
            task="zero-shot-object-detection",
            model=model_id,
            device=device_idx,
            torch_dtype=model_dtype,
        )

    def __call__(self, image: Image.Image, query_words: list[str]) -> list[dict[str, Any]]:
        if not query_words:
            return []
        outputs = self.pipe(image, candidate_labels=query_words)
        dets: list[dict[str, Any]] = []
        for out in outputs:
            score = float(out.get("score", 0.0))
            if score < self.score_thresh:
                continue
            box = out.get("box", {})
            dets.append(
                {
                    "label": str(out.get("label", "object")),
                    "score": score,
                    "bbox": [
                        float(box.get("xmin", 0.0)),
                        float(box.get("ymin", 0.0)),
                        float(box.get("xmax", 0.0)),
                        float(box.get("ymax", 0.0)),
                    ],
                }
            )
        dets.sort(key=lambda item: item["score"], reverse=True)
        return dets[: self.max_total]


def detections_to_bbox_vector(detections: list[dict[str, Any]], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    if not detections:
        return np.zeros((10,), dtype=np.float32)
    boxes = []
    scores = []
    for det in detections:
        x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
        boxes.append([x1 / max(width, 1), y1 / max(height, 1), x2 / max(width, 1), y2 / max(height, 1)])
        scores.append(float(det.get("score", 0.0)))
    bboxes = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    widths = np.clip(bboxes[:, 2] - bboxes[:, 0], 0.0, 1.0)
    heights = np.clip(bboxes[:, 3] - bboxes[:, 1], 0.0, 1.0)
    areas = widths * heights
    cx = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    cy = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    top = int(np.argmax(scores_arr)) if scores_arr.size else 0
    return np.array(
        [
            float(len(bboxes)),
            float(scores_arr.mean()) if scores_arr.size else 0.0,
            float(scores_arr.max()) if scores_arr.size else 0.0,
            float(cx.mean()),
            float(cy.mean()),
            float(widths.mean()),
            float(heights.mean()),
            float(areas.mean()),
            float(cx[top]),
            float(cy[top]),
        ],
        dtype=np.float32,
    )


def edge_vector_from_image(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, threshold1=60, threshold2=160).astype(np.uint8)
    edge_bin = (edge > 0).astype(np.float32)
    if edge_bin.size == 0:
        return np.zeros((5,), dtype=np.float32)
    h, w = edge_bin.shape
    top = edge_bin[: h // 2].mean() if h > 1 else edge_bin.mean()
    bottom = edge_bin[h // 2 :].mean() if h > 1 else edge_bin.mean()
    left = edge_bin[:, : w // 2].mean() if w > 1 else edge_bin.mean()
    right = edge_bin[:, w // 2 :].mean() if w > 1 else edge_bin.mean()
    return np.array([float(edge_bin.mean()), float(top), float(bottom), float(left), float(right)], dtype=np.float32)


def expand_dependencies(gates: dict[str, bool]) -> dict[str, bool]:
    plan = dict(gates)
    if plan.get("relation", False):
        plan["bbox"] = True
    return plan


def extract_online_features(
    image: Image.Image,
    prev_image: Image.Image | None,
    instruction: str,
    query_words: list[str],
    detector: OwlDetector,
    channels: tuple[str, ...],
    gates: dict[str, bool],
) -> tuple[dict[str, np.ndarray], float]:
    plan = expand_dependencies(gates)
    features = {ch: np.zeros((FEATURE_DIMS[ch],), dtype=np.float32) for ch in channels}
    start = time.time()
    need_detections = plan.get("bbox", False) or plan.get("relation", False)
    detections = detector(image, query_words) if need_detections else []
    if "bbox" in channels and plan.get("bbox", False):
        features["bbox"] = detections_to_bbox_vector(detections, image.size)
    if "edge" in channels and plan.get("edge", False):
        features["edge"] = edge_vector_from_image(image)
    if "motion" in channels and plan.get("motion", False):
        motion_u8 = compute_motion_map(prev_image, image)
        features["motion"] = motion_vector_from_map(motion_u8)
    if "relation" in channels and plan.get("relation", False):
        relation_stats = build_relation_stats(
            instruction=instruction,
            query_words=query_words,
            detections=detections,
            image_size=image.size,
        )
        features["relation"] = relation_vector_from_stats(relation_stats)
    return features, time.time() - start


def build_evidence_batch(
    features: dict[str, np.ndarray],
    mask: np.ndarray,
    channels: tuple[str, ...],
    stage: str,
    ratio: float,
    device: torch.device,
) -> SoftEvidenceBatch:
    return SoftEvidenceBatch(
        channel_features={
            ch: torch.tensor(features[ch].tolist(), dtype=torch.float32, device=device).unsqueeze(0)
            for ch in channels
        },
        channel_mask=torch.tensor(mask.tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        stage_one_hot=torch.tensor(stage_to_one_hot(stage).tolist(), dtype=torch.float32, device=device).unsqueeze(0),
        step_ratio=torch.tensor([[float(ratio)]], dtype=torch.float32, device=device),
    )


def predict_action_original(vla, processor, image: Image.Image, instruction: str, model_path: str, device: str, dtype, unnorm_key: str):
    start = time.perf_counter()
    prompt = make_openvla_prompt(instruction, model_path)
    inputs = processor(prompt, image).to(device, dtype=dtype)
    with torch.inference_mode():
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float32), time.perf_counter() - start


def summarize_variant(episode_rows: list[dict[str, Any]], channels: tuple[str, ...]) -> dict[str, Any]:
    completion_all = [float(row["proxy_completion_time_s"]) for row in episode_rows]
    completion_success = [float(row["proxy_completion_time_s"]) for row in episode_rows if row["success"]]
    wall_clock_all = [float(row["wall_clock_completion_time_s"]) for row in episode_rows]
    wall_clock_success = [float(row["wall_clock_completion_time_s"]) for row in episode_rows if row["success"]]
    step_latencies = [float(row["avg_step_latency_s"]) for row in episode_rows if row["control_steps"] > 0]
    out = {
        "n": int(len(episode_rows)),
        "success_rate": float(np.mean([1.0 if row["success"] else 0.0 for row in episode_rows])) if episode_rows else 0.0,
        "avg_completion_time_s": float(np.mean(completion_success)) if completion_success else None,
        "timeout_penalized_completion_time_s": float(np.mean(completion_all)) if completion_all else None,
        "avg_step_latency_s": float(np.mean(step_latencies)) if step_latencies else None,
        "avg_selected_channels": float(np.mean([float(row["avg_selected_channels"]) for row in episode_rows])) if episode_rows else 0.0,
        "avg_wall_clock_completion_time_s": float(np.mean(wall_clock_success)) if wall_clock_success else None,
        "timeout_penalized_wall_clock_completion_time_s": float(np.mean(wall_clock_all)) if wall_clock_all else None,
    }
    for ch in channels:
        out[f"{ch}_keep_rate"] = float(
            np.mean([float(row["keep_counts"].get(ch, 0.0)) for row in episode_rows])
        ) if episode_rows else 0.0
    return out


def _fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.4f}"


def write_summary(
    output_dir: Path,
    suite_name: str,
    run_name: str,
    benchmark_type: str,
    model_rows: dict[str, list[dict[str, Any]]],
    channels: tuple[str, ...],
) -> None:
    summary = {
        "task_suite": suite_name,
        "run_name": run_name,
        "benchmark_type": benchmark_type,
        "completion_metrics_available": True,
        "completion_time_definition": "proxy_sum_of_step_inference_latency_per_episode",
        "raw_wall_clock_completion_time_available": True,
        "models": {
            name: summarize_variant(rows, channels) for name, rows in model_rows.items()
        },
    }
    (output_dir / "paper_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics = [
        "success_rate",
        "avg_completion_time_s",
        "timeout_penalized_completion_time_s",
        "avg_step_latency_s",
        "avg_selected_channels",
        "avg_wall_clock_completion_time_s",
        "timeout_penalized_wall_clock_completion_time_s",
    ] + [f"{ch}_keep_rate" for ch in channels]
    lines = [
        "| Metric | OpenVLA | FullSoft | DynamicSoft |",
        "|---|---:|---:|---:|",
    ]
    model_metrics = summary["models"]
    for metric in metrics:
        lines.append(
            f"| {metric} | "
            f"{_fmt(model_metrics['openvla'].get(metric))} | "
            f"{_fmt(model_metrics['full_soft'].get(metric))} | "
            f"{_fmt(model_metrics['dynamic_soft'].get(metric))} |"
        )
    (output_dir / "summary_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_episode_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def evaluate_variant(
    bundle: ModelBundle,
    args: argparse.Namespace,
    vla,
    processor,
    tokenizer,
    detector: OwlDetector,
    channels: tuple[str, ...],
    unnorm_key: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Any]]:
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks if args.task_limit <= 0 else min(task_suite.n_tasks, args.task_limit)
    resize_size = 224
    episode_rows: list[dict[str, Any]] = []

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"{bundle.name} tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        query_words = extract_query_words(task_description)
        max_steps = args.max_episode_steps_override if args.max_episode_steps_override > 0 else SUITE_TO_MAX_STEPS[args.task_suite_name]

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"{bundle.name} episodes", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            done = False
            prev_image: Image.Image | None = None
            episode_start = time.time()
            step_latencies: list[float] = []
            selected_counts: list[float] = []
            keep_history: list[dict[str, float]] = []
            control_steps = 0

            while t < max_steps + args.num_steps_wait:
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue

                current_np = get_libero_image(obs, resize_size)
                current_image = Image.fromarray(current_np).convert("RGB")
                observation = {
                    "full_image": current_np,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }
                step_ratio = float(control_steps) / float(max(1, max_steps - 1))
                stage = infer_stage_from_ratio(step_ratio)
                step_t0 = time.perf_counter()

                if bundle.name == "openvla":
                    action, step_latency = predict_action_original(
                        vla=vla,
                        processor=processor,
                        image=current_image,
                        instruction=task_description,
                        model_path=args.model_path,
                        device=str(device),
                        dtype=dtype,
                        unnorm_key=unnorm_key,
                    )
                    selected_channels = 0.0
                    keep_counts = {ch: 0.0 for ch in channels}
                else:
                    if bundle.name == "full_soft":
                        hard_gates = {ch: True for ch in channels}
                        effective_mask = np.asarray(bundle.fixed_mask, dtype=np.float32)
                    else:
                        gate_batch = build_gate_batch(
                            image=current_image,
                            instruction=task_description,
                            query_words=query_words,
                            stage=stage,
                            step_ratio=step_ratio,
                            policy=bundle.gate_policy,
                            resolved=bundle.gate_resolved,
                            lazy_tokenizer=bundle.gate_tokenizer,
                            device=device,
                        )
                        effective_mask, hard_gates = infer_online_mask(
                            policy=bundle.gate_policy,
                            resolved=bundle.gate_resolved,
                            batch=gate_batch,
                            soft_mask_blend=args.soft_mask_blend,
                        )
                    features, extract_time = extract_online_features(
                        image=current_image,
                        prev_image=prev_image,
                        instruction=task_description,
                        query_words=query_words,
                        detector=detector,
                        channels=channels,
                        gates=hard_gates,
                    )
                    evidence_batch = build_evidence_batch(
                        features=features,
                        mask=effective_mask,
                        channels=channels,
                        stage=stage,
                        ratio=step_ratio,
                        device=device,
                    )
                    action = predict_action_with_soft_evidence(
                        vla=vla,
                        processor=processor,
                        tokenizer=tokenizer,
                        adapter=bundle.adapter,
                        image=current_image,
                        instruction=task_description,
                        evidence_batch=evidence_batch,
                        model_path=args.model_path,
                        device=device,
                        dtype=dtype,
                        unnorm_key=unnorm_key,
                    )
                    selected_channels = float(sum(1 for value in hard_gates.values() if value))
                    keep_counts = {ch: float(hard_gates.get(ch, False)) for ch in channels}
                    step_latency = time.perf_counter() - step_t0
                    _ = extract_time

                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                control_steps += 1
                step_latencies.append(float(step_latency))
                selected_counts.append(selected_channels)
                keep_history.append(keep_counts)
                prev_image = current_image

                print(
                    f"[{bundle.name}] task={task_id+1}/{num_tasks_in_suite} "
                    f"episode={episode_idx+1}/{args.num_trials_per_task} step={control_steps} "
                    f"latency={step_latency:.4f}s done={done}",
                    flush=True,
                )
                if done:
                    break
                t += 1

            avg_keep = {
                ch: float(np.mean([row[ch] for row in keep_history])) if keep_history else 0.0
                for ch in channels
            }
            episode_row = {
                "model": bundle.name,
                "task_suite": args.task_suite_name,
                "task_id": int(task_id),
                "task_description": task_description,
                "episode_idx": int(episode_idx),
                "success": bool(done),
                "proxy_completion_time_s": float(sum(step_latencies)),
                "wall_clock_completion_time_s": float(time.time() - episode_start),
                "control_steps": int(control_steps),
                "avg_step_latency_s": float(np.mean(step_latencies)) if step_latencies else None,
                "avg_selected_channels": float(np.mean(selected_counts)) if selected_counts else 0.0,
                "keep_counts": avg_keep,
                "num_steps_wait": int(args.num_steps_wait),
                "max_steps": int(max_steps),
            }
            episode_rows.append(episode_row)
    return episode_rows


def warmup_variants(
    vla,
    processor,
    tokenizer,
    detector: OwlDetector,
    channels: tuple[str, ...],
    bundles: tuple[ModelBundle, ...],
    suite_name: str,
    soft_mask_blend: float,
    device: torch.device,
    dtype: torch.dtype,
    model_path: str,
    unnorm_key: str,
) -> None:
    image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    instruction = "pick up the cup"
    query_words = extract_query_words(instruction)
    _ = detector(image, query_words)
    _ = predict_action_original(
        vla=vla,
        processor=processor,
        image=image,
        instruction=instruction,
        model_path=model_path,
        device=str(device),
        dtype=dtype,
        unnorm_key=unnorm_key,
    )
    for bundle in bundles:
        if bundle.name == "openvla":
            continue
        if bundle.name == "full_soft":
            hard_gates = {ch: True for ch in channels}
            effective_mask = np.asarray(bundle.fixed_mask, dtype=np.float32)
        else:
            gate_batch = build_gate_batch(
                image=image,
                instruction=instruction,
                query_words=query_words,
                stage=infer_stage_from_ratio(0.0),
                step_ratio=0.0,
                policy=bundle.gate_policy,
                resolved=bundle.gate_resolved,
                lazy_tokenizer=bundle.gate_tokenizer,
                device=device,
            )
            effective_mask, hard_gates = infer_online_mask(
                policy=bundle.gate_policy,
                resolved=bundle.gate_resolved,
                batch=gate_batch,
                soft_mask_blend=soft_mask_blend,
            )
        features, _ = extract_online_features(
            image=image,
            prev_image=None,
            instruction=instruction,
            query_words=query_words,
            detector=detector,
            channels=channels,
            gates=hard_gates,
        )
        evidence_batch = build_evidence_batch(
            features=features,
            mask=effective_mask,
            channels=channels,
            stage=infer_stage_from_ratio(0.0),
            ratio=0.0,
            device=device,
        )
        _ = predict_action_with_soft_evidence(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            adapter=bundle.adapter,
            image=image,
            instruction=instruction,
            evidence_batch=evidence_batch,
            model_path=model_path,
            device=device,
            dtype=dtype,
            unnorm_key=unnorm_key,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_suite_name", required=True, choices=sorted(SUITE_TO_UNNORM_KEY))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--norm_stats", required=True)
    parser.add_argument("--full_checkpoint_dir", required=True)
    parser.add_argument("--dynamic_checkpoint_dir", required=True)
    parser.add_argument("--gate_checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_trials_per_task", type=int, default=1)
    parser.add_argument("--task_limit", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_episode_steps_override", type=int, default=0)
    parser.add_argument("--soft_mask_blend", type=float, default=0.35)
    parser.add_argument("--owl_model_id", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--owl_score_thresh", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_log_path = output_dir / "benchmark.log"
    benchmark_log_file = open(benchmark_log_path, "w", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeStream(original_stdout, benchmark_log_file)
    sys.stderr = TeeStream(original_stderr, benchmark_log_file)
    try:
        set_seed_everywhere(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        print(f"[info] suite={args.task_suite_name} model={args.model_path} device={device} output={output_dir}", flush=True)

        vla, processor, tokenizer = load_vla_and_processor(args.model_path, args.norm_stats, device, dtype)
        unnorm_key = ensure_unnorm_key(vla, args.task_suite_name)
        hidden_size = int(vla.get_input_embeddings().weight.shape[1])
        full_resolved = json.loads((Path(args.full_checkpoint_dir) / "resolved_config.json").read_text(encoding="utf-8"))
        channels = tuple(full_resolved["channels"])

        full_bundle = ModelBundle(
            name="full_soft",
            adapter=load_adapter(args.full_checkpoint_dir, channels, hidden_size, device),
            gate_policy=None,
            gate_resolved=None,
            gate_tokenizer=None,
            fixed_mask=load_full_mask(args.full_checkpoint_dir),
        )
        dyn_policy, dyn_resolved, dyn_tokenizer = load_gate_policy(args.gate_checkpoint_dir, device)
        dynamic_bundle = ModelBundle(
            name="dynamic_soft",
            adapter=load_adapter(args.dynamic_checkpoint_dir, channels, hidden_size, device),
            gate_policy=dyn_policy,
            gate_resolved=dyn_resolved,
            gate_tokenizer=dyn_tokenizer,
            fixed_mask=None,
        )
        openvla_bundle = ModelBundle(
            name="openvla",
            adapter=None,
            gate_policy=None,
            gate_resolved=None,
            gate_tokenizer=None,
            fixed_mask=None,
        )

        detector = OwlDetector(
            model_id=args.owl_model_id,
            device="cuda" if device.type == "cuda" else "cpu",
            score_thresh=args.owl_score_thresh,
        )
        warmup_variants(
            vla=vla,
            processor=processor,
            tokenizer=tokenizer,
            detector=detector,
            channels=channels,
            bundles=(openvla_bundle, full_bundle, dynamic_bundle),
            suite_name=args.task_suite_name,
            soft_mask_blend=args.soft_mask_blend,
            device=device,
            dtype=dtype,
            model_path=args.model_path,
            unnorm_key=unnorm_key,
        )
        print("[info] warmup complete", flush=True)

        model_rows = {}
        for bundle in (openvla_bundle, full_bundle, dynamic_bundle):
            print(f"[info] evaluating variant={bundle.name}", flush=True)
            rows = evaluate_variant(
                bundle=bundle,
                args=args,
                vla=vla,
                processor=processor,
                tokenizer=tokenizer,
                detector=detector,
                channels=channels,
                unnorm_key=unnorm_key,
                device=device,
                dtype=dtype,
            )
            model_rows[bundle.name] = rows
            write_episode_jsonl(output_dir / f"{bundle.name}_episode_metrics.jsonl", rows)

        write_summary(
            output_dir=output_dir,
            suite_name=args.task_suite_name,
            run_name=args.run_name,
            benchmark_type="libero_closed_loop",
            model_rows=model_rows,
            channels=channels,
        )
        print(f"[ok] summary={output_dir / 'summary_table.md'}", flush=True)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        benchmark_log_file.close()


if __name__ == "__main__":
    main()
