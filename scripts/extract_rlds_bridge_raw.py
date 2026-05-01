#!/usr/bin/env python3
"""Extract frame image, instruction, and action labels from raw RLDS TFRecord."""

import argparse
import glob
import json
import io
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tensorflow as tf  # type: ignore
except Exception:
    tf = None


def _to_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    if hasattr(value, "tobytes"):
        return value.tobytes()
    return bytes(value)


def _decode_instruction_step(step_value) -> str:
    if isinstance(step_value, bytes):
        return step_value.decode("utf-8", errors="ignore").strip()
    if np.isscalar(step_value):
        return str(step_value).strip()
    arr = np.asarray(step_value)
    if arr.dtype.kind in {"S", "O"}:
        return _to_bytes(arr).decode("utf-8", errors="ignore").strip()
    if arr.dtype.kind in {"i", "u"}:
        return "".join(chr(int(x)) for x in arr.reshape(-1) if int(x)).strip()
    return str(arr).strip()


def _pick_image_key(feat: dict, preferred_key: str | None) -> str:
    if preferred_key and preferred_key in feat:
        return preferred_key
    candidates = [
        "steps/observation/image_0",
        "steps/observation/image",
        "steps/observation/rgb/image",
        "steps/observation/wrist_image",
        "steps/observation/image_1",
    ]
    for k in candidates:
        if k in feat:
            return k
    raise KeyError("No supported image key found in record.")


def _pick_language_key(feat: dict, preferred_key: str | None) -> str:
    if preferred_key and preferred_key in feat:
        return preferred_key
    candidates = [
        "steps/language_instruction",
        "steps/observation/natural_language_instruction",
        "steps/observation/instruction",
        "steps/instruction",
    ]
    for k in candidates:
        if k in feat:
            return k
    raise KeyError("No supported language key found in record.")


def _pick_language_key_fallback(feat: dict, preferred_key: str | None) -> str:
    return _pick_language_key(feat, preferred_key)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _invert_gripper(x: np.ndarray) -> np.ndarray:
    return 1.0 - _clip01(x)


def _rel2abs_gripper(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).reshape(-1)
    thresholded = np.where(actions < -0.1, 1.0, np.where(actions > 0.1, -1.0, 0.0)).astype(np.float32)
    nonzero = np.nonzero(thresholded != 0)[0]
    start = -thresholded[nonzero[0]] if nonzero.size else 1.0
    carry = start
    out = np.empty_like(thresholded, dtype=np.float32)
    for i, val in enumerate(thresholded):
        if val != 0:
            carry = val
        out[i] = carry
    return out / 2.0 + 0.5


def _nyu_franka_play_actions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"nyu_franka_play expects action dim >= 8, got shape={arr.shape}")
    world_and_rot = arr[:, -8:-2].astype(np.float32)
    grip = _clip01(arr[:, -2:-1]).astype(np.float32)
    return np.concatenate([world_and_rot, grip], axis=1).astype(np.float32)


def _stanford_hydra_actions(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"stanford_hydra expects action dim >= 7, got shape={arr.shape}")
    core = arr[:, :6].astype(np.float32)
    grip = _invert_gripper(arr[:, -1:]).astype(np.float32)
    return np.concatenate([core, grip], axis=1).astype(np.float32)


def _kuka_actions_from_components(world: np.ndarray, rot: np.ndarray, grip_rel: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    grip_rel = np.asarray(grip_rel, dtype=np.float32).reshape(len(world))
    grip = _rel2abs_gripper(grip_rel).reshape(len(world), 1)
    return np.concatenate([world, rot, grip], axis=1).astype(np.float32)


def _zero_gripper_actions(world: np.ndarray, rot: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    grip = np.zeros((len(world), 1), dtype=np.float32)
    return np.concatenate([world, rot, grip], axis=1).astype(np.float32)


def _invert_absolute_gripper_actions(world: np.ndarray, rot: np.ndarray, grip_abs: np.ndarray) -> np.ndarray:
    world = np.asarray(world, dtype=np.float32)
    rot = np.asarray(rot, dtype=np.float32)
    grip_abs = np.asarray(grip_abs, dtype=np.float32).reshape(len(world), 1)
    grip = _invert_gripper(_clip01(grip_abs))
    return np.concatenate([world, rot, grip], axis=1).astype(np.float32)


def _parse_actions_from_features(
    feat: dict,
    n_steps: int,
    action_mode: str | None,
) -> list[list[float]]:
    if n_steps <= 0:
        return []

    mode = action_mode or "auto"
    feature_names = set(feat.keys())

    if mode in {"auto", "taco_play"} and "steps/action/rel_actions_world" in feature_names:
        actions_flat = np.asarray(feat["steps/action/rel_actions_world"].float_list.value, dtype=np.float32)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        arr[:, -1:] = _clip01(arr[:, -1:])
        return arr.tolist()

    if mode in {"auto", "roboturk"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(feat["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        grip = np.asarray(feat["steps/action/gripper_closedness_action"].float_list.value, dtype=np.float32).reshape(n_steps, 1)
        arr = np.concatenate([world, rot, _invert_gripper(grip)], axis=1).astype(np.float32)
        return arr.tolist()

    if mode in {"auto", "jaco_play"} and {
        "steps/action/world_vector",
        "steps/action/gripper_closedness_action",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.zeros_like(world, dtype=np.float32)
        grip_rel = np.asarray(feat["steps/action/gripper_closedness_action"].float_list.value, dtype=np.float32).reshape(n_steps)
        grip = _rel2abs_gripper(grip_rel).reshape(n_steps, 1)
        arr = np.concatenate([world, rot, grip], axis=1).astype(np.float32)
        return arr.tolist()

    if mode in {"auto", "kuka"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(feat["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        grip_rel = np.asarray(feat["steps/action/gripper_closedness_action"].float_list.value, dtype=np.float32).reshape(n_steps)
        arr = _kuka_actions_from_components(world, rot, grip_rel)
        return arr.tolist()

    if mode in {"auto", "berkeley_cable_routing"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(feat["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        arr = _zero_gripper_actions(world, rot)
        return arr.tolist()

    if mode in {"auto", "viola"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(feat["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        grip_abs = np.asarray(feat["steps/action/gripper_closedness_action"].float_list.value, dtype=np.float32).reshape(n_steps, 1)
        arr = _invert_absolute_gripper_actions(world, rot, grip_abs)
        return arr.tolist()

    if mode in {"auto", "berkeley_autolab_ur5"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(feature_names):
        world = np.asarray(feat["steps/action/world_vector"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(feat["steps/action/rotation_delta"].float_list.value, dtype=np.float32).reshape(n_steps, 3)
        grip_rel = np.asarray(feat["steps/action/gripper_closedness_action"].float_list.value, dtype=np.float32).reshape(n_steps)
        arr = _kuka_actions_from_components(world, rot, grip_rel)
        return arr.tolist()

    if mode in {"auto", "nyu_franka_play"} and "steps/action" in feature_names:
        actions_flat = np.asarray(feat["steps/action"].float_list.value, dtype=np.float32)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        arr = _nyu_franka_play_actions(arr)
        return arr.tolist()

    if mode in {"auto", "stanford_hydra"} and "steps/action" in feature_names:
        actions_flat = np.asarray(feat["steps/action"].float_list.value, dtype=np.float32)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        arr = _stanford_hydra_actions(arr)
        return arr.tolist()

    if "steps/action" in feature_names:
        actions_flat = np.asarray(feat["steps/action"].float_list.value, dtype=np.float32)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        return arr.tolist()

    raise KeyError(f"No supported action key found for action_mode={mode}.")


def _parse_actions_from_record(
    record: dict,
    n_steps: int,
    action_mode: str | None,
) -> list[list[float]]:
    if n_steps <= 0:
        return []

    mode = action_mode or "auto"
    record_keys = set(record.keys())

    if mode in {"auto", "taco_play"} and "steps/action/rel_actions_world" in record_keys:
        arr = np.asarray(record["steps/action/rel_actions_world"], dtype=np.float32).reshape(n_steps, -1)
        arr[:, -1:] = _clip01(arr[:, -1:])
        return arr.tolist()

    if mode in {"auto", "roboturk"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(record["steps/action/rotation_delta"], dtype=np.float32).reshape(n_steps, 3)
        grip = np.asarray(record["steps/action/gripper_closedness_action"], dtype=np.float32).reshape(n_steps, 1)
        arr = np.concatenate([world, rot, _invert_gripper(grip)], axis=1).astype(np.float32)
        return arr.tolist()

    if mode in {"auto", "jaco_play"} and {
        "steps/action/world_vector",
        "steps/action/gripper_closedness_action",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.zeros_like(world, dtype=np.float32)
        grip_rel = np.asarray(record["steps/action/gripper_closedness_action"], dtype=np.float32).reshape(n_steps)
        grip = _rel2abs_gripper(grip_rel).reshape(n_steps, 1)
        arr = np.concatenate([world, rot, grip], axis=1).astype(np.float32)
        return arr.tolist()

    if mode in {"auto", "kuka"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(record["steps/action/rotation_delta"], dtype=np.float32).reshape(n_steps, 3)
        grip_rel = np.asarray(record["steps/action/gripper_closedness_action"], dtype=np.float32).reshape(n_steps)
        arr = _kuka_actions_from_components(world, rot, grip_rel)
        return arr.tolist()

    if mode in {"auto", "berkeley_cable_routing"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(record["steps/action/rotation_delta"], dtype=np.float32).reshape(n_steps, 3)
        arr = _zero_gripper_actions(world, rot)
        return arr.tolist()

    if mode in {"auto", "viola"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(record["steps/action/rotation_delta"], dtype=np.float32).reshape(n_steps, 3)
        grip_abs = np.asarray(record["steps/action/gripper_closedness_action"], dtype=np.float32).reshape(n_steps, 1)
        arr = _invert_absolute_gripper_actions(world, rot, grip_abs)
        return arr.tolist()

    if mode in {"auto", "berkeley_autolab_ur5"} and {
        "steps/action/world_vector",
        "steps/action/rotation_delta",
        "steps/action/gripper_closedness_action",
    }.issubset(record_keys):
        world = np.asarray(record["steps/action/world_vector"], dtype=np.float32).reshape(n_steps, 3)
        rot = np.asarray(record["steps/action/rotation_delta"], dtype=np.float32).reshape(n_steps, 3)
        grip_rel = np.asarray(record["steps/action/gripper_closedness_action"], dtype=np.float32).reshape(n_steps)
        arr = _kuka_actions_from_components(world, rot, grip_rel)
        return arr.tolist()

    if mode in {"auto", "nyu_franka_play"} and "steps/action" in record_keys:
        actions_flat = np.asarray(record["steps/action"], dtype=np.float32).reshape(-1)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        arr = _nyu_franka_play_actions(arr)
        return arr.tolist()

    if mode in {"auto", "stanford_hydra"} and "steps/action" in record_keys:
        actions_flat = np.asarray(record["steps/action"], dtype=np.float32).reshape(-1)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        arr = _stanford_hydra_actions(arr)
        return arr.tolist()

    if "steps/action" in record_keys:
        actions_flat = np.asarray(record["steps/action"], dtype=np.float32).reshape(-1)
        action_dim = max(1, int(actions_flat.size // n_steps))
        arr = actions_flat.reshape(n_steps, action_dim).astype(np.float32)
        return arr.tolist()

    raise KeyError(f"No supported action key found for action_mode={mode}.")


def parse_episode(
    raw_bytes: bytes,
    preferred_image_key: str | None = None,
    preferred_language_key: str | None = None,
    action_mode: str | None = None,
) -> dict:
    ex = tf.train.Example()
    ex.ParseFromString(raw_bytes)
    feat = ex.features.feature

    image_key = _pick_image_key(feat, preferred_image_key)
    language_key = _pick_language_key(feat, preferred_language_key)
    images = feat[image_key].bytes_list.value
    n_steps = len(images)
    if n_steps == 0:
        return {
            "n_steps": 0,
            "images": [],
            "langs": [],
            "actions": [],
            "image_key": image_key,
            "language_key": language_key,
        }
    lang_feat = feat[language_key]
    if lang_feat.bytes_list.value:
        langs = list(lang_feat.bytes_list.value)
    elif lang_feat.int64_list.value:
        lang_arr = np.asarray(lang_feat.int64_list.value, dtype=np.int64)
        if lang_arr.size == 512:
            single = _decode_instruction_step(lang_arr)
            langs = [single for _ in range(n_steps)]
        elif lang_arr.size % n_steps == 0:
            lang_width = max(1, int(lang_arr.size // n_steps))
            langs = [
                _decode_instruction_step(lang_arr[i * lang_width : (i + 1) * lang_width])
                for i in range(n_steps)
            ]
        else:
            single = _decode_instruction_step(lang_arr)
            langs = [single for _ in range(n_steps)]
    else:
        langs = ["" for _ in range(n_steps)]

    if len(langs) < n_steps:
        filler = langs[-1] if langs else ""
        langs.extend([filler] * (n_steps - len(langs)))
    elif len(langs) > n_steps:
        langs = langs[:n_steps]

    actions = _parse_actions_from_features(feat, n_steps=n_steps, action_mode=action_mode)

    return {
        "n_steps": n_steps,
        "images": images,
        "langs": langs,
        "actions": actions,
        "image_key": image_key,
        "language_key": language_key,
    }


def parse_episode_fallback(
    record: dict,
    preferred_image_key: str | None = None,
    preferred_language_key: str | None = None,
    action_mode: str | None = None,
) -> dict:
    image_key = _pick_image_key(record, preferred_image_key)
    language_key = _pick_language_key_fallback(record, preferred_language_key)
    images_arr = np.asarray(record[image_key])
    langs_arr = np.asarray(record[language_key])
    n_steps = int(len(images_arr))
    if n_steps == 0:
        return {
            "n_steps": 0,
            "images": [],
            "langs": [],
            "actions": [],
            "image_key": image_key,
            "language_key": language_key,
        }

    if langs_arr.ndim == 1 and len(langs_arr) == n_steps and langs_arr.dtype.kind in {"S", "O"}:
        langs = [_decode_instruction_step(v) for v in langs_arr]
    elif langs_arr.size == 512:
        single = _decode_instruction_step(langs_arr)
        langs = [single for _ in range(n_steps)]
    else:
        if langs_arr.size % n_steps == 0:
            lang_width = max(1, int(langs_arr.size // n_steps))
            langs = [
                _decode_instruction_step(langs_arr[i * lang_width : (i + 1) * lang_width])
                for i in range(n_steps)
            ]
        else:
            single = _decode_instruction_step(langs_arr)
            langs = [single for _ in range(n_steps)]

    if not langs:
        langs = ["" for _ in range(n_steps)]
    elif len(langs) < n_steps:
        langs.extend([langs[-1]] * (n_steps - len(langs)))
    elif len(langs) > n_steps:
        langs = langs[:n_steps]

    actions = _parse_actions_from_record(record, n_steps=n_steps, action_mode=action_mode)
    images = [_to_bytes(v) for v in images_arr]

    return {
        "n_steps": n_steps,
        "images": images,
        "langs": langs,
        "actions": actions,
        "image_key": image_key,
        "language_key": language_key,
    }


def iter_records(tfrecord_glob: str, skip_bad_shards: bool = False):
    if tf is not None:
        tf.config.set_visible_devices([], "GPU")
        paths = sorted(tf.io.gfile.glob(tfrecord_glob))
        if not paths:
            raise FileNotFoundError(f"no TFRecord files matched: {tfrecord_glob}")
        for path in paths:
            try:
                dataset = tf.data.TFRecordDataset([path])
                for raw in dataset:
                    yield raw.numpy(), "tensorflow"
            except tf.errors.OpError as exc:
                if not skip_bad_shards:
                    raise
                print(f"[warn] skip bad tfrecord shard: {path} ({type(exc).__name__}: {exc})")
        return

    from tfrecord.reader import tfrecord_loader

    paths = sorted(glob.glob(tfrecord_glob))
    if not paths:
        raise FileNotFoundError(f"no TFRecord files matched: {tfrecord_glob}")
    for path in paths:
        for record in tfrecord_loader(path, None, description=None):
            yield record, "tfrecord"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfrecord_glob", required=True, help="Example: /path/*.tfrecord-*")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=32)
    parser.add_argument("--step_stride", type=int, default=1)
    parser.add_argument(
        "--image_key",
        default=None,
        help="Optional explicit image key, e.g. steps/observation/image_0 or steps/observation/image",
    )
    parser.add_argument(
        "--language_key",
        default=None,
        help="Optional explicit language key, e.g. steps/language_instruction or steps/observation/instruction",
    )
    parser.add_argument(
        "--resize_image_size",
        type=int,
        default=0,
        help="Optional square size for resizing extracted images before writing JPEGs. 0 disables resizing.",
    )
    parser.add_argument(
        "--action_mode",
        default="auto",
        help="Action parsing mode: auto | direct | taco_play | roboturk | jaco_play | nyu_franka_play | stanford_hydra | kuka | berkeley_cable_routing | viola | berkeley_autolab_ur5",
    )
    parser.add_argument(
        "--skip_bad_shards",
        action="store_true",
        help="Skip unreadable/truncated TFRecord shards instead of aborting extraction.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    img_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    written = 0
    backend = None
    with manifest_path.open("w", encoding="utf-8") as f:
        for ep_idx, (raw, rec_backend) in enumerate(
            iter_records(args.tfrecord_glob, skip_bad_shards=args.skip_bad_shards)
        ):
            if ep_idx >= args.max_episodes:
                break
            backend = rec_backend
            if rec_backend == "tensorflow":
                ep = parse_episode(
                    raw,
                    preferred_image_key=args.image_key,
                    preferred_language_key=args.language_key,
                    action_mode=args.action_mode,
                )
            else:
                ep = parse_episode_fallback(
                    raw,
                    preferred_image_key=args.image_key,
                    preferred_language_key=args.language_key,
                    action_mode=args.action_mode,
                )
            n_steps = ep["n_steps"]
            if n_steps == 0:
                continue

            for step_idx in range(0, min(n_steps, args.max_steps_per_episode), args.step_stride):
                image_path = img_dir / f"ep{ep_idx:06d}_s{step_idx:04d}.jpg"
                image_bytes = ep["images"][step_idx]
                if args.resize_image_size > 0:
                    with Image.open(io.BytesIO(image_bytes)) as im:
                        im = im.convert("RGB").resize((args.resize_image_size, args.resize_image_size), Image.Resampling.BILINEAR)
                        im.save(image_path, format="JPEG", quality=95)
                else:
                    image_path.write_bytes(image_bytes)
                instruction_value = ep["langs"][step_idx]
                instruction = (
                    instruction_value.decode("utf-8", errors="ignore").strip()
                    if isinstance(instruction_value, bytes)
                    else str(instruction_value).strip()
                )

                rec = {
                    "episode_idx": ep_idx,
                    "step_idx": step_idx,
                    "image_path": str(image_path),
                    "instruction": instruction,
                    "action": ep["actions"][step_idx],
                    "image_key": ep["image_key"],
                    "language_key": ep["language_key"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"[ok] backend: {backend or 'unknown'}")
    print(f"[ok] wrote manifest: {manifest_path}")
    print(f"[ok] extracted samples: {written}")


if __name__ == "__main__":
    main()
