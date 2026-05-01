# Getting Started

## Environment

EIGT follows the OpenVLA dependency stack. A minimal setup is:

```bash
conda create -n eigt python=3.10 -y
conda activate eigt
pip install -e .
pip install -r requirements-min.txt
```

For full training and closed-loop evaluation, install the OpenVLA training dependencies from `pyproject.toml` and
FlashAttention if your GPU stack supports it.

## Required External Assets

The repository does not ship datasets or model weights. Provide them through environment variables:

```bash
export OPENVLA_ROOT="$PWD"
export VLA_PATH=openvla/openvla-7b
export BRIDGE_DATA_ROOT=/path/to/bridge_or_oxe_rlds
export LIBERO_DATA_ROOT=/path/to/modified_libero_rlds
```

Use local checkpoint paths instead of Hugging Face model ids when running in offline mode.

## Smoke Test

```bash
bash commands/project/02_check_env.sh
```

Then run a small extraction and feature pass:

```bash
export DATASET=bridge
export RUN_NAME=smoke_bridge
export EXTRACT_FRACTION=0.001
export MAX_STEPS_PER_EPISODE=2
bash commands/project/04_extract_subset.sh
bash commands/project/06_batch_features.sh
```

Outputs are written under `runs/smoke_bridge/` and are ignored by git.
