# Reproducibility Notes

## What Is Versioned

The repository versions the code path needed to regenerate the experiments:

- routing and soft-evidence model code,
- data extraction and feature scripts,
- training and benchmark launchers,
- configuration files,
- audit and ablation summarization scripts.

## What Is Not Versioned

Large or generated artifacts are excluded:

- `data/`, `datasets/`, and local RLDS copies,
- `runs/`, `reports/`, and generated benchmark tables,
- `artifacts/`, `checkpoints/`, and model weight files,
- `wandb/`, logs, caches, and tmux outputs.

## Recommended Practice

Use a fresh `RUN_NAME` for each experiment and keep a copy of the exact environment variables:

```bash
env | sort | grep -E 'OPENVLA|EIGT|DATASET|RUN|VLA|CUDA|HF|LIBERO|BRIDGE' > runs/$RUN_NAME/env.txt
```

For paper-style ablations, use proportional subsets so the experiment fits the available GPU budget, then rerun the
most important variants at larger scale if needed.
