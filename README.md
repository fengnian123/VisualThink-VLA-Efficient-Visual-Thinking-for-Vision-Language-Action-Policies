# EIGT

### Efficient Image-Grounded Thinking for Vision-Language-Action Policies

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10-2f6f9f?style=flat-square"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-c95f3f?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-2f8f68?style=flat-square"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-475569?style=flat-square">
</p>

<p align="center">
  <b>EIGT</b> adds a sparse, auditable, image-grounded evidence interface to frozen VLA policies.
  Instead of asking the policy to verbalize long chains of thought, it routes compact visual cues
  such as <code>bbox</code>, <code>edge</code>, <code>motion</code>, and <code>relation</code>
  through learned soft evidence states before action decoding.
</p>

<p align="center">
  <img src="assets/method_overview.png" width="92%" alt="EIGT method overview">
</p>

## Why EIGT?

Modern VLA policies usually act from raw images and language alone. Adding reasoning can help, but long text traces
and dense auxiliary perception are expensive in closed-loop control. EIGT takes a lighter path:

| Design Goal | EIGT Choice |
| --- | --- |
| Keep control efficient | Route only a sparse subset of evidence channels per step. |
| Stay image-grounded | Use structured visual cues instead of verbose prompt text. |
| Preserve the base policy | Freeze the VLA backbone and train a small evidence adapter. |
| Make behavior inspectable | Export route masks, utility ranks, and channel-grounded rationales. |

## Main Results at a Glance

<p align="center">
  <img src="assets/benchmark_tradeoff.png" width="92%" alt="Benchmark success-latency tradeoff">
</p>

EIGT is designed for the success-latency tradeoff: it keeps the action path close to the frozen VLA policy while
selectively injecting only the evidence channels needed for the current step.

<p align="center">
  <img src="assets/channel_screening.png" width="82%" alt="Evidence channel screening dashboard">
</p>

The final evidence bank is intentionally compact. Heavy or weakly routed candidates such as depth, segmentation,
and caption/query-style text are screened out when they add cost without robust sparse-route utility.

## EvidenceTrace-VLA

EIGT also builds an audit layer, **EvidenceTrace-VLA**, that records what evidence was routed and why.

<p align="center">
  <img src="assets/evidencetrace_pipeline.png" width="92%" alt="EvidenceTrace-VLA construction pipeline">
</p>

Each trace stores the instruction, route mask, selected evidence names, counterfactual utility ranking, compact
channel snippets, channel-grounded rationale, and action intent. These traces support both supervision and
faithfulness diagnostics.

<p align="center">
  <img src="assets/routing_stage_sensitivity.png" width="78%" alt="Routing stage sensitivity">
</p>

The route is not just a static channel subset: selected evidence changes across manipulation stages such as
approach, grasp, and place.

## Repository Layout

```text
commands/project/     End-to-end shell recipes for extraction, training, benchmarking, and ablations
configs/              Gating and soft-evidence configuration files
models/               EIGT gating and soft-evidence adapter modules
scripts/              Feature extraction, training, benchmarking, trace, and summary scripts
prismatic/            OpenVLA/Prismatic base code
vla-scripts/          OpenVLA training and deployment entry points
docs/                 Practical setup and reproducibility notes
assets/               Paper figures used by this project page
```

## Quick Start

```bash
git clone https://github.com/fengnian123/EIGT-Efficient-Image-Grounded-Thinking-for-Vision-Language-Action-Policies.git
cd EIGT-Efficient-Image-Grounded-Thinking-for-Vision-Language-Action-Policies

conda create -n eigt python=3.10 -y
conda activate eigt
pip install -e .
pip install -r requirements-min.txt
```

Set paths through environment variables rather than editing scripts:

```bash
export OPENVLA_ROOT="$PWD"
export DATASET=bridge
export RUN_NAME=bridge_eigt_demo
export VLA_PATH=openvla/openvla-7b
export BRIDGE_DATA_ROOT=/path/to/rlds/datasets
```

Run a minimal environment check:

```bash
bash commands/project/02_check_env.sh
```

## Core Workflow

Most commands write to `runs/$RUN_NAME/`, which is intentionally ignored by git.

```bash
# 1. Extract a proportional RLDS subset and visual/evidence features.
bash commands/project/04_extract_subset.sh
bash commands/project/06_batch_features.sh

# 2. Train sparse routing and soft-evidence adapters.
bash commands/project/13_train_learned_gating.sh
bash commands/project/16_train_openvla_soft_full.sh
bash commands/project/17_train_openvla_soft_dynamic.sh

# 3. Compare BaseVLA, FullSoft, and EIGT.
bash commands/project/18_benchmark_openvla_soft_three_way.sh

# 4. Build and audit EvidenceTrace outputs.
bash commands/project/21_build_evidence_trace_dataset.sh
bash commands/project/22_benchmark_evidence_trace_faithfulness.sh
```

See [docs/PIPELINE.md](docs/PIPELINE.md) for the longer paper-oriented workflow, including ablations.

## Main Entry Points

- `models/evidence_gating.py`: learned dynamic evidence router.
- `models/openvla_soft_evidence.py`: soft evidence adapter and action prediction wrapper.
- `scripts/build_evidence_trace_dataset.py`: converts routed evidence into auditable EvidenceTrace rows.
- `scripts/benchmark_evidencetrace_audit_methods.py`: builds the supervision/audit table used by the paper.
- `commands/project/46_launch_evidencetrace_audit_table_tmux.sh`: tmux launcher for the audit benchmark.

## Release Scope

This repository includes source code, configuration files, run recipes, and paper-page assets.

It intentionally does **not** include:

- robot/RLDS datasets,
- pretrained or fine-tuned model weights,
- generated traces, cached features, or benchmark outputs,
- local logs or `wandb` runs.

Use the scripts to regenerate those artifacts locally after downloading the required datasets and base checkpoints.

## Citation

If you use this code, please cite the EIGT paper once the final citation is available. This release also builds on
OpenVLA, so please cite the original OpenVLA work when using the base policy code.

```bibtex
@misc{eigt2026,
  title  = {EIGT: Efficient Image-Grounded Thinking for Vision-Language-Action Policies},
  author = {Anonymous},
  year   = {2026},
  note   = {Research code release}
}
```

## Acknowledgements

This repository reuses and extends the OpenVLA/Prismatic codebase. The EIGT-specific additions are the evidence
routing, soft-evidence adapter, trace governance, and paper ablation workflows.
