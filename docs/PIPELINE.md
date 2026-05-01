# EIGT Pipeline

The project scripts are intentionally numbered. The typical paper workflow is:

## 1. Data Subset and Features

```bash
export DATASET=bridge
export RUN_NAME=bridge_eigt
bash commands/project/04_extract_subset.sh
bash commands/project/06_batch_features.sh
```

`06_batch_features.sh` builds the evidence bank used by the router. The current evidence channels are:

- `bbox`: object localization and detection evidence.
- `edge`: shape/contact structure evidence.
- `motion`: temporal or episode-level change evidence.
- `relation`: instruction-to-object relation evidence.

## 2. Router and Soft Evidence Adapter

```bash
bash commands/project/13_train_learned_gating.sh
bash commands/project/16_train_openvla_soft_full.sh
bash commands/project/17_train_openvla_soft_dynamic.sh
```

The full adapter uses all evidence channels. The dynamic adapter uses the learned route mask and is the EIGT policy path.

## 3. Benchmark

```bash
bash commands/project/18_benchmark_openvla_soft_three_way.sh
```

This compares:

- `BaseVLA`: no evidence interface.
- `FullSoft`: all evidence channels.
- `EIGT`: dynamically routed sparse evidence.

## 4. EvidenceTrace and Audit

```bash
bash commands/project/21_build_evidence_trace_dataset.sh
bash commands/project/22_benchmark_evidence_trace_faithfulness.sh
bash commands/project/46_launch_evidencetrace_audit_table_tmux.sh
```

The audit script reports route-rationale agreement, useful-evidence mention rate, average selected channels, and a
composite trace score. It expects generated trace JSONL files under `runs/EvidenceTrace-VLA/`.

## 5. Ablations

```bash
bash commands/project/40_launch_feature_mask_ablation_tmux.sh
bash commands/project/42_launch_recipe_training_ablation_tmux.sh
bash commands/project/43_launch_trace_ablation_tmux.sh
bash commands/project/44_launch_channel_screening_tmux.sh
```

These launch tmux jobs for feature ablations, recipe ablations, trace-supervision ablations, and channel screening.
All result directories remain under `runs/` and should not be committed.
