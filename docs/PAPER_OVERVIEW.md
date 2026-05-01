# Paper Overview

## Problem

Dense visual reasoning modules can improve VLA policies, but running every evidence channel at every control step is
expensive and difficult to audit. EIGT asks whether a frozen VLA can use a sparse, dynamically selected evidence set
while keeping the route visible.

## Method

EIGT separates the policy path into three parts:

- an evidence bank with visual channels such as bounding boxes, edges, motion, and object relations,
- a dynamic router that selects which channels are useful for the current control step,
- a soft-evidence adapter that injects selected evidence into the frozen VLA action decoder.

The same route is also written into EvidenceTrace records, enabling post-hoc checks of whether the stated rationale
matches the routed evidence.

## Evaluation Axes

The paper evaluates:

- task success and latency tradeoffs against BaseVLA, FullSoft, and reasoning-heavy VLA variants,
- feature/channel ablations to test whether evidence channels are complementary,
- recipe ablations for sparse masks, blend training, distillation, and SeqText-Gate routing,
- trace/audit metrics for route-rationale agreement and useful-evidence mention.

Generated benchmark tables and figures are not included in this repository; use the scripts under `commands/project/`
and `scripts/` to regenerate them locally.
