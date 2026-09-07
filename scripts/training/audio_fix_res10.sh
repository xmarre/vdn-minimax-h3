#!/usr/bin/env bash
# Generated-audio correction LoRA on the frozen released VDN/Turbo stack.
#
# Single GPU (e.g. RTX PRO 6000 96 GB):
#   bash scripts/training/audio_fix_res10.sh \
#     data.index_file=/path/to/video_index.jsonl distributed.shard_size=1
#
# Eight GPUs:
#   NPROC_PER_NODE=8 bash scripts/training/audio_fix_res10.sh \
#     data.index_file=/path/to/video_index.jsonl
#
# The source is ckpts/stage-dmd-step-250; Stage-B and Larry's raw Turbo initializer are
# NOT separate inputs to this recipe.
set -euo pipefail
cd "$(dirname "$0")/../.."

NPROC_PER_NODE=${NPROC_PER_NODE:-1}

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  src/training/train_audio_fix.py \
  --config configs/training/audio_fix_res10.yaml \
  "$@"
