#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --dataset_mode blockface_unaligned \
  --model stainviz_3d_cycle_gan \
  --context_slices "${CONTEXT_SLICES:-3}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --preprocess "${PREPROCESS:-crop}" \
  "$@"

