#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --dataset_mode blockface_paired \
  --model stainviz_3d_pix2pix \
  --context_slices "${CONTEXT_SLICES:-3}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --preprocess "${PREPROCESS:-crop}" \
  "$@"

