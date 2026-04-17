#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-outputs/exp_cot/best}
MODEL_NAME=${MODEL_NAME:-model/Qwen2.5-VL-7B-Instruct}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_cot/eval}
TAG=${TAG:-best_eval}
DEVICE=${DEVICE:-cuda:0}

python eval.py \
  --checkpoint "$CHECKPOINT" \
  --model_name "$MODEL_NAME" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --tag "$TAG" \
  --device "$DEVICE" \
  --test_batch_size 2 \
  --num_workers 4 \
  --max_length 2048 \
  --eval_max_new_tokens 512 \
  --eval_iou 0.5 \
  --bf16
