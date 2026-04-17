#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-model/Qwen2.5-VL-7B-Instruct}
TRAIN_JSONL=${TRAIN_JSONL:-data/sft/train_sft.jsonl}
DEV_JSONL=${DEV_JSONL:-data/sft/dev_sft.jsonl}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_cot}
DEVICE=${DEVICE:-cuda:0}

python train.py \
  --model_name "$MODEL_NAME" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs 10 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --max_length 2048 \
  --bf16 \
  --grad_ckpt \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --dev_batch_size 2 \
  --test_batch_size 2 \
  --eval_max_new_tokens 512 \
  --eval_iou 0.5 \
  --num_workers 4 \
  --train_bbox_jitter \
  --jitter_beta 0.1 \
  --jitter_gamma 0.1