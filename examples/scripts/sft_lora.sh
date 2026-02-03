#!/bin/bash
# Qwen3 SFT with LoRA - Example training script
# Requires: ms-swift, transformers, CUDA GPU with 22GB+ VRAM for 8B model

set -e

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-examples/data/sample_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen3-lora}"
MAX_LENGTH="${MAX_LENGTH:-2048}"

echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 swift sft \
  --model "$MODEL" \
  --tuner_type lora \
  --dataset "$DATASET" \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --max_length "$MAX_LENGTH" \
  --output_dir "$OUTPUT_DIR" \
  --warmup_ratio 0.05 \
  --logging_steps 5 \
  --save_steps 50 \
  --save_total_limit 2

echo "Training complete. Output saved to $OUTPUT_DIR"
