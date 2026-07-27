#!/bin/bash
# Offline teacher trajectory generation for SCOPE.
# Regenerates teacher_reasoning from a math dataset (default: GSM8K train).
# Does not ship any pre-generated data.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
OUTPUT_PATH="${OUTPUT_PATH:-data/gsm8k_teacher_reasoning_qwen3_8b}"
DATASET_NAME="${DATASET_NAME:-openai/gsm8k}"
DATASET_CONFIG="${DATASET_CONFIG:-main}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"

python scripts/gen_teacher_reasoning.py \
  --model_name_or_path "${TEACHER_MODEL}" \
  --dataset_name "${DATASET_NAME}" \
  --dataset_config "${DATASET_CONFIG}" \
  --dataset_split "${DATASET_SPLIT}" \
  --output_path "${OUTPUT_PATH}" \
  --max_new_tokens 2048 \
  --temperature 0.7 \
  --top_p 0.95 \
  --filter_failed_answers

echo "Saved offline teacher reasoning to ${OUTPUT_PATH}"
