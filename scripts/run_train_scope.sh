#!/bin/bash
# Train SCOPE (OPD+ST main method).
#
# Pipeline:
#   1) Offline 8B teacher trajectories  (scripts/run_gen_teacher_reasoning.sh)
#   2) Online: OPD+RC + frozen 8B teacher + OPD+ST loss
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-8B}"
TEACHER_REASONING_DATA="${TEACHER_REASONING_DATA:-data/gsm8k_teacher_reasoning_qwen3_8b}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SCOPE}"
LOG_WITH="${LOG_WITH:-none}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-SCOPE}"
TEACHER_ON_CPU="${TEACHER_ON_CPU:-false}"

EXTRA=()
if [[ "${TEACHER_ON_CPU}" == "true" || "${TEACHER_ON_CPU}" == "1" ]]; then
  EXTRA+=(--teacher_on_cpu)
fi
if [[ -n "${TEACHER_DEVICE:-}" ]]; then
  EXTRA+=(--teacher_device "${TEACHER_DEVICE}")
fi

python scripts/train.py \
  --model_name_or_path "${STUDENT_MODEL}" \
  --teacher_model_name_or_path "${TEACHER_MODEL}" \
  --dataset_path "${TEACHER_REASONING_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_config SCOPE \
  --learning_rate 5e-6 \
  --max_grad_norm 0.1 \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --gradient_checkpointing \
  --num_train_epochs 1 \
  --max_completion_length 1024 \
  --max_reasoning_length 1536 \
  --save_steps 100 \
  --logging_steps 2 \
  --attn_implementation flash_attention_2 \
  --torch_dtype bfloat16 \
  --max_length 4096 \
  --beta 0 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k 20 \
  --lmbda 1 \
  --fixed_teacher \
  --use_peft \
  --lora_r 64 \
  --lora_alpha 128 \
  --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --teacher_thinking false \
  --student_thinking false \
  --teacher_reasoning_column teacher_reasoning \
  --opd_rc_max_retries 1 \
  --lambda_wd 0.1 \
  --wd_topk 32 \
  --wd_epsilon 0.05 \
  --wd_sinkhorn_iters 20 \
  --wd_interval 4 \
  --log_with "${LOG_WITH}" \
  --swanlab_project "${SWANLAB_PROJECT}" \
  "${EXTRA[@]}"

echo "Training finished. Checkpoints under ${OUTPUT_DIR}"
