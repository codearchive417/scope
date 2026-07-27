# Hyperparameters for SCOPE

Defaults match `scripts/run_train_scope.sh`.

## Models

| Role | Default |
|------|---------|
| Student | Qwen3-1.7B + LoRA |
| Teacher | Qwen3-8B (frozen) |
| Offline trajectories | Qwen3-8B generations written by `scripts/gen_teacher_reasoning.py` |

## Optimizer / batch

| Param | Value |
|-------|-------|
| `learning_rate` | `5e-6` |
| `lr_scheduler_type` | linear (TRL default) |
| `warmup_ratio` / `warmup_steps` | `0` |
| `optim` | `adamw_torch` |
| `max_grad_norm` | `0.1` |
| `per_device_train_batch_size` | `1` |
| `gradient_accumulation_steps` | `8` |
| `gradient_checkpointing` | true |
| `num_train_epochs` | `1` |
| `torch_dtype` | `bfloat16` |

## LoRA (student)

| Param | Value |
|-------|-------|
| `lora_r` | `64` |
| `lora_alpha` | `128` |
| `lora_target_modules` | `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` |
| `fixed_teacher` | true |

## Sequence lengths

| Param | Value | Use |
|-------|-------|-----|
| `max_length` | `4096` | collator tokenize cap |
| `max_completion_length` | `1024` | on-policy sampling |
| `max_reasoning_length` | `1536` | OPD+RC generation |

## Sampling

| Param | Value |
|-------|-------|
| `temperature` | `1.0` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| student / teacher thinking | false / false |

## OPD+ST

| Param | Value |
|-------|-------|
| `lambda_wd` | `0.1` |
| `wd_topk` | `32` |
| `wd_epsilon` | `0.05` |
| `wd_sinkhorn_iters` | `20` |
| `wd_interval` | `4` |

## OPD+RC

| Param | Value |
|-------|-------|
| `opd_rc_max_retries` | `1` |
| `teacher_reasoning_column` | `teacher_reasoning` |

## Offline trajectory generation

| Param | Value |
|-------|-------|
| `max_new_tokens` | `2048` |
| `temperature` | `0.7` |
| `top_p` | `0.95` |
| `filter_failed_answers` | on |
