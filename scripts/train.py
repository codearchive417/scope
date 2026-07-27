import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.gold import GOLDConfig

from scope.experiment_logging import init_experiment_logging, resolve_report_to
from scope.trainer import SCOPETrainer

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


@dataclass
class CustomScriptArguments(ScriptArguments):
    """SCOPE training script arguments (main method only)."""

    fixed_teacher: bool = field(
        default=True,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher for self-distill paths. "
            "Only works with use_peft=True. With a separate teacher model, teacher forward uses "
            "the loaded frozen teacher; fixed_teacher still applies to student LoRA handling."
        },
    )
    run_config: str = field(
        default="SCOPE",
        metadata={
            "help": "Run name for this experiment. Used for the output directory "
            "(appended to output_dir) and WandB/SwanLab run name."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    max_reasoning_length: int = field(
        default=1536,
        metadata={"help": "Maximum tokens for online OPD+RC generation."},
    )
    lambda_wd: float = field(
        default=0.1,
        metadata={"help": "Weight for the token-level Wasserstein distance term in OPD+ST loss."},
    )
    wd_topk: int = field(
        default=32,
        metadata={"help": "Top-k support size per side when building the WD token support union."},
    )
    wd_epsilon: float = field(
        default=0.05,
        metadata={"help": "Entropic regularization strength for Sinkhorn optimal transport."},
    )
    wd_sinkhorn_iters: int = field(
        default=20,
        metadata={"help": "Number of Sinkhorn iterations for Wasserstein approximation."},
    )
    wd_interval: int = field(
        default=4,
        metadata={
            "help": "Compute WD loss every N valid completion tokens to reduce compute. "
            "Set to 1 to compute WD on every valid token."
        },
    )
    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "Mutually exclusive with a separate --teacher_model_name_or_path."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    student_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the student during rollout. "
            "Default False (SCOPE main setup)."
        },
    )
    teacher_thinking: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable Qwen3 thinking mode for the teacher privileged context. "
            "Default False (SCOPE main setup)."
        },
    )
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to dataset with teacher_reasoning column (from scripts/gen_teacher_reasoning.py). Required."
        },
    )
    teacher_reasoning_column: str = field(
        default="teacher_reasoning",
        metadata={"help": "Column name for offline teacher reasoning trajectories."},
    )
    opd_rc_max_retries: int = field(
        default=1,
        metadata={"help": "Max retries when OPD+RC fails numerical answer check."},
    )
    opd_rc_min_len_ratio: float | None = field(
        default=None,
        metadata={
            "help": "Min rewrite/reference char length ratio (vs offline teacher_reasoning). "
            "Retry or fallback if too short."
        },
    )
    opd_rc_refresh_steps: int = field(
        default=1,
        metadata={"help": "Regenerate OPD+RC outputs every N global steps (1 = every step)."},
    )
    log_with: str = field(
        default="wandb",
        metadata={
            "help": "Experiment tracking backend: 'wandb', 'swanlab', 'both', or 'none'. "
            "SwanLab requires `pip install swanlab` and works with transformers>=4.50 via report_to."
        },
    )
    swanlab_project: str = field(
        default="SCOPE",
        metadata={"help": "SwanLab project name. Only used when --log_with swanlab or both."},
    )
    swanlab_workspace: str = field(
        default=None,
        metadata={
            "help": "SwanLab workspace/team name. Optional; can also set env SWANLAB_WORKSPACE."
        },
    )
    log_traces: bool = field(
        default=True,
        metadata={
            "help": "Log full training traces (offline teacher reasoning, OPD+RC, distill context, "
            "on-policy completion) to SwanLab/WandB and local JSON."
        },
    )
    log_traces_steps: int = field(
        default=10,
        metadata={"help": "Upload training traces every N global steps. Only used when --log_traces."},
    )
    generation_debug: bool = field(
        default=False,
        metadata={"help": "Print per-generation timing/debug logs during training."},
    )
    save_generations_local: bool = field(
        default=None,
        metadata={
            "help": "Write generations_step_*.json under output_dir. "
            "Default: same as --log_traces when unset."
        },
    )
    num_traces_per_log: int = field(
        default=1,
        metadata={"help": "Number of random batch samples to upload per trace log event."},
    )
    teacher_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Keep the separate teacher model on CPU (forward copies tensors GPU↔CPU). "
            "Useful when GPU memory is tight for 8B+1.7B dual-model training."
        },
    )
    teacher_device: str | None = field(
        default=None,
        metadata={
            "help": "Pin separate teacher to a device, e.g. cuda:1 with CUDA_VISIBLE_DEVICES=0,3."
        },
    )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.max_reasoning_length = script_args.max_reasoning_length
    training_args.remove_unused_columns = False

    if not script_args.dataset_path:
        raise ValueError(
            "--dataset_path is required. Pass a local HF dataset disk path "
            "(from scripts/gen_teacher_reasoning.py) containing a teacher_reasoning column."
        )

    if not training_args.teacher_model_name_or_path:
        raise ValueError(
            "--teacher_model_name_or_path is required for SCOPE "
            "(frozen separate teacher for OPD+ST)."
        )

    if script_args.log_with not in {"wandb", "swanlab", "both", "none"}:
        raise ValueError(
            f"--log_with must be 'wandb', 'swanlab', 'both', or 'none', got {script_args.log_with!r}"
        )

    ################
    # Run Name & Output Directory
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    run_config = script_args.run_config or "SCOPE"
    run_name = f"{run_config}_lr{lr_str}_bs{effective_batch_size}"
    if not training_args.output_dir.endswith(run_config):
        training_args.output_dir = str(Path(training_args.output_dir) / run_config)

    if getattr(training_args, "seed", None) is not None:
        run_name += f"_seed{training_args.seed}"

    training_args.run_name = run_name
    training_args.report_to = resolve_report_to(script_args.log_with)

    experiment_config = {
        "method": "SCOPE",
        "model_name": model_args.model_name_or_path,
        "teacher_model_name_or_path": training_args.teacher_model_name_or_path,
        "learning_rate": training_args.learning_rate,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "num_train_epochs": training_args.num_train_epochs,
        "max_completion_length": training_args.max_completion_length,
        "temperature": training_args.temperature,
        "beta": training_args.beta,
        "lmbda": training_args.lmbda,
        "max_length": training_args.max_length,
        "use_peft": model_args.use_peft,
        "lora_r": model_args.lora_r if model_args.use_peft else None,
        "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "num_processes": num_processes,
        "fixed_teacher": script_args.fixed_teacher,
        "offline_teacher_reasoning": True,
        "opd_rc_online": True,
        "distillation_loss": "wd",
        "dataset_path": script_args.dataset_path,
        "teacher_reasoning_column": script_args.teacher_reasoning_column,
        "lambda_wd": script_args.lambda_wd,
        "wd_topk": script_args.wd_topk,
        "wd_epsilon": script_args.wd_epsilon,
        "wd_sinkhorn_iters": script_args.wd_sinkhorn_iters,
        "wd_interval": script_args.wd_interval,
        "seed": training_args.seed,
        "data_seed": training_args.data_seed,
        "use_ema_teacher": script_args.use_ema_teacher,
        "ema_decay": script_args.ema_decay if script_args.use_ema_teacher else None,
        "student_thinking": script_args.student_thinking,
        "teacher_thinking": script_args.teacher_thinking,
        "log_with": script_args.log_with,
        "log_traces": script_args.log_traces,
        "log_traces_steps": script_args.log_traces_steps,
        "num_traces_per_log": script_args.num_traces_per_log,
    }

    print(f"\n{'='*80}")
    print("SCOPE RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run Name: {run_name}")
    print(f"Logging Backend: {script_args.log_with}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    ################
    # Validation
    ################
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. "
            "The fixed teacher is implemented by disabling LoRA adapters."
        )

    if script_args.use_ema_teacher and training_args.teacher_model_name_or_path:
        raise ValueError(
            "use_ema_teacher=True is incompatible with --teacher_model_name_or_path. "
            "Use one teacher strategy only."
        )

    if training_args.use_vllm:
        raise ValueError(
            "SCOPE online OPD+RC is not compatible with use_vllm yet. "
            "Remove --use_vllm for training."
        )

    init_experiment_logging(
        log_with=script_args.log_with,
        run_name=run_name,
        config=experiment_config,
        wandb_entity=training_args.wandb_entity,
        wandb_project=training_args.wandb_project or "SCOPE",
        swanlab_project=script_args.swanlab_project,
        swanlab_workspace=script_args.swanlab_workspace,
        is_main_process=os.environ.get("LOCAL_RANK", "0") == "0",
    )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    teacher_model_init_kwargs = dict(model_kwargs)
    teacher_model_init_kwargs.pop("device_map", None)
    teacher_model_init_kwargs.pop("quantization_config", None)
    training_args.teacher_model_init_kwargs = teacher_model_init_kwargs
    print(f"\n{'='*80}")
    print(f"Dual-model mode: student={model_args.model_name_or_path}")
    print(f"                 teacher={training_args.teacher_model_name_or_path}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    training_args.presence_penalty = script_args.presence_penalty

    dataset_path = Path(script_args.dataset_path)
    if dataset_path.exists() and (dataset_path / "dataset_info.json").exists():
        train_dataset = load_from_disk(str(dataset_path))
    else:
        dataset = load_dataset(script_args.dataset_path)
        train_dataset = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    def _normalize_columns(example):
        out = dict(example)
        if "problem" not in out and "question" in out:
            out["problem"] = out["question"]
        if "solution" not in out and "answer" in out:
            out["solution"] = out["answer"]
        return out

    train_dataset = train_dataset.map(_normalize_columns)
    if "_scope_index" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("_scope_index")
    train_dataset = train_dataset.map(
        lambda _example, index: {"_scope_index": index}, with_indices=True
    )

    col = script_args.teacher_reasoning_column
    if col not in train_dataset.column_names:
        raise ValueError(
            f"Column '{col}' not found in dataset. Run scripts/gen_teacher_reasoning.py first. "
            f"Available columns: {train_dataset.column_names}"
        )

    ################
    # Training
    ################
    trainer = SCOPETrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        fixed_teacher=script_args.fixed_teacher,
        teacher_reasoning_column=script_args.teacher_reasoning_column,
        opd_rc_max_retries=script_args.opd_rc_max_retries,
        opd_rc_min_len_ratio=script_args.opd_rc_min_len_ratio,
        opd_rc_refresh_steps=script_args.opd_rc_refresh_steps,
        lambda_wd=script_args.lambda_wd,
        wd_topk=script_args.wd_topk,
        wd_epsilon=script_args.wd_epsilon,
        wd_sinkhorn_iters=script_args.wd_sinkhorn_iters,
        wd_interval=script_args.wd_interval,
        log_traces=script_args.log_traces,
        log_traces_steps=script_args.log_traces_steps,
        num_traces_per_log=script_args.num_traces_per_log,
        generation_debug=script_args.generation_debug,
        save_generations_local=script_args.save_generations_local,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        student_thinking=script_args.student_thinking,
        teacher_thinking=script_args.teacher_thinking,
        teacher_on_cpu=script_args.teacher_on_cpu,
        teacher_device=script_args.teacher_device,
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_completion_length,
            do_sample=True,
            temperature=training_args.temperature,
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
