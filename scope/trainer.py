# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.integrations.integration_utils import is_wandb_available
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
from trl.extras.profiling import profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import (
    DataCollatorForChatML,
    create_model_from_path,
    disable_dropout_in_model,
    empty_cache,
    ensure_master_addr_port,
    pad,
)
from trl.experimental.gold.gold_config import GOLDConfig
from scope.data_collator import SCOPEDataCollator
from scope.experiment_logging import normalize_report_to
from scope.opd_st_loss import compute_opd_st_loss
from scope.prompts import (
    build_opd_rc_user_message,
    build_teacher_privileged_user_message,
)
from scope.opd_rc_utils import (
    get_fallback_text,
    get_len_ratio_fallback_text,
    strip_think_blocks,
    verify_opd_rc_answer,
    verify_opd_rc_length_ratio,
)
from scope.training_trace import log_training_traces_to_backends


if is_peft_available():
    from peft import PeftConfig

if is_wandb_available():
    import wandb

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


class EMAUpdateCallback(TrainerCallback):
    """每次 optimizer step 结束后，用 Student 权重更新 EMA Teacher。"""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync student weights into the vLLM engine after each training step (TRL GOLD)."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class SCOPETrainer(SFTTrainer):
    """
    SCOPE trainer: offline 8B teacher_reasoning + online OPD+RC + frozen teacher + OPD+ST.

    Inherits TRL SFTTrainer infrastructure (model load, LoRA, distributed, optimizer, checkpoint).
    Main loop:
      - Student sees problem-only prompts; teacher sees privileged OPD+RC context
      - Each step: online OPD+RC → on-policy student completion → OPD+ST distillation
      - Dual-model mode with --teacher_model_name_or_path (frozen 8B teacher)
      - Optional fixed_teacher (LoRA-off) / EMA teacher for self-distill setups
    """

    _tag_names = ["trl", "scope"]
    _name = "SCOPE"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        teacher_model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        fixed_teacher: bool = False,
        teacher_reasoning_column: str = "teacher_reasoning",
        opd_rc_max_retries: int = 1,
        opd_rc_min_len_ratio: float | None = None,
        opd_rc_refresh_steps: int = 1,
        lambda_wd: float = 0.1,
        wd_topk: int = 32,
        wd_epsilon: float = 0.05,
        wd_sinkhorn_iters: int = 20,
        wd_interval: int = 4,
        log_traces: bool = True,
        log_traces_steps: int = 10,
        num_traces_per_log: int = 1,
        generation_debug: bool = False,
        save_generations_local: bool | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        student_thinking: bool = False,
        teacher_thinking: bool = False,
        teacher_on_cpu: bool = False,
        teacher_device: str | None = None,
    ):
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        teacher_model_path = self._resolve_teacher_model_path(teacher_model, args)
        if teacher_model_path and use_ema_teacher:
            raise ValueError(
                "use_ema_teacher=True is incompatible with a separate teacher model. "
                "Unset --teacher_model_name_or_path or disable --use_ema_teacher."
            )

        loaded_teacher_model = self._load_teacher_model_if_needed(
            teacher_model=teacher_model,
            teacher_model_path=teacher_model_path,
            args=args,
        )
        self.use_separate_teacher_model = loaded_teacher_model is not None
        self.teacher_on_cpu = teacher_on_cpu
        self.teacher_device = teacher_device

        # SCOPE collator: problem-only student prompts + OPD+RC prompts; teacher context filled after OPD+RC
        if data_collator is None:
            data_collator = SCOPEDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                teacher_reasoning_column=teacher_reasoning_column,
                calibration_dataset=train_dataset,
                student_thinking=student_thinking,
                teacher_thinking=teacher_thinking,
            )

        # Hand off model/LoRA/DataLoader/optimizer setup to SFTTrainer
        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.teacher_model = None
        if loaded_teacher_model is not None:
            self._setup_separate_teacher_model(loaded_teacher_model)
        else:
            self.teacher_on_cpu = False

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.fixed_teacher = fixed_teacher
        # SCOPE main method (hardcoded)
        self.offline_teacher_reasoning = True
        self.opd_rc_online = True
        self.distillation_loss = "wd"
        self.teacher_reasoning_column = teacher_reasoning_column
        self.opd_rc_max_retries = opd_rc_max_retries
        self.opd_rc_min_len_ratio = opd_rc_min_len_ratio
        self.opd_rc_refresh_steps = max(1, opd_rc_refresh_steps)
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking
        self.lambda_wd = lambda_wd
        self.wd_topk = wd_topk
        self.wd_epsilon = wd_epsilon
        self.wd_sinkhorn_iters = wd_sinkhorn_iters
        self.wd_interval = wd_interval
        self.log_traces = log_traces
        self.log_traces_steps = max(1, log_traces_steps)
        self.generation_debug = generation_debug
        self.save_generations_local = log_traces if save_generations_local is None else save_generations_local
        self.num_traces_per_log = max(1, num_traces_per_log)
        self._last_opd_st_metrics: dict[str, float] = {}
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self._ema_params = None  # lazy-init on first optimizer step

        # --- Teacher strategy (choose one) ---
        # fixed_teacher: 关闭 LoRA，Teacher = step-0 初始 base 模型（主实验默认）
        # use_ema_teacher: Teacher = Student 权重的指数滑动平均
        # 默认 dynamic: Teacher = 当前 Student 权重（易不稳定，不推荐）

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.use_separate_teacher_model and self.fixed_teacher:
            print(f"\n{'='*80}")
            print("SEPARATE TEACHER MODEL ENABLED")
            print("Teacher forward uses the loaded teacher model (frozen).")
            print("--fixed_teacher applies only to the student model / legacy self-distill paths.")
            print(f"{'='*80}\n")

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        print(f"\n{'='*80}")
        print("SCOPE DISTILLATION LOSS: OPD+ST")
        print(
            f"OPD+ST: lambda_wd={self.lambda_wd}, wd_topk={self.wd_topk}, "
            f"wd_epsilon={self.wd_epsilon}, wd_sinkhorn_iters={self.wd_sinkhorn_iters}, "
            f"wd_interval={self.wd_interval}"
        )
        print(f"{'='*80}\n")

        print(f"\n{'='*80}")
        print("SCOPE MODE ENABLED")
        print(f"Offline teacher_reasoning column: {self.teacher_reasoning_column}")
        print("Online OPD+RC: True")
        print(f"OPD+RC refresh every {self.opd_rc_refresh_steps} optimizer step(s)")
        print(f"{'='*80}\n")

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs and full training traces for saving / logging
        self._generation_outputs_buffer = []
        self._trace_logs_buffer: list[dict] = []
        self._generation_save_frequency = 5 if self.save_generations_local else 0

        # Student on-policy 采样配置（每步用当前策略生成 completion）
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for online OPD+RC
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        # Optional vLLM path for on-policy generation; sync weights via GOLDVLLMSyncCallback
        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    @staticmethod
    def _resolve_teacher_model_path(
        teacher_model: PreTrainedModel | nn.Module | str | None,
        args: GOLDConfig | None,
    ) -> str | None:
        if isinstance(teacher_model, str):
            return teacher_model
        if args is not None and args.teacher_model_name_or_path:
            return args.teacher_model_name_or_path
        return None

    @staticmethod
    def _load_teacher_model_if_needed(
        teacher_model: PreTrainedModel | nn.Module | str | None,
        teacher_model_path: str | None,
        args: GOLDConfig | None,
    ) -> PreTrainedModel | nn.Module | None:
        if teacher_model is not None and not isinstance(teacher_model, str):
            return teacher_model
        if teacher_model_path is None:
            return None

        if args is None:
            raise ValueError("GOLDConfig (training args) is required to load a separate teacher model.")

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs: dict[str, Any] = {}
        elif not isinstance(teacher_model, str) and teacher_model is not None:
            raise ValueError(
                "teacher_model_init_kwargs was provided, but teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = dict(args.teacher_model_init_kwargs)
            torch_dtype = teacher_model_init_kwargs.get("torch_dtype")
            if torch_dtype not in {"auto", None} and isinstance(torch_dtype, str):
                teacher_model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        init_kwargs = dict(teacher_model_init_kwargs)
        if "torch_dtype" in init_kwargs and "dtype" not in init_kwargs:
            init_kwargs["dtype"] = init_kwargs.pop("torch_dtype")
        # Load teacher on CPU first; student loads on GPU in super().__init__, then accelerator.prepare
        # moves teacher. Avoids OOM from loading 8B onto GPU before 1.7B student.
        init_kwargs["device_map"] = "cpu"
        init_kwargs.setdefault("low_cpu_mem_usage", True)

        print(f"\n{'='*80}")
        print("Loading separate teacher model")
        print(f"Teacher path: {teacher_model_path}")
        print(f"Teacher init kwargs: {init_kwargs}")
        print(f"{'='*80}\n")

        return create_model_from_path(teacher_model_path, **init_kwargs)

    def _setup_separate_teacher_model(self, teacher_model: PreTrainedModel | nn.Module) -> None:
        disable_dropout_in_model(teacher_model)
        teacher_model.config.use_cache = False

        student_vocab = self.model.config.vocab_size
        teacher_vocab = teacher_model.config.vocab_size
        if teacher_vocab != student_vocab:
            print(
                f"Resizing teacher token embeddings: {teacher_vocab} -> {student_vocab} "
                "to match the student model."
            )
            teacher_model.resize_token_embeddings(student_vocab)

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        if self.teacher_on_cpu:
            if hasattr(teacher_model, "set_attn_implementation"):
                teacher_model.set_attn_implementation("sdpa")
            else:
                teacher_model.config._attn_implementation = "sdpa"
            self.teacher_model = teacher_model.to("cpu")
            print(
                "\n[INFO] Teacher kept on CPU (--teacher_on_cpu, attn=sdpa). "
                "Student forward stays on GPU.\n"
            )
        elif self.teacher_device:
            self.teacher_model = teacher_model.to(self.teacher_device)
            print(f"\n[INFO] Teacher pinned to {self.teacher_device}\n")
        else:
            try:
                if self.is_deepspeed_enabled:
                    self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
                else:
                    self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                empty_cache()
                self.teacher_on_cpu = True
                if hasattr(teacher_model, "set_attn_implementation"):
                    teacher_model.set_attn_implementation("sdpa")
                else:
                    teacher_model.config._attn_implementation = "sdpa"
                self.teacher_model = teacher_model.to("cpu")
                print(
                    "\n[WARN] CUDA OOM while placing teacher on GPU; keeping teacher on CPU. "
                    "Forward will be slower but training can continue.\n"
                )

        teacher_name = getattr(self.teacher_model.config, "_name_or_path", "teacher")
        num_params = sum(p.numel() for p in self.teacher_model.parameters())
        teacher_device = self._teacher_device()
        print(f"\n{'='*80}")
        print("SEPARATE TEACHER MODEL READY")
        print(f"Teacher: {teacher_name}")
        print(f"Parameters: {num_params:,} (frozen, eval mode)")
        print(f"Teacher device: {teacher_device}")
        print(f"Student: {self.model_name_or_path}")
        print(f"{'='*80}\n")

    def _teacher_device(self):
        if not self.use_separate_teacher_model:
            return self.accelerator.device
        return next(self.teacher_model.parameters()).device

    def _prepare_dataset(self, dataset, *args, **kwargs):
        # Keep raw problem/solution rows for SCOPEDataCollator.
        # Skip SFTTrainer's default text-field preprocessing.
        return dataset

    def _set_signature_columns_if_needed(self):
        # Keep raw problem/solution/teacher_reasoning for SCOPEDataCollator
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
            self.teacher_reasoning_column,
            "_scope_index",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    def _update_ema(self):
        """EMA teacher weight update: ema = decay * ema + (1 - decay) * student.

        首次调用时复制当前 Student 权重作为初始 EMA，不做 decay。

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Temporarily swap in EMA weights for teacher forward, then restore student weights.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Dual-context distillation:
          1. Student forward on [problem prompt][on-policy completion]
          2. Teacher forward (no_grad) on [privileged OPD+RC context][same completion]
          3. OPD+ST on completion tokens only
        """
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === Student forward (with grad) ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_logits_for_loss = student_logits
        del student_logits

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        # === Teacher forward (no grad) ===
        if self.use_separate_teacher_model:
            self.teacher_model.eval()
            teacher_device = self._teacher_device()
            student_device = inputs["student_input_ids"].device
            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["teacher_input_ids"].to(teacher_device),
                    attention_mask=inputs["teacher_attention_mask"].to(teacher_device),
                )
                teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :].to(student_device)
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

                del outputs_teacher
            empty_cache()
        else:
            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            with torch.no_grad(), adapter_context:
                outputs_teacher = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

                del outputs_teacher
            empty_cache()

        # === Distillation loss (OPD+ST) ===
        loss_mask = shifted_labels != -100 if shifted_labels is not None else None
        if loss_mask is None:
            loss_mask = torch.ones(
                student_logits_for_loss.shape[:2],
                dtype=torch.bool,
                device=student_logits_for_loss.device,
            )

        unwrapped_model = self.accelerator.unwrap_model(model)
        embedding_weight = unwrapped_model.get_input_embeddings().weight

        loss_dict = compute_opd_st_loss(
            student_logits=student_logits_for_loss,
            teacher_logits=teacher_logits_for_loss,
            loss_mask=loss_mask,
            embedding_weight=embedding_weight,
            temperature=self.temperature,
            topk=self.wd_topk,
            lambda_wd=self.lambda_wd,
            epsilon=self.wd_epsilon,
            sinkhorn_iters=self.wd_sinkhorn_iters,
            wd_interval=self.wd_interval,
        )
        loss = loss_dict["loss"]
        self._last_opd_st_metrics = {
            "rkl_loss": float(loss_dict["rkl_loss"]),
            "wd_loss": float(loss_dict["wd_loss"]),
            "num_wd_positions": float(loss_dict["num_wd_positions"]),
            "num_valid_loss_tokens": float(loss_dict["num_valid_loss_tokens"]),
        }
        del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_opd_rc(self, model, opd_rc_prompts, opd_rc_attention_mask=None):
        """Online OPD+RC using current student weights (with LoRA)."""
        if self.use_vllm:
            raise NotImplementedError(
                "opd_rc_online with use_vllm is not supported yet; disable --use_vllm for this mode."
            )

        with torch.no_grad():
            original_use_cache = model.config.use_cache
            original_gen_use_cache = self.reasoning_generation_config.use_cache
            model.config.use_cache = True
            self.reasoning_generation_config.use_cache = True

            try:
                rewrite_outputs = model.generate(
                    input_ids=opd_rc_prompts,
                    attention_mask=opd_rc_attention_mask,
                    generation_config=self.reasoning_generation_config,
                    return_dict_in_generate=True,
                    use_cache=True,
                )
                return rewrite_outputs.sequences
            finally:
                model.config.use_cache = original_use_cache
                self.reasoning_generation_config.use_cache = original_gen_use_cache

    def _encode_teacher_prompt_texts(self, teacher_prompt_texts: list[str]) -> dict[str, torch.Tensor]:
        teacher_messages_batch = [[{"role": "user", "content": t}] for t in teacher_prompt_texts]
        teacher_prompts = [
            self.processing_class.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=self.teacher_thinking
            )
            for m in teacher_messages_batch
        ]

        teacher_encoded_no_pad = self.processing_class(
            teacher_prompts, padding=False, truncation=True, max_length=self.args.max_length
        )
        teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
        max_teacher_prompt_len = max(teacher_prompt_lengths)

        teacher_encoded = self.processing_class(
            teacher_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_teacher_prompt_len,
            return_tensors="pt",
        )

        device = self.accelerator.device
        return {
            "teacher_prompts": teacher_encoded["input_ids"].to(device),
            "teacher_prompt_attention_mask": teacher_encoded["attention_mask"].to(device),
            "teacher_prompt_length": max_teacher_prompt_len,
            "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths, device=device),
        }

    def _opd_rc_acceptable(self, text: str, solution: str, reference_reasoning: str) -> bool:
        if not verify_opd_rc_answer(text, solution):
            return False
        if self.opd_rc_min_len_ratio is None:
            return True
        return verify_opd_rc_length_ratio(text, reference_reasoning, self.opd_rc_min_len_ratio)

    def _run_opd_rc(self, model, inputs: dict) -> tuple[list[str], list[dict]]:
        calibration_problems = inputs["calibration_problems"]
        calibration_solutions = inputs["calibration_solutions"]
        calibration_reasoning_texts = inputs["calibration_teacher_reasoning_texts"]
        batch_size = len(calibration_problems)

        rewrite_prompt_len = inputs["opd_rc_prompt_length"]
        rewrite_ids = self.generate_opd_rc(
            model,
            inputs["opd_rc_prompts"],
            inputs.get("opd_rc_attention_mask"),
        )
        rewrite_completions = rewrite_ids[:, rewrite_prompt_len:]
        opd_rc_texts = self.processing_class.batch_decode(rewrite_completions, skip_special_tokens=True)

        final_texts = []
        opd_rc_traces = []
        for i in range(batch_size):
            raw_text = opd_rc_texts[i].strip()
            text = raw_text
            solution = calibration_solutions[i]
            problem = calibration_problems[i]
            fallback_reasoning = (
                calibration_reasoning_texts[i]
                if isinstance(calibration_reasoning_texts[i], str)
                else calibration_reasoning_texts[i]
            )
            fallback_reasoning = strip_think_blocks(fallback_reasoning)

            trace = {
                "calibration_index": inputs["calibration_indices"][i],
                "calibration_problem": problem,
                "calibration_solution": solution,
                "teacher_reasoning_offline": fallback_reasoning,
                "opd_rc_raw": raw_text,
                "opd_rc_final": raw_text,
                "opd_rc_answer_ok": verify_opd_rc_answer(raw_text, solution),
                "opd_rc_len_ok": (
                    self.opd_rc_min_len_ratio is None
                    or verify_opd_rc_length_ratio(
                        raw_text, fallback_reasoning, self.opd_rc_min_len_ratio
                    )
                ),
                "opd_rc_retried": False,
                "used_fallback_reasoning": False,
            }

            if self._opd_rc_acceptable(text, solution, fallback_reasoning):
                final_texts.append(text)
                opd_rc_traces.append(trace)
                continue

            retried = False
            for _ in range(self.opd_rc_max_retries):
                trace["opd_rc_retried"] = True
                fallback_msg = build_opd_rc_user_message(
                    problem, fallback_reasoning
                )
                if not verify_opd_rc_answer(text, solution):
                    fallback_msg += "\n" + get_fallback_text(solution)
                elif self.opd_rc_min_len_ratio is not None:
                    fallback_msg += "\n" + get_len_ratio_fallback_text(self.opd_rc_min_len_ratio)
                messages = [{"role": "user", "content": fallback_msg}]
                retry_prompt = self.processing_class.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                retry_encoded = self.processing_class(retry_prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    retry_out = model.generate(
                        **retry_encoded,
                        generation_config=self.reasoning_generation_config,
                        return_dict_in_generate=True,
                        use_cache=True,
                    )
                retry_new = retry_out.sequences[0, retry_encoded["input_ids"].shape[1] :]
                text = self.processing_class.decode(retry_new, skip_special_tokens=True).strip()
                if self._opd_rc_acceptable(text, solution, fallback_reasoning):
                    retried = True
                    break

            if not retried and not self._opd_rc_acceptable(text, solution, fallback_reasoning):
                text = fallback_reasoning
                trace["used_fallback_reasoning"] = True

            trace["opd_rc_final"] = text
            trace["opd_rc_answer_ok"] = verify_opd_rc_answer(text, solution)
            trace["opd_rc_len_ok"] = (
                self.opd_rc_min_len_ratio is None
                or verify_opd_rc_length_ratio(
                    text, fallback_reasoning, self.opd_rc_min_len_ratio
                )
            )
            final_texts.append(text)
            opd_rc_traces.append(trace)

        if random.random() < 0.01:
            sample_idx = random.randint(0, batch_size - 1)
            print(f"\n{'='*80}")
            print(f"OPD+RC SAMPLE (Step {self.state.global_step}):")
            print(f"Calibration problem: {calibration_problems[sample_idx][:200]}...")
            print(f"Rewrite:\n{final_texts[sample_idx][:500]}...")
            print(f"{'='*80}\n")

        return final_texts, opd_rc_traces

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """On-policy completion from problem-only student prompts."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        if self.generation_debug:
            print(f"\n{'='*80}")
            print(f"GENERATION DEBUG INFO:")
            print(f"  Model dtype: {model.dtype}")
            print(f"  Model config use_cache: {model.config.use_cache}")
            print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
            print(f"  Generation config use_cache: {generation_config.use_cache}")
            print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
            print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
            print(f"  Max new tokens: {generation_config.max_new_tokens}")
            print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        if self.generation_debug:
            print(
                f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
            )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """vLLM on-policy generation path."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Sync trained student (including LoRA) weights into vLLM."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        """Save generation outputs and training traces to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0 and len(self._trace_logs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._trace_logs_buffer) or len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
            "traces": self._trace_logs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(
            f"Saved {len(self._trace_logs_buffer) or len(self._generation_outputs_buffer)} "
            f"generation/trace outputs to:"
        )
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()
        self._trace_logs_buffer.clear()

    def _build_training_traces(
        self,
        inputs: dict,
        prompt_texts: list[str],
        completion_texts: list[str],
        opd_rc_traces: list[dict] | None = None,
        teacher_reasoning_online: list[str] | None = None,
    ) -> list[dict]:
        problems = inputs["problems"]
        solutions = inputs["solutions"]
        teacher_distill_prompts = self.processing_class.batch_decode(
            inputs["teacher_prompts"], skip_special_tokens=False
        )
        batch_size = len(problems)
        traces = []

        for i in range(batch_size):
            trace = {
                "step": self.state.global_step,
                "problem": problems[i] if isinstance(problems[i], str) else str(problems[i]),
                "solution": solutions[i] if isinstance(solutions[i], str) else str(solutions[i]),
                "teacher_reasoning_offline": None,
                "opd_rc_raw": None,
                "opd_rc_final": None,
                "opd_rc_answer_ok": None,
                "opd_rc_retried": None,
                "used_fallback_reasoning": None,
                "teacher_reasoning_online": None,
                "teacher_distill_prompt": teacher_distill_prompts[i],
                "student_on_policy_prompt": prompt_texts[i],
                "student_on_policy_completion": completion_texts[i],
            }

            if opd_rc_traces is not None and i < len(opd_rc_traces):
                trace.update(opd_rc_traces[i])
            elif "calibration_teacher_reasoning_texts" in inputs:
                offline_reasoning = inputs["calibration_teacher_reasoning_texts"][i]
                trace["teacher_reasoning_offline"] = (
                    offline_reasoning if isinstance(offline_reasoning, str) else str(offline_reasoning)
                )

            if teacher_reasoning_online is not None and i < len(teacher_reasoning_online):
                trace["teacher_reasoning_online"] = teacher_reasoning_online[i]

            traces.append(trace)

        return traces

    def _maybe_log_training_traces(self) -> None:
        if not self.log_traces or not self.accelerator.is_main_process:
            return
        if self.state.global_step <= 0 or self.state.global_step % self.log_traces_steps != 0:
            return
        if len(self._trace_logs_buffer) == 0:
            return

        sample_size = min(self.num_traces_per_log, len(self._trace_logs_buffer))
        sampled_traces = random.sample(self._trace_logs_buffer, sample_size)
        log_training_traces_to_backends(
            sampled_traces,
            step=self.state.global_step,
            report_to=self.args.report_to,
        )

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Per-step SCOPE loop:
          1. Sample a disjoint calibration exemplar and run online OPD+RC
          2. Prepend the calibration problem--rewrite pair to the target teacher context
          3. Student on-policy completion
          4. Dual-sequence pack + OPD+ST via super().training_step
        """
        on_policy = True
        opd_rc_traces = None
        teacher_reasoning_online = None

        # === Online OPD+RC + teacher privileged context ===
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            opd_rc_texts, opd_rc_traces = self._run_opd_rc(unwrapped_model, inputs)
            inputs["opd_rc_texts"] = opd_rc_texts
            teacher_prompt_texts = [
                build_teacher_privileged_user_message(target, calibration, rewrite)
                for target, calibration, rewrite in zip(
                    inputs["problems"], inputs["calibration_problems"], opd_rc_texts
                )
            ]
            encoded = self._encode_teacher_prompt_texts(teacher_prompt_texts)
            inputs.update(encoded)

        # === Student on-policy completion ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        student_prompt_len = inputs["student_prompt_length"]
        generation_ids = generated_ids[:, student_prompt_len:]

        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        if "teacher_prompts" not in inputs:
            raise KeyError("teacher_prompts missing after online OPD+RC encoding.")
        teacher_prompts = inputs["teacher_prompts"]
        teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

        teacher_attention_mask = torch.ones_like(teacher_full_ids)
        if self.processing_class.pad_token_id is not None:
            teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

        inputs["teacher_input_ids"] = teacher_full_ids
        inputs["teacher_attention_mask"] = teacher_attention_mask

        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        if self.save_generations_local or self.log_traces:
            step_traces = self._build_training_traces(
                inputs,
                prompt_texts,
                completion_texts,
                opd_rc_traces=opd_rc_traces,
                teacher_reasoning_online=teacher_reasoning_online,
            )
            self._trace_logs_buffer.extend(step_traces)

            if self.save_generations_local:
                for prompt, completion in zip(prompt_texts, completion_texts):
                    self._generation_outputs_buffer.append(
                        {"step": self.state.global_step, "prompt": prompt, "completion": completion}
                    )

        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'='*80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

        if (
            self.save_generations_local
            and self._generation_save_frequency > 0
            and self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # Aggregate metrics then delegate logging to the parent trainer
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # on/off-policy loss buckets (SCOPE is always on-policy)
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

            if self._last_opd_st_metrics:
                logs.update(self._last_opd_st_metrics)

            self._maybe_log_training_traces()

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if (
            self.accelerator.is_main_process
            and self.log_completions
            and ((self.state.global_step % self.log_completion_steps) == 0)
        ):
            report_targets = normalize_report_to(self.args.report_to)
            import pandas as pd

            table = {
                "step": [str(self.state.global_step)] * len(self._textual_logs["prompt"]),
                "prompt": self._textual_logs["prompt"],
                "completion": self._textual_logs["completion"],
            }
            df = pd.DataFrame(table)
            if self.wandb_log_unique_prompts:
                df = df.drop_duplicates(subset=["prompt"])
            if self.num_completions_to_print and len(df) > 0:
                df = df.sample(n=self.num_completions_to_print, random_state=42)

            if "wandb" in report_targets and is_wandb_available() and wandb.run is not None:
                wandb.log({"completions": wandb.Table(dataframe=df)})

            if "swanlab" in report_targets and len(df) > 0:
                try:
                    import swanlab

                    if swanlab.get_run() is not None:
                        completion_text = "\n\n".join(
                            f"### Sample {idx + 1}\n**Prompt:**\n{row.prompt}\n\n**Completion:**\n{row.completion}"
                            for idx, row in enumerate(df.itertuples(index=False))
                        )
                        swanlab.log(
                            {"completions": swanlab.Text(completion_text, caption="on-policy samples")},
                            step=self.state.global_step,
                        )
                except ImportError:
                    pass
