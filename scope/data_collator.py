"""SCOPE data collator: problem-only student prompts + online OPD+RC prompts.

Teacher privileged context is filled later in the trainer after OPD+RC.
"""

import torch

from scope.prompts import (
    build_student_on_policy_user_message,
    build_opd_rc_user_message,
)
from scope.opd_rc_utils import strip_think_blocks


class SCOPEDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        teacher_reasoning_column: str = "teacher_reasoning",
        student_thinking: bool = False,
        teacher_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.teacher_reasoning_column = teacher_reasoning_column
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

        print("[SCOPEDataCollator] mode: offline teacher_reasoning + online OPD+RC")
        print(f"[SCOPEDataCollator] teacher_reasoning_column: {self.teacher_reasoning_column}")
        print(f"[SCOPEDataCollator] student_thinking: {self.student_thinking}")
        print(f"[SCOPEDataCollator] teacher_thinking: {self.teacher_thinking}")

    def __call__(self, features):
        student_prompts = []
        opd_rc_prompts = []
        problems = []
        solutions = []
        teacher_reasoning_texts = []

        for feature in features:
            problem = feature.get("problem") or feature.get("question")
            solution = feature.get("solution") or feature.get("answer")
            if problem is None or solution is None:
                raise KeyError(f"Missing problem/solution in feature keys: {list(feature.keys())}")

            teacher_reasoning = strip_think_blocks(feature.get(self.teacher_reasoning_column, ""))
            if not teacher_reasoning:
                raise ValueError(
                    f"Empty '{self.teacher_reasoning_column}'. "
                    "Run gen_teacher_reasoning.py first."
                )

            problems.append(problem)
            solutions.append(solution)
            teacher_reasoning_texts.append(teacher_reasoning)

            student_messages = [
                {"role": "user", "content": build_student_on_policy_user_message(problem)}
            ]
            student_prompts.append(
                self.tokenizer.apply_chat_template(
                    student_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.student_thinking,
                )
            )

            rewrite_messages = [
                {
                    "role": "user",
                    "content": build_opd_rc_user_message(problem, teacher_reasoning),
                }
            ]
            opd_rc_prompts.append(
                self.tokenizer.apply_chat_template(
                    rewrite_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.student_thinking,
                )
            )

        student_encoded_no_pad = self.tokenizer(
            student_prompts, padding=False, truncation=True, max_length=self.max_length
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        rewrite_encoded_no_pad = self.tokenizer(
            opd_rc_prompts, padding=False, truncation=True, max_length=self.max_length
        )
        rewrite_prompt_lengths = [len(ids) for ids in rewrite_encoded_no_pad["input_ids"]]
        max_rewrite_prompt_len = max(rewrite_prompt_lengths)
        rewrite_encoded = self.tokenizer(
            opd_rc_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_rewrite_prompt_len,
            return_tensors="pt",
        )

        return {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
            "opd_rc_prompts": rewrite_encoded["input_ids"],
            "opd_rc_attention_mask": rewrite_encoded["attention_mask"],
            "opd_rc_prompt_length": max_rewrite_prompt_len,
            "problems": problems,
            "solutions": solutions,
            "teacher_reasoning_texts": teacher_reasoning_texts,
        }
