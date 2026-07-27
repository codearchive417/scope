"""Training prompts for SCOPE.

Offline teacher trajectories + online OPD+RC + teacher transition.
Evaluation prompts are not included.
"""

REASON_FIRST_PROMPT = (
    "\n\nThe reference reasoning above arrives at the correct answer. "
    "Please reconstruct it into a clear, correct, self-contained step-by-step solution. "
    "Preserve all essential mathematical steps and keep the same final numerical answer. "
    "Do not merely summarize or analyze the reference reasoning. "
    "Write the solution as if solving the problem directly. "
    "Do not mention the reference reasoning. "
    "Do NOT use <think> tags. "
    "Put the final answer within \\boxed{}.\n"
)

OPD_RC_PROMPT = (
    "\n\nRewrite the reference solution into a short, natural solution that a smaller student model "
    "could realistically produce when solving the problem on its own.\n\n"
    "Keep the main mathematical idea, the necessary computations, and the same final answer.\n"
    "Do not preserve every sentence or every intermediate detail from the reference.\n"
    "Use simple equations and concise explanations.\n"
    "Use only information stated in the problem.\n"
    "Make sure each equation matches the problem conditions.\n"
    "Avoid markdown headings, decorative separators, and long teacher-like templates.\n"
    "Do not mention the reference solution, rewriting, teacher, or student.\n"
    "Put the final answer within \\boxed{}.\n"
)

OFFLINE_TRANSITION_PROMPT = (
    "\n\nUsing the reasoning above as privileged guidance, solve the problem directly and coherently. "
    "Preserve the same mathematical logic and final answer. "
    "Do not mention the reference solution or that you were given one. "
    "Do not copy the text verbatim, but keep the reasoning faithful. "
    "Do NOT use <think> tags. "
    "Put the final answer within \\boxed{}.\n"
)

FALLBACK_NO_GOLD = (
    "The original numerical answer is unavailable, so this rewrite cannot be verified. "
    "Please rely on the reference reasoning above and avoid changing the final answer."
)


def fallback_answer_mismatch(gold_answer: str) -> str:
    return (
        "The attempted rewritten reasoning did not pass the numerical answer check. "
        f"The original final numerical answer should be {gold_answer}. "
        "Please rely on the reference reasoning above and preserve this final answer. "
        "Do not introduce a different final answer."
    )


def build_offline_teacher_user_message(problem: str, solution: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"Here is a correct reasoning to this problem:"
        f"=== Reference Reasoning Start ===\n"
        f"{solution}\n"
        f"=== Reference Reasoning End ===\n"
        f"{REASON_FIRST_PROMPT}"
    )


def build_student_on_policy_user_message(problem: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"Please reason step by step, and put your final answer within \\boxed{{}}."
    )


def build_opd_rc_user_message(problem: str, teacher_reasoning: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"=== Reference Reasoning Start ===\n"
        f"{teacher_reasoning}\n"
        f"=== Reference Reasoning End ===\n"
        f"{OPD_RC_PROMPT}"
    )


def build_teacher_privileged_user_message(problem: str, rewritten_reasoning: str) -> str:
    return (
        f"Problem: {problem}\n\n"
        f"=== Reasoning Start ===\n"
        f"{rewritten_reasoning}\n"
        f"=== Reasoning End ===\n"
        f"{OFFLINE_TRANSITION_PROMPT}"
    )
