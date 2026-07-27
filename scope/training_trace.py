"""Helpers for logging full SCOPE training traces to SwanLab / WandB / disk."""

from __future__ import annotations

from typing import Any


def format_training_trace_markdown(trace: dict[str, Any], index: int = 0) -> str:
    """Render one training trace as readable Markdown."""
    lines = [f"## Trace {index + 1} (step {trace.get('step', '?')})"]

    def _section(title: str, value: str | None) -> None:
        if value is None:
            return
        lines.append(f"\n### {title}\n{value}")

    _section("Problem", trace.get("problem"))
    _section("Ground Truth Solution", trace.get("solution"))
    _section("Offline Teacher Reasoning", trace.get("teacher_reasoning_offline"))

    rewrite_meta = []
    if trace.get("opd_rc_answer_ok") is not None:
        rewrite_meta.append(f"- answer_check_passed: `{trace['opd_rc_answer_ok']}`")
    if trace.get("opd_rc_retried") is not None:
        rewrite_meta.append(f"- retried: `{trace['opd_rc_retried']}`")
    if trace.get("used_fallback_reasoning") is not None:
        rewrite_meta.append(f"- used_fallback_reasoning: `{trace['used_fallback_reasoning']}`")
    if rewrite_meta:
        lines.append("\n### OPD+RC Meta\n" + "\n".join(rewrite_meta))

    _section("OPD+RC (raw)", trace.get("opd_rc_raw"))
    _section("OPD+RC (final)", trace.get("opd_rc_final"))
    _section("Teacher Distill Context", trace.get("teacher_distill_prompt"))
    _section("Student On-Policy Prompt", trace.get("student_on_policy_prompt"))
    _section("Student On-Policy Completion", trace.get("student_on_policy_completion"))

    return "\n".join(lines)


def log_training_traces_to_backends(
    traces: list[dict[str, Any]],
    *,
    step: int,
    report_to: str | list[str] | None,
) -> None:
    """Upload sampled training traces to configured experiment backends."""
    if not traces:
        return

    from scope.experiment_logging import normalize_report_to

    report_targets = normalize_report_to(report_to)
    markdown = "\n\n---\n\n".join(
        format_training_trace_markdown(trace, idx) for idx, trace in enumerate(traces)
    )

    if "wandb" in report_targets:
        try:
            from transformers.integrations.integration_utils import is_wandb_available

            if is_wandb_available():
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "training_traces": wandb.Table(
                                columns=list(traces[0].keys()),
                                data=[[trace.get(col) for col in traces[0].keys()] for trace in traces],
                            ),
                        },
                        step=step,
                    )
                    wandb.log({"training_traces_markdown": markdown}, step=step)
        except ImportError:
            pass

    if "swanlab" in report_targets:
        try:
            import swanlab

            if swanlab.get_run() is not None:
                swanlab.log(
                    {"training_traces": swanlab.Text(markdown, caption="SCOPE training trace")},
                    step=step,
                )
        except ImportError:
            pass
