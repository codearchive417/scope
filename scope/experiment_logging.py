"""Unified experiment logging setup for WandB and SwanLab."""

from __future__ import annotations

import os
from typing import Any


VALID_LOG_BACKENDS = {"wandb", "swanlab", "both", "none"}


def normalize_report_to(report_to: str | list[str] | None) -> list[str]:
    if report_to is None:
        return []
    if isinstance(report_to, str):
        return [report_to]
    return list(report_to)


def resolve_report_to(log_with: str) -> list[str]:
    if log_with == "wandb":
        return ["wandb"]
    if log_with == "swanlab":
        return ["swanlab"]
    if log_with == "both":
        return ["wandb", "swanlab"]
    return []


def prepare_swanlab_env(swanlab_project: str | None = None) -> None:
    """Configure SwanLab env vars before any `import swanlab`.

    SWANLAB_PROJECT is reserved for JSON settings in swanlab>=0.8 and must not
    hold a plain project name. Use SWANLAB_PROJ_NAME instead.
    """
    if swanlab_project:
        os.environ["SWANLAB_PROJ_NAME"] = swanlab_project
    os.environ.pop("SWANLAB_PROJECT", None)


def init_experiment_logging(
    *,
    log_with: str,
    run_name: str,
    config: dict[str, Any],
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    swanlab_project: str | None = None,
    swanlab_workspace: str | None = None,
    is_main_process: bool = True,
) -> None:
    if log_with not in VALID_LOG_BACKENDS:
        raise ValueError(f"log_with must be one of {sorted(VALID_LOG_BACKENDS)}, got {log_with!r}")

    if not is_main_process or log_with == "none":
        return

    if log_with in {"wandb", "both"}:
        import wandb

        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=run_name,
            config=config,
        )

    if log_with in {"swanlab", "both"}:
        prepare_swanlab_env(swanlab_project)
        if swanlab_workspace:
            os.environ["SWANLAB_WORKSPACE"] = swanlab_workspace

        import swanlab

        swanlab.init(
            project=swanlab_project or "SCOPE",
            workspace=swanlab_workspace,
            experiment_name=run_name,
            config=config,
        )
