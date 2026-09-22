"""Optional W&B configuration and checkpoint history, without a metric writer."""

import copy
import os
from typing import TYPE_CHECKING, Literal, Mapping

from exex import execution

if TYPE_CHECKING:
    from wandb import Run


def configure_wandb(
    project: str,
    entity: str,
    group: str = "{title}_{xid}_{wid}",
    mode: str = "online",
):
    """Wrap a Job/ArrayJob with W&B settings, preserving its other fields.

    Configured values override matching job environment entries. Application
    calls still own initialization. No run IDs or credentials are set.
    """
    from exex import xm
    from exex import xm_cluster

    def wrap(job):
        async def job_gen(work_unit):
            title = work_unit.experiment._experiment_title
            xid, wid = work_unit.experiment_id, work_unit.work_unit_id
            name = f"{title}_{xid}_{wid}"
            common = {
                "WANDB_PROJECT": project,
                "WANDB_ENTITY": entity,
                "WANDB_MODE": mode,
                "WANDB_RUN_GROUP": group.format(title=title, xid=xid, wid=wid),
            }
            wrapped = copy.copy(job)
            if isinstance(job, xm.Job):
                wrapped.env_vars = {**job.env_vars, **common, "WANDB_NAME": name}
            elif isinstance(job, xm_cluster.ArrayJob):
                wrapped.env_vars = [
                    {**env, **common, "WANDB_NAME": f"{name}_{task + 1}"}
                    for task, env in enumerate(job.env_vars)
                ]
            else:
                raise TypeError(f"Unsupported job type: {type(job)}")
            return await work_unit.add(wrapped)

        return job_gen

    return wrap


def init(
    *,
    state: Mapping | None = None,
    history: Literal["new", "append", "fork"] = "new",
    **wandb_kwargs,
) -> "Run":
    """Initialize a native W&B run and attach its URL to the executing task.

    new records parent metadata without inheriting history. append resumes the
    saved ID without truncation. fork inherits through the saved last row and
    requires W&B permission. Errors never trigger fallback.
    """
    import wandb

    if history not in {"new", "append", "fork"}:
        raise ValueError("history must be 'new', 'append', or 'fork'")
    settings = wandb_kwargs.get("settings") or {}
    settings = (
        settings
        if isinstance(settings, dict)
        else settings.model_dump(exclude_unset=True)
    )
    policy_keys = {"resume", "resume_from", "fork_from", "reinit"}
    configured = {**settings, **wandb_kwargs}
    if any(configured.get(key) not in (None, "default") for key in policy_keys) or any(
        os.environ.get(f"WANDB_{key.upper()}") for key in policy_keys
    ):
        raise ValueError(
            "Select history with history=, not resume/fork/reinit settings"
        )
    mode = (
        wandb_kwargs.get("mode")
        or settings.get("mode")
        or os.environ.get("WANDB_MODE")
        or wandb.setup().settings.mode
    )
    if history != "new" and (state is None or mode != "online"):
        raise ValueError("append/fork require saved W&B state and online mode")

    kwargs = dict(wandb_kwargs)
    kwargs.update(
        id=kwargs.get("id") or wandb.util.generate_id(), reinit="create_new", mode=mode
    )
    if state is not None:
        for key, setting, env in (
            ("entity", "entity", "WANDB_ENTITY"),
            ("project", "project", "WANDB_PROJECT"),
            ("group", "run_group", "WANDB_RUN_GROUP"),
        ):
            kwargs.setdefault(
                key, settings.get(setting, os.environ.get(env, state.get(key)))
            )
    if history == "append":
        kwargs.update(
            id=state["run_id"],
            resume="must",
            entity=state["entity"],
            project=state["project"],
        )
    elif history == "fork":
        if state["next_step"] <= 0:
            raise ValueError("Cannot fork before a committed W&B history row")
        kwargs.update(
            fork_from=f"{state['run_id']}?_step={state['next_step'] - 1}",
            entity=state["entity"],
            project=state["project"],
        )
    elif mode == "online":
        kwargs["resume"] = "never"
    run = wandb.init(**kwargs)
    if state is not None and not run.disabled:
        origin = "exex/resumed_from" if history == "append" else "exex/parent"
        run.config.update(
            {origin: dict(state), "exex/history": history}, allow_val_change=True
        )
    attach(run)
    return run


def checkpoint_state(run: "Run") -> dict | None:
    """Capture actual identity and the next history row; None when disabled.

    Commit pending metrics before calling. This neither logs an extra row nor
    waits for server upload, and is not a durability barrier.
    """
    if run.disabled:
        return None
    return dict(
        entity=run.entity,
        project=run.project,
        group=run.group,
        run_id=run.id,
        # Before the first new log, W&B can report step=0 after append/fork.
        next_step=max(run.step, run.starting_step),
    )


def attach(run: "Run") -> None:
    """Report an existing run's actual URL without taking over its lifecycle."""
    if not run.disabled and run.url:
        execution.link("wandb", run.url)
