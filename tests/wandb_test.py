"""W&B policy translation is offline-testable; fork permissions are never assumed."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex.contrib import wandb as integration

STATE = dict(
    entity="team", project="science", group="logical-run", run_id="parent", next_step=8
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    for key in os.environ:
        if key.startswith("WANDB_") or key == "EXEX_LINKS_FILE":
            monkeypatch.delenv(key)
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def backend(monkeypatch):
    run = SimpleNamespace(
        disabled=False,
        url="https://wandb.ai/team/science/runs/child",
        entity="team",
        project="science",
        group="logical-run",
        id="child",
        step=8,
        starting_step=0,
        config=mock.Mock(),
    )
    sdk = SimpleNamespace(
        init=mock.Mock(return_value=run),
        setup=mock.Mock(
            return_value=SimpleNamespace(settings=SimpleNamespace(mode="online"))
        ),
        util=SimpleNamespace(generate_id=lambda: "child"),
    )
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk


@pytest.mark.parametrize("array", [False, True])
def test_launcher_wrapper_preserves_fields_and_environment(array):
    bundle = xc.AppBundle("worker", "true", "/source.zip")
    executor = xc.Local()
    env = {"CUSTOM": "kept", "WANDB_PROJECT": "replaced", "WANDB_RUN_ID": "explicit"}
    job = (
        xc.ArrayJob(
            bundle, executor, name="original", args=[["a"], ["b"]], env_vars=[env]
        )
        if array
        else xm.Job(bundle, executor, name="original", args=["a"], env_vars=env)
    )
    job.extra = "preserved"
    unit = SimpleNamespace(
        experiment=SimpleNamespace(_experiment_title="trial"),
        experiment_id=12,
        work_unit_id=3,
        add=mock.AsyncMock(),
    )
    wrapper = integration.configure_wandb("science", "team", group="{title}_{xid}")
    asyncio.run(wrapper(job)(unit))
    actual = unit.add.call_args.args[0]
    assert actual is not job
    assert actual.name == "original" and actual.extra == "preserved"
    assert actual.executable is bundle and actual.executor is executor
    assert actual.args == job.args
    environments = actual.env_vars if array else [actual.env_vars]
    for index, value in enumerate(environments):
        assert value == {
            **env,
            "WANDB_PROJECT": "science",
            "WANDB_ENTITY": "team",
            "WANDB_MODE": "online",
            "WANDB_RUN_GROUP": "trial_12",
            "WANDB_NAME": "trial_12_3" + (f"_{index + 1}" if array else ""),
        }
    assert env["WANDB_PROJECT"] == "replaced"


@pytest.mark.parametrize("history", ["new", "append", "fork"])
def test_history_translation_uses_actual_saved_state(
    backend, history, monkeypatch, tmp_path
):
    path = tmp_path / "links.json"
    monkeypatch.setenv("EXEX_LINKS_FILE", str(path))
    run = integration.init(state=STATE, history=history, config={"lr": 0.1})
    assert run is backend.init.return_value
    kwargs = backend.init.call_args.kwargs
    assert kwargs["config"] == {"lr": 0.1}
    assert kwargs["entity"] == "team" and kwargs["project"] == "science"
    assert kwargs["group"] == "logical-run" and kwargs["reinit"] == "create_new"
    if history == "append":
        assert kwargs["id"] == "parent" and kwargs["resume"] == "must"
        assert "fork_from" not in kwargs
    elif history == "fork":
        assert kwargs["id"] == "child" and kwargs["fork_from"] == "parent?_step=7"
        assert "resume" not in kwargs
    else:
        assert kwargs["id"] == "child" and kwargs["resume"] == "never"
        assert "fork_from" not in kwargs
    assert "resume_from" not in kwargs
    origin = "exex/resumed_from" if history == "append" else "exex/parent"
    run.config.update.assert_called_once_with(
        {origin: STATE, "exex/history": history}, allow_val_change=True
    )
    assert json.loads(path.read_text()) == {"wandb": run.url}


def test_fork_denial_is_not_retried_or_replaced(backend):
    failure = RuntimeError("org does not have permission")
    backend.init.side_effect = failure
    with pytest.raises(RuntimeError) as error:
        integration.init(state=STATE, history="fork")
    assert error.value is failure
    assert backend.init.call_count == 1


@pytest.mark.parametrize("history", ["append", "fork"])
def test_missing_state_and_offline_history_are_not_silently_downgraded(
    backend, history
):
    with pytest.raises(ValueError, match="saved W&B state"):
        integration.init(history=history)
    with pytest.raises(ValueError, match="online"):
        integration.init(state=STATE, history=history, mode="offline")
    backend.setup.return_value.settings.mode = "offline"
    with pytest.raises(ValueError, match="online"):
        integration.init(state=STATE, history=history)
    backend.init.assert_not_called()


def test_empty_history_cannot_fork(backend):
    with pytest.raises(ValueError, match="committed"):
        integration.init(state={**STATE, "next_step": 0}, history="fork")
    backend.init.assert_not_called()


def test_rewind_is_not_supported(backend):
    with pytest.raises(ValueError, match="history must"):
        integration.init(state=STATE, history="rewind")
    backend.init.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("resume", "must"),
        ("resume_from", "old?_step=7"),
        ("fork_from", "old?_step=7"),
        ("reinit", "return_previous"),
    ],
)
def test_ambient_or_native_history_cannot_override_policy(
    backend, monkeypatch, key, value
):
    with pytest.raises(ValueError, match="Select history"):
        integration.init(**{key: value})
    with pytest.raises(ValueError, match="Select history"):
        integration.init(settings={key: value})
    monkeypatch.setenv("WANDB_" + key.upper(), value)
    with pytest.raises(ValueError, match="Select history"):
        integration.init()
    backend.init.assert_not_called()


@pytest.mark.parametrize(
    "step,starting_step,next_step",
    [(0, 0, 0), (0, 7, 7), (0, 2, 2), (9, 7, 9), (4, 2, 4)],
    ids=["fresh", "empty-append", "empty-fork", "logged-append", "logged-fork"],
)
def test_checkpoint_does_not_log_or_invent_state(
    backend, step, starting_step, next_step
):
    run = backend.init.return_value
    run.step, run.starting_step = step, starting_step
    assert integration.checkpoint_state(run) == {
        **STATE,
        "run_id": "child",
        "next_step": next_step,
    }
    run.disabled = True
    assert integration.checkpoint_state(run) is None
    integration.attach(run)
    backend.init.assert_not_called()


def test_fresh_run_ignores_inherited_run_id_and_respects_new_destination(
    backend, monkeypatch
):
    monkeypatch.setenv("WANDB_RUN_ID", "parent")
    monkeypatch.setenv("WANDB_PROJECT", "other-project")
    monkeypatch.setenv("WANDB_RUN_GROUP", "new-group")
    integration.init(state=STATE)
    kwargs = backend.init.call_args.kwargs
    assert kwargs["id"] == "child"
    assert kwargs["project"] == "other-project" and kwargs["group"] == "new-group"
    integration.init(state=STATE, settings={"project": "explicit", "run_group": ""})
    kwargs = backend.init.call_args.kwargs
    assert kwargs["project"] == "explicit" and kwargs["group"] == ""


def test_worker_imports_no_scheduler_or_tracking_sdk():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from exex.contrib import wandb; import sys; "
            "assert not any(n in sys.modules for n in "
            "['wandb', 'torch', 'tensorflow', 'jax', 'exex.xm_cluster', 'fabric'])",
        ],
        check=True,
    )


def test_real_sdk_offline_new_state_and_disabled(tmp_path):
    pytest.importorskip("wandb")
    # A fresh interpreter avoids W&B's process-wide settings cache.
    script = """
import os
import wandb
from exex.contrib import wandb as lxw
os.environ["WANDB_RUN_ID"] = "not-the-new-run"
os.environ.pop("WANDB_MODE", None)
wandb.setup(wandb.Settings(mode="offline"))
with lxw.init(project="qualification") as parent:
    parent.log({"train/global_step": 100, "loss": 0.5})
    state = lxw.checkpoint_state(parent)
    assert state["next_step"] == 1
    assert state["run_id"] != "not-the-new-run"
    assert parent.url is None
    assert parent.settings.mode == "offline"
with lxw.init(state=state, settings=wandb.Settings(mode="offline", project="new-project", run_group="different-group")) as child:
    assert child.id != parent.id
    assert child.project == "new-project"
    assert child.group == "different-group"
    assert child.step == 0
    assert child.config["exex/parent"] == state
    assert child.config["exex/history"] == "new"
with lxw.init(mode="disabled") as disabled:
    assert lxw.checkpoint_state(disabled) is None
assert "torch" not in __import__("sys").modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "WANDB_MODE": "offline", "WANDB_SILENT": "true"},
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )


def test_documented_example_retains_offline_history(tmp_path):
    pytest.importorskip("wandb")
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    store = tmp_path / "store"
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "[local.storage]\nstaging = " + json.dumps(str(store)) + "\n"
    )
    launcher = Path(__file__).resolve().parents[1] / "examples/wandb/launch.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "exex.cli.cli",
            "launch",
            str(launcher),
            f"--exex_config={config_file}",
        ],
        cwd=tmp_path,
        env={**os.environ, "WANDB_SILENT": "true"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    [experiment] = xc.list_experiments(config=xc.Config.from_file(str(config_file)))
    unit = experiment.work_units()[1]
    assert unit.get_status().is_completed
    assert unit.get_links() == {}  # Offline runs have no remote URL.
    artifacts = unit.artifacts()
    state = json.loads(
        artifacts["tracking_state"].fetch(tmp_path / "state.json").read_text()
    )
    assert state["next_step"] == 3
    assert state["project"] == "exex-example" and state["entity"] == "example"
    history = artifacts["wandb_history"].fetch(tmp_path / "history.wandb")
    reader = DataStore()
    reader.open_for_scan(str(history))
    rows = []
    try:
        while data := reader.scan_data():
            record = Record.FromString(data)
            if record.HasField("history"):
                rows.append(
                    {
                        tuple(item.nested_key): json.loads(item.value_json)
                        for item in record.history.item
                    }
                )
    finally:
        reader.close()
    assert [row[("train/global_step",)] for row in rows] == [0, 1, 2]
