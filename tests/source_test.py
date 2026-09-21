"""Source capture through the public API, including execution without a checkout."""

import asyncio
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from unittest import mock

import attr
import pytest

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.xm_cluster import experiment as experiment_lib
from lxm3.xm_cluster.packaging import router


@pytest.fixture(autouse=True)
def isolated_loop_policy():
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.delenv("LXM_PROJECT", raising=False)
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    config = xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "store")}},
            "clusters": [
                {"name": "other", "storage": {"staging": str(tmp_path / "other")}}
            ],
        }
    )
    return xc.create_experiment("source", project="test", config=config)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "payload.txt").write_text("captured")
    (root / "worker.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "Path(sys.argv[1]).write_text(json.dumps([Path('payload.txt').read_text(), sys.argv[2:], os.environ.get('MESSAGE')]))\n"
    )
    return root


def spec(root, entrypoint=None, files=("worker.py", "payload.txt")):
    return xc.SourceTree(entrypoint or xc.ModuleName("worker"), root, files=files)


def contents(frozen):
    with tarfile.open(frozen._archive_path) as archive:
        return {member.name: archive.extractfile(member).read() for member in archive}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def test_git_working_tree_selection_does_not_touch_head_or_index(experiment, checkout):
    git(checkout, "init", "-q")
    (checkout / ".gitignore").write_text("*.ignore\n")
    git(checkout, "add", ".")
    git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    (checkout / "payload.txt").write_text("staged")
    git(checkout, "add", "payload.txt")
    (checkout / "payload.txt").write_text("working tree")
    (checkout / "worker.py").unlink()
    odd_name = "new ' $file\n.txt"
    (checkout / odd_name).write_text("untracked")
    (checkout / "secret.ignore").write_text("excluded")
    before = (
        git(checkout, "rev-parse", "HEAD"),
        (checkout / ".git/index").read_bytes(),
    )
    frozen = experiment.freeze(spec(checkout, files=None))
    assert contents(frozen) == {
        ".gitignore": b"*.ignore\n",
        "payload.txt": b"working tree",
        odd_name: b"untracked",
    }
    assert before == (
        git(checkout, "rev-parse", "HEAD"),
        (checkout / ".git/index").read_bytes(),
    )


def test_git_selection_is_relative_to_source_subdirectory(experiment, checkout):
    git(checkout, "init", "-q")
    nested = checkout / "nested"
    nested.mkdir()
    (nested / "data").write_text("selected")
    frozen = experiment.freeze(spec(nested, files=None))
    assert contents(frozen) == {"data": b"selected"}


def test_identity_ignores_checkout_name_order_and_timestamps(
    experiment, checkout, tmp_path
):
    first = experiment.freeze(spec(checkout))
    moved = tmp_path / "different-name"
    shutil.copytree(checkout, moved)
    os.utime(moved / "payload.txt", (1, 1))
    second = experiment.freeze(
        spec(moved, files=["payload.txt", "./worker.py", "payload.txt"])
    )
    assert first.id == second.id
    assert first._archive_path == second._archive_path
    assert first.name != second.name
    assert experiment.sources() == {first.id: first}
    with pytest.raises(attr.exceptions.FrozenInstanceError):
        first.id = "changed"


@pytest.mark.parametrize("change", ["bytes", "path", "executable", "entrypoint"])
def test_semantic_changes_change_identity(experiment, checkout, change):
    source = spec(checkout)
    first = experiment.freeze(source)
    if change == "bytes":
        (checkout / "payload.txt").write_text("changed")
    elif change == "path":
        (checkout / "payload.txt").rename(checkout / "renamed.txt")
        source.files = ["worker.py", "renamed.txt"]
    elif change == "executable":
        (checkout / "worker.py").chmod(0o755)
    else:
        source.entrypoint = xc.CommandList(["python3 worker.py"])
    assert first.id != experiment.freeze(source).id


@pytest.mark.parametrize(
    "entrypoint",
    [
        xc.ModuleName("worker"),
        xc.CommandList(["python3 worker.py"]),
        xc.CommandList(['python3 worker.py "$@"']),
    ],
)
def test_capture_survives_deleted_checkout_and_reprepares_for_other_site(
    experiment, checkout, tmp_path, entrypoint
):
    frozen = experiment.freeze(spec(checkout, entrypoint))
    (checkout / "payload.txt").write_text("too late")
    shutil.rmtree(checkout)  # Only this test's generated fixture.
    output = tmp_path / "result.json"
    message = "literal $value ' with spaces\nand a newline"
    with experiment:
        local, remote = experiment.package(
            [
                xm.Packageable(frozen, xc.Local.Spec()),
                xm.Packageable(frozen, xc.Slurm(cluster="other").Spec()),
            ]
        )
        experiment.add(
            xm.Job(
                local,
                xc.Local(),
                args=[str(output), message],
                env_vars={"MESSAGE": message},
            )
        )
    assert json.loads(output.read_text()) == ["captured", [message], message]
    assert local.resource_uri != remote.resource_uri
    assert (
        Path(local.resource_uri).read_bytes() == Path(remote.resource_uri).read_bytes()
    )


def test_executable_bits_and_command_list_failure(experiment, checkout, tmp_path):
    script = checkout / "run.sh"
    script.write_text('#!/bin/sh\nprintf "%s" "$1" > "$2"\nexit 7\n')
    script.chmod(0o751)
    source = spec(checkout, xc.CommandList(["./run.sh"]), ["run.sh"])
    frozen = experiment.freeze(source)
    # Mutating the caller's command list cannot change the frozen entrypoint.
    source.entrypoint.commands[:] = ["true"]
    with tarfile.open(frozen._archive_path) as archive:
        assert archive.getmember("run.sh").mode == 0o755
    output = tmp_path / "result"
    with pytest.raises(subprocess.CalledProcessError) as error:
        with experiment:
            [bundle] = experiment.package([xm.Packageable(frozen, xc.Local.Spec())])
            experiment.add(
                xm.Job(bundle, xc.Local(), args=["literal $value", str(output)])
            )
    assert error.value.returncode == 7
    assert output.read_text() == "literal $value"


def test_explicit_freeze_and_queued_packaging_have_distinct_capture_times(
    experiment, checkout
):
    source = spec(checkout)
    frozen = experiment.freeze(source)
    queued = experiment.package_async(xm.Packageable(source, xc.Local.Spec()))
    assert experiment.sources() == {frozen.id: frozen}
    (checkout / "payload.txt").write_text("after queue")
    [earlier] = experiment.package([xm.Packageable(frozen, xc.Local.Spec())])

    async def resolve():
        return await queued

    later = asyncio.run(resolve())
    assert set(experiment.sources()) == {frozen.id, later._source.id}
    with tarfile.open(earlier.resource_uri) as archive:
        assert archive.extractfile("payload.txt").read() == b"captured"
    with tarfile.open(later.resource_uri) as archive:
        assert archive.extractfile("payload.txt").read() == b"after queue"


@pytest.mark.parametrize(
    "wrapper,image",
    [
        (xc.DockerContainer, "python:3.12"),
        (xc.SingularityContainer, "docker://python:3.12"),
        (xc.ShifterContainer, "id:installed"),
    ],
)
@pytest.mark.parametrize("freeze_first", [False, True])
def test_existing_container_wrappers_accept_both_source_specs(
    experiment, checkout, wrapper, image, freeze_first
):
    source = spec(checkout)
    if freeze_first:
        source = experiment.freeze(source)
    [bundle] = experiment.package(
        [xm.Packageable(wrapper(source, image), xc.Local.Spec())]
    )
    assert bundle.container_image.name == image
    retrieved = xc.get_experiment(
        experiment.experiment_id, config=experiment._config
    ).sources()
    assert list(retrieved) == [bundle._source.id]
    [bare] = experiment.package(
        [xm.Packageable(retrieved[bundle._source.id], xc.Local.Spec())]
    )
    assert bare.container_image is None  # Source lookup does not reconstruct a runtime.
    with tarfile.open(bundle.resource_uri) as archive:
        assert archive.extractfile("payload.txt").read() == b"captured"


def test_corrupt_retained_source_fails_before_transfer(experiment, checkout):
    frozen = experiment.freeze(spec(checkout))
    Path(frozen._archive_path).write_bytes(b"corrupt test archive")
    with mock.patch.object(router, "_transfer_file") as transfer:
        with pytest.raises(ValueError, match="content identity"):
            experiment.package([xm.Packageable(frozen, xc.Local.Spec())])
    transfer.assert_not_called()


def test_freeze_does_not_build_upload_or_submit(experiment, checkout):
    with mock.patch.object(router, "_transfer_file") as transfer:
        with mock.patch.object(subprocess, "Popen") as process:
            frozen = experiment.freeze(spec(checkout))
    assert contents(frozen)["payload.txt"] == b"captured"
    transfer.assert_not_called()
    process.assert_not_called()


def test_empty_allowlist_can_run_a_command(experiment, checkout, tmp_path):
    output = tmp_path / "result"
    source = spec(checkout, xc.CommandList(['printf captured > "$1"']), files=[])
    with experiment:
        [bundle] = experiment.package([xm.Packageable(source, xc.Local.Spec())])
        experiment.add(xm.Job(bundle, xc.Local(), args=[str(output)]))
    assert output.read_text() == "captured"


@pytest.mark.parametrize("name", ["job-param.sh", ".environment", ".environment/file"])
def test_runtime_files_cannot_overwrite_selected_source(experiment, checkout, name):
    path = checkout / name
    path.parent.mkdir(exist_ok=True)
    path.write_text("must not be overwritten")
    with pytest.raises(ValueError, match="reserved"):
        experiment.freeze(spec(checkout, files=[name]))


@pytest.mark.parametrize("selection", [["../outside"], ["/etc/hosts"], ["missing"]])
def test_invalid_explicit_paths_do_not_become_archives(
    experiment, checkout, tmp_path, selection
):
    with pytest.raises((ValueError, FileNotFoundError)):
        experiment.freeze(spec(checkout, files=selection))
    assert not list((tmp_path / "store/sources").glob("*.tar"))
    assert experiment.sources() == {}


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_non_regular_files_are_not_silently_captured(experiment, checkout, kind):
    path = checkout / "selected"
    if kind == "symlink":
        path.symlink_to("payload.txt")
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError, match="regular files"):
        experiment.freeze(spec(checkout, files=["selected"]))


def test_symlinked_parent_cannot_capture_outside_tree(experiment, checkout, tmp_path):
    (tmp_path / "outside").write_text("not selected")
    (checkout / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="within"):
        experiment.freeze(spec(checkout, files=["escape/outside"]))


def test_local_store_inside_checkout_does_not_capture_itself(checkout, monkeypatch):
    git(checkout, "init", "-q")
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    config = xc.Config({"local": {"storage": {"staging": str(checkout / ".lxm")}}})
    experiment = xc.create_experiment("capture", config=config)
    first = experiment.freeze(spec(checkout, files=None))
    second = experiment.freeze(spec(checkout, files=None))
    assert first.id == second.id
    assert set(contents(first)) == {"worker.py", "payload.txt"}
