"""Configuration and staging isolation, with no network or scheduler jobs."""

import asyncio
import os
from pathlib import Path
from unittest import mock

import pytest

from exex import xm
from exex import xm_cluster as xc
from exex._vendor.xmanager.xm.async_packager import PackageHasNotBeenCalledError
from exex.clusters import gridengine as native_gridengine
from exex.clusters import slurm as native_slurm
from exex.singularity import image_cache
from exex.xm_cluster import config as config_lib
from exex.xm_cluster import experiment as experiment_lib
from exex.xm_cluster.execution import job_script_builder


@pytest.fixture(autouse=True)
def isolated_event_loop_policy():
    # asyncio.run clears the current loop. Do not leak that state to vendor tests.
    previous = asyncio.get_event_loop_policy()
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    yield
    asyncio.set_event_loop_policy(previous)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("EXEX_CLUSTER", raising=False)
    monkeypatch.delenv("EXEX_PROJECT", raising=False)
    monkeypatch.setattr(experiment_lib, "_load_vcsinfo", lambda: None)
    return xc.Config(
        {
            "local": {"storage": {"staging": str(tmp_path / "local")}},
            "clusters": [
                {"name": name, "storage": {"staging": str(tmp_path / name)}}
                for name in ("s3df", "nersc")
            ],
        }
    )


@pytest.fixture
def source():
    return xc.UniversalPackage(
        entrypoint=["python3", "main.py"],
        path=str(Path(__file__).parent / "testdata/test_universal"),
        build_script="build.sh",
    )


async def resolved(awaitable):
    return await awaitable


def test_explicit_cluster_bypasses_environment(config, monkeypatch):
    monkeypatch.setenv("EXEX_CLUSTER", "nersc")
    assert config.cluster_settings().storage_root.endswith("nersc")
    assert config.cluster_settings("s3df").storage_root.endswith("s3df")
    assert os.environ["EXEX_CLUSTER"] == "nersc"
    with pytest.raises(ValueError, match="Unknown cluster: 'missing'"):
        config.cluster_settings("missing")


@pytest.mark.parametrize("executor_type", [xc.Slurm, xc.GridEngine])
def test_spec_carries_selected_site(executor_type):
    assert executor_type(cluster="nersc").Spec().cluster == "nersc"
    assert executor_type().Spec().cluster is None
    with pytest.raises(TypeError):
        executor_type.Spec()
    assert xc.Local.Spec() == xc.Local().Spec()


def test_explicit_project_does_not_mutate_config(config, monkeypatch):
    monkeypatch.setenv("EXEX_PROJECT", "ambient")
    one = xc.create_experiment("one", project="alpha", config=config)
    two = xc.create_experiment("two", project="beta", config=config)
    ambient = xc.create_experiment("ambient", config=config)
    assert (one._project, two._project, ambient._project) == (
        "alpha",
        "beta",
        "ambient",
    )
    assert config._project is None
    assert config.project() == "ambient"


def test_config_snapshot_pins_environment_paths_and_nested_data(
    config, source, monkeypatch, tmp_path
):
    monkeypatch.setenv("EXEX_CLUSTER", "s3df")
    monkeypatch.setenv("EXEX_PROJECT", "original")
    experiment = xc.create_experiment("snapshot", config=config)
    pending = experiment.package_async(xm.Packageable(source, xc.Slurm().Spec()))

    config._data["clusters"][0]["storage"]["staging"] = str(tmp_path / "changed")
    monkeypatch.setenv("EXEX_CLUSTER", "nersc")
    monkeypatch.setenv("EXEX_PROJECT", "changed")
    with mock.patch.object(
        config_lib, "default", side_effect=AssertionError("global lookup")
    ):
        experiment.package()
        executable = asyncio.run(resolved(pending))
        assert (
            str(tmp_path / "s3df/projects/original/archives") in executable.resource_uri
        )
        assert Path(executable.resource_uri).is_file()
        with mock.patch.object(
            native_slurm.SlurmCluster, "launch", return_value="101"
        ) as submit:
            with experiment:
                experiment.add(xm.Job(executable, xc.Slurm()))
        assert str(tmp_path / "s3df/projects/original/jobs") in submit.call_args.args[0]
    assert not (tmp_path / "changed").exists()


def test_relative_local_and_onsite_roots_are_pinned(config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config._data["local"]["storage"]["staging"] = "local"
    config._data["clusters"][0]["storage"]["staging"] = "s3df"
    experiment = xc.create_experiment("relative", config=config)
    monkeypatch.chdir(tmp_path.parent)
    assert experiment._config.local_settings().storage_root == str(tmp_path / "local")
    assert experiment._config.cluster_settings("s3df").storage_root == str(
        tmp_path / "s3df"
    )
    assert config.local_settings().storage_root == "local"


def test_packaging_queues_belong_to_experiments(config, source):
    one = xc.create_experiment("one", project="alpha", config=config)
    two = xc.create_experiment("two", project="beta", config=config)
    packageable = xm.Packageable(source, xc.Local.Spec())
    first = one.package_async(packageable)
    second = two.package_async(packageable)
    one.package()
    assert "/projects/alpha/" in asyncio.run(resolved(first)).resource_uri
    with pytest.raises(PackageHasNotBeenCalledError):
        asyncio.run(resolved(second))
    two.package()
    assert "/projects/beta/" in asyncio.run(resolved(second)).resource_uri


def test_local_package_and_execution_use_same_project(config, source, tmp_path):
    marker = tmp_path / "executed"
    source.entrypoint = ["touch", str(marker)]
    experiment = xc.create_experiment("local", project="alpha", config=config)
    executor = xc.Local()
    [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
    with experiment:
        experiment.add(xm.Job(executable, executor))
    assert marker.is_file()
    assert Path(executable.resource_uri).is_relative_to(
        tmp_path / "local/projects/alpha"
    )


@pytest.mark.parametrize(
    "executor_type,native",
    [
        (xc.Slurm, native_slurm.SlurmCluster),
        (xc.GridEngine, native_gridengine.GridEngineCluster),
    ],
)
def test_two_sites_two_projects_package_and_submit(
    config, source, tmp_path, executor_type, native
):
    experiments = [
        xc.create_experiment(name, project=name, config=config)
        for name in ("alpha", "beta")
    ]
    executors = [executor_type(cluster=name) for name in ("s3df", "nersc")]
    bundles = {
        experiment: experiment.package(
            [xm.Packageable(source, executor.Spec()) for executor in executors]
        )
        for experiment in experiments
    }
    with mock.patch.object(native, "launch", return_value="101") as submit:
        for experiment in reversed(experiments):
            with experiment:
                for executor, executable in zip(executors, bundles[experiment]):
                    root = (
                        tmp_path / executor.cluster / "projects" / experiment._project
                    )
                    assert Path(executable.resource_uri).is_relative_to(
                        root / "archives"
                    )
                    assert Path(executable.resource_uri).is_file()
                    experiment.add(xm.Job(executable, executor))
            submitted = [Path(call.args[0]) for call in submit.call_args_list[-2:]]
            for executor, script in zip(executors, submitted):
                assert script.is_relative_to(
                    tmp_path
                    / executor.cluster
                    / "projects"
                    / experiment._project
                    / "jobs"
                )
                assert script.is_file()
    assert submit.call_count == 4
    assert config._project is None


def test_async_context_accepts_arrays_and_generators(config, source):
    executor = xc.Slurm(cluster="nersc")
    experiment = xc.create_experiment("async", project="alpha", config=config)
    [executable] = experiment.package([xm.Packageable(source, executor.Spec())])

    async def generator(unit):
        await unit.add(xm.Job(executable, executor))

    async def launch():
        async with experiment:
            await experiment.add(generator)
            await experiment.add(
                xc.ArrayJob(executable, executor, args=[{"seed": 0}, {"seed": 1}])
            )

    with mock.patch.object(
        native_slurm.SlurmCluster, "launch", return_value="101"
    ) as submit:
        asyncio.run(launch())
    assert submit.call_count == 2
    array_script = Path(submit.call_args.args[0]).read_text()
    assert "#SBATCH --array=1-2" in array_script
    assert "/nersc/projects/alpha/" in submit.call_args.args[0]


@pytest.mark.parametrize("change", ["site", "project", "profile", "backend"])
def test_wrong_destination_is_rejected_before_submission(
    config, source, change, tmp_path
):
    experiment = xc.create_experiment("owner", project="alpha", config=config)
    executor = xc.Slurm(cluster="s3df")
    [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
    if change == "site":
        executor = xc.Slurm(cluster="nersc")
    elif change == "backend":
        executor = xc.GridEngine(cluster="s3df")
    elif change == "profile":
        config._data["clusters"][0]["storage"]["staging"] = str(tmp_path / "changed")
    destination = xc.create_experiment(
        "destination", project="beta" if change == "project" else "alpha", config=config
    )
    with (
        mock.patch.object(experiment_lib.local_execution, "client") as local_client,
        mock.patch.object(experiment_lib.slurm_execution, "client") as slurm_client,
        mock.patch.object(
            experiment_lib.gridengine_execution, "client"
        ) as gridengine_client,
    ):
        with pytest.raises(ValueError, match="different destination"):
            asyncio.run(
                experiment_lib._launch(
                    "destination",
                    "1",
                    xm.JobGroup(job=xm.Job(executable, executor)),
                    config=destination._config,
                    project=destination._project,
                )
            )
        local_client.assert_not_called()
        slurm_client.assert_not_called()
        gridengine_client.assert_not_called()
    assert not (tmp_path / "changed").exists()


def test_submission_error_is_not_retried(config, source):
    experiment = xc.create_experiment("failure", config=config)
    executor = xc.Slurm(cluster="s3df")
    [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
    with mock.patch.object(
        native_slurm.SlurmCluster, "launch", side_effect=ConnectionError("disconnected")
    ) as submit:
        with pytest.raises(ConnectionError, match="disconnected"):
            asyncio.run(
                experiment_lib._launch(
                    "failure",
                    "1",
                    xm.JobGroup(job=xm.Job(executable, executor)),
                    config=experiment._config,
                    project=experiment._project,
                )
            )
        submit.assert_called_once()


def test_store_uses_supplied_remote_settings_and_project():
    settings = config_lib.ClusterSettings(
        {
            "server": "selected-host",
            "user": "selected-user",
            "storage": {"staging": "remote-staging"},
        }
    )
    with (
        mock.patch.object(job_script_builder.sftp, "SFTPFileSystem") as filesystem,
        mock.patch.object(
            config_lib, "default", side_effect=AssertionError("global lookup")
        ),
    ):
        filesystem.return_value.ftp.normalize.return_value = "/resolved/staging"
        store = job_script_builder.create_artifact_store(
            settings=settings, project="alpha"
        )
        filesystem.assert_called_once_with(
            host="selected-host", username="selected-user"
        )
        filesystem.return_value.ftp.normalize.assert_called_once_with("remote-staging")
        assert store.storage_root == "/resolved/staging/projects/alpha"


def test_nested_container_uses_experiment_image_cache(config, source, tmp_path):
    first = xc.create_experiment("first", config=config)
    config._data["local"]["storage"]["staging"] = str(tmp_path / "other-local")
    second = xc.create_experiment("second", config=config)
    spec = xc.SingularityContainer(source, image_path="docker-daemon://test:latest")
    blob = tmp_path / "test.sif"
    blob.touch()
    with (
        mock.patch.object(
            config_lib, "default", side_effect=AssertionError("global lookup")
        ),
        mock.patch.object(
            image_cache,
            "get_cached_image",
            return_value=image_cache.ImageInfo(
                digest="test",
                path=str(blob),
                blob_path=str(blob),
            ),
        ) as cached_image,
    ):
        for experiment in (first, second):
            experiment.package([xm.Packageable(spec, xc.Local.Spec())])
    assert [call.kwargs["cache_dir"] for call in cached_image.call_args_list] == [
        str(tmp_path / "local/image_cache"),
        str(tmp_path / "other-local/image_cache"),
    ]
