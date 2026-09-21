import datetime
from unittest import mock

import pytest
from absl.testing import absltest
from absl.testing import parameterized

from lxm3 import xm
from lxm3 import xm_cluster as xc
from lxm3.xm_cluster import config
from lxm3.xm_cluster import executables
from lxm3.xm_cluster import executors
from lxm3.xm_cluster.execution import gridengine
from lxm3.xm_cluster.execution import local
from lxm3.xm_cluster.execution import slurm


class ExecutorTest(parameterized.TestCase):
    @parameterized.parameters(
        (None, None),
        (5 * 3600 + 20 * 60 + 25, datetime.timedelta(hours=5, minutes=20, seconds=25)),
        (datetime.timedelta(hours=3), datetime.timedelta(hours=3)),
    )
    def test_convert_time(self, input, expected):
        self.assertEqual(executors._convert_time(input), expected)

    @parameterized.parameters(
        (datetime.timedelta(hours=5, minutes=20, seconds=25), "05:20:25"),
        (datetime.timedelta(hours=55, minutes=20, seconds=25), "55:20:25"),
        (datetime.timedelta(hours=102, minutes=20, seconds=25), "102:20:25"),
    )
    def test_format_sge_time(self, input, expected):
        self.assertEqual(gridengine._format_time(input.total_seconds()), expected)

    @parameterized.parameters(
        (datetime.timedelta(hours=5, minutes=20, seconds=25), "05:20:25"),
        (datetime.timedelta(hours=55, minutes=20, seconds=25), "02-07:20:25"),
    )
    def test_format_slurm_time(self, input, expected):
        self.assertEqual(slurm._format_slurm_time(input), expected)


@pytest.mark.parametrize(
    "executor_type,builder_type",
    [
        (xc.Local, local.LocalJobScriptBuilder),
        (xc.Slurm, slurm.SlurmJobScriptBuilder),
        (xc.GridEngine, gridengine.GridEngineJobScriptBuilder),
    ],
)
@pytest.mark.parametrize(
    "runtime,matching_options",
    [
        (None, None),
        ("singularity", xc.SingularityOptions),
        ("docker", xc.DockerOptions),
        ("shifter", xc.ShifterOptions),
    ],
)
@pytest.mark.parametrize(
    "options_type", [None, xc.SingularityOptions, xc.DockerOptions, xc.ShifterOptions]
)
def test_container_options_match_executable(
    executor_type, builder_type, runtime, matching_options, options_type
):
    image = (
        executables.ContainerImage(
            "test-image", executables.ContainerImageType(runtime)
        )
        if runtime
        else None
    )
    executable = xc.AppBundle("test", "true", "/source.zip", container_image=image)
    options = options_type() if options_type else None
    job = xm.Job(executable, executor_type(container_options=options))
    builder = builder_type()
    if options_type is None or options_type is matching_options:
        assert builder.build(job, "test", "/logs")
    else:
        with pytest.raises(TypeError, match="container_options=.*does not match"):
            builder.build(job, "test", "/logs")


def test_mismatch_never_reaches_scheduler_submission():
    store = mock.Mock()
    client = slurm.SlurmClient(config.ClusterSettings({}), store)
    job = xm.Job(
        xc.AppBundle(
            "test",
            "true",
            "/source.zip",
            container_image=executables.ContainerImage(
                "id:test", executables.ContainerImageType.SHIFTER
            ),
        ),
        xc.Slurm(container_options=xc.DockerOptions()),
    )
    with mock.patch.object(client._cluster, "launch") as submit:
        with pytest.raises(TypeError, match="DockerOptions.*shifter"):
            client.launch("test", job)
    submit.assert_not_called()
    store.put_text.assert_not_called()


if __name__ == "__main__":
    absltest.main()
