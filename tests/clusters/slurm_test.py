import subprocess
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

from exex.clusters import slurm


class SlurmTest(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": "job", "text": "6\n", "expected": 6},
            {"testcase_name": "federation", "text": "6;cluster\n", "expected": 6},
        ]
    )
    def test_parse_job_id(self, text, expected):
        job_id = slurm.parse_job_id(text)
        self.assertEqual(job_id, expected)

    def test_parse_invalid_job_id(self):
        with self.assertRaises(ValueError):
            slurm.parse_job_id("Failed")


class ClusterTest(absltest.TestCase):
    @mock.patch.object(slurm.ssh, "run")
    def test_cluster(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="6\n")
        cluster = slurm.SlurmCluster(hostname="host", username="user")
        job_id = cluster.launch("job.sbatch")
        self.assertEqual(job_id, "6")
        run.assert_called_once_with(
            ["sbatch", "--parsable", "--", "job.sbatch"],
            hostname="host",
            username="user",
        )


if __name__ == "__main__":
    absltest.main()
