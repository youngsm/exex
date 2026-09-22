# type: ignore
import os
import unittest.mock

from absl.testing import absltest
from absl.testing import parameterized

from exex.xm_cluster import config as config_lib

_SAMPLE_CONFIG = """
[local]
[local.storage]
staging = ".exex"

[[clusters]]
name = "cs"
server = "beaker.cs.ucl.ac.uk"
user = "foo"

[clusters.storage]
staging = "/home/foo/exex-staging"

[[clusters]]
name = "myriad"
server = "myriad.rc.ucl.ac.uk"
user = "ucaby36"

[clusters.storage]
staging = "/home/bar/Scratch/exex-staging"

"""


def _test_config():
    return config_lib.Config.from_string(_SAMPLE_CONFIG)


class ConfigTest(parameterized.TestCase):
    def test_config(self):
        config = _test_config()
        self.assertTrue(isinstance(config._data["clusters"], list))
        self.assertEqual(config._data["clusters"][0]["name"], "cs")

    def test_local_config(self):
        config = _test_config()
        self.assertEqual(config.local_settings().storage_root, ".exex")

    def test_cluster_not_configured(self):
        config = config_lib.Config({}, None)
        with self.assertRaisesRegex(ValueError, "No cluster.*"):
            config.cluster_settings()

    def test_empty_local_config(self):
        local = config_lib.LocalSettings({})
        self.assertEqual(local.storage_root, os.path.join(os.getcwd(), ".exex"))

    def test_empty_cluster_config(self):
        config = config_lib.ClusterSettings({})
        self.assertEqual(config.storage_root, "exex-staging")
        self.assertEqual(config.user, None)
        self.assertEqual(config.hostname, None)
        self.assertEqual(config.ssh_config, {})

    def test_default_cluster(self):
        config = _test_config()
        self.assertEqual(config.default_cluster(), "cs")
        with unittest.mock.patch.dict("os.environ", {"EXEX_CLUSTER": "myriad"}):
            self.assertEqual(config.default_cluster(), "myriad")

    def test_config_project(self):
        config = config_lib.Config.from_string("")
        self.assertEqual(config.project(), None)
        with unittest.mock.patch.dict("os.environ", {"EXEX_PROJECT": "test"}):
            self.assertEqual(config.project(), "test")

    def test_default_config(self):
        config = config_lib.Config.default()
        local = config.local_settings()
        assert local.storage_root
        assert config.project
        with self.assertRaises(ValueError):
            config.cluster_settings()


if __name__ == "__main__":
    absltest.main()
