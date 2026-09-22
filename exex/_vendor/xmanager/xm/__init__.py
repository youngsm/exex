# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XManager client API.

Provides XManager public API for configuring and launching experiments.
"""

from exex._vendor.xmanager.xm import job_operators
from exex._vendor.xmanager.xm.compute_units import *
from exex._vendor.xmanager.xm.core import AuxiliaryUnitJob
from exex._vendor.xmanager.xm.core import AuxiliaryUnitRole
from exex._vendor.xmanager.xm.core import Experiment
from exex._vendor.xmanager.xm.core import ExperimentUnit
from exex._vendor.xmanager.xm.core import ExperimentUnitError
from exex._vendor.xmanager.xm.core import ExperimentUnitFailedError
from exex._vendor.xmanager.xm.core import ExperimentUnitNotCompletedError
from exex._vendor.xmanager.xm.core import ExperimentUnitRole
from exex._vendor.xmanager.xm.core import ExperimentUnitStatus
from exex._vendor.xmanager.xm.core import Importance
from exex._vendor.xmanager.xm.core import LaunchedJob
from exex._vendor.xmanager.xm.core import NotFoundError
from exex._vendor.xmanager.xm.core import WorkUnit
from exex._vendor.xmanager.xm.core import WorkUnitCompletedAwaitable
from exex._vendor.xmanager.xm.core import WorkUnitRole
from exex._vendor.xmanager.xm.executables import BazelBinary
from exex._vendor.xmanager.xm.executables import BazelContainer
from exex._vendor.xmanager.xm.executables import Binary
from exex._vendor.xmanager.xm.executables import BinaryDependency
from exex._vendor.xmanager.xm.executables import CommandList
from exex._vendor.xmanager.xm.executables import Container
from exex._vendor.xmanager.xm.executables import Dockerfile
from exex._vendor.xmanager.xm.executables import ModuleName
from exex._vendor.xmanager.xm.executables import PythonContainer
from exex._vendor.xmanager.xm.job_blocks import Constraint
from exex._vendor.xmanager.xm.job_blocks import Executable
from exex._vendor.xmanager.xm.job_blocks import ExecutableSpec
from exex._vendor.xmanager.xm.job_blocks import Executor
from exex._vendor.xmanager.xm.job_blocks import ExecutorSpec
from exex._vendor.xmanager.xm.job_blocks import get_args_for_all_jobs
from exex._vendor.xmanager.xm.job_blocks import Job
from exex._vendor.xmanager.xm.job_blocks import JobConfig
from exex._vendor.xmanager.xm.job_blocks import JobGeneratorType
from exex._vendor.xmanager.xm.job_blocks import JobGroup
from exex._vendor.xmanager.xm.job_blocks import JobType
from exex._vendor.xmanager.xm.job_blocks import merge_args
from exex._vendor.xmanager.xm.job_blocks import Packageable
from exex._vendor.xmanager.xm.job_blocks import SequentialArgs
from exex._vendor.xmanager.xm.job_blocks import UserArgs
from exex._vendor.xmanager.xm.metadata_context import ContextAnnotations
from exex._vendor.xmanager.xm.metadata_context import MetadataContext
from exex._vendor.xmanager.xm.packagables import bazel_binary
from exex._vendor.xmanager.xm.packagables import bazel_container
from exex._vendor.xmanager.xm.packagables import binary
from exex._vendor.xmanager.xm.packagables import container
from exex._vendor.xmanager.xm.packagables import dockerfile_container
from exex._vendor.xmanager.xm.packagables import python_container
from exex._vendor.xmanager.xm.resources import GpuType
from exex._vendor.xmanager.xm.resources import InvalidTpuTopologyError
from exex._vendor.xmanager.xm.resources import JobRequirements
from exex._vendor.xmanager.xm.resources import ResourceDict
from exex._vendor.xmanager.xm.resources import ResourceQuantity
from exex._vendor.xmanager.xm.resources import ResourceType
from exex._vendor.xmanager.xm.resources import ServiceTier
from exex._vendor.xmanager.xm.resources import Topology
from exex._vendor.xmanager.xm.resources import TpuType
from exex._vendor.xmanager.xm.utils import run_in_asyncio_loop
from exex._vendor.xmanager.xm.utils import ShellSafeArg
