# Fork baseline

Recorded 2026-09-16, before any execution-code modifications.

| Item | Evidence |
| --- | --- |
| Fork | `youngsm/lxm3`, confirmed GitHub fork of `ethanluoyc/lxm3` |
| Upstream/base commit | `30f55855e595ee7d8397daacb1475442265f37ef` |
| Planning branch | `r&d/fork-plan`; `main` retains the upstream baseline |
| Remotes | `origin` is the user's fork; `upstream` is the original repository |
| Scope of changes | Fork-status README, implementation-delta plan and this record only |
| Isolation | Temporary virtual environment; no changes to pimm/exex runtimes or code |
| Selected non-live tests | **72 passed, 2 deselected**, 2 deprecation warnings, 2.28 seconds |

## Test environment and command

Linux, CPython 3.12.13. Editable upstream package version
`0.4.4.dev22+g30f5585`; pytest 9.1.1, pytest-golden 1.0.1, absl-py 2.5.0,
attrs 26.1.0, fsspec 2026.7.0, Fabric 3.2.3, Paramiko 5.0.0,
vcsinfo 2.2.116 and optional PEX 2.103.0. Dependencies were resolved in an isolated
environment, not installed from upstream's older `pdm.lock`. This establishes the
recorded environment only, not every supported Python/dependency combination.

From the checkout, with that environment's executables on PATH:

```sh
timeout --signal=INT --kill-after=5s 60s python -m pytest \
  -q -p no:cacheprovider -o faulthandler_timeout=20 -m 'not integration' \
  tests/config_test.py tests/executor_test.py tests/execution_test.py \
  tests/experiment_test.py tests/clusters/slurm_test.py tests/array_job_test.py \
  tests/artifact_test.py tests/packaging_test.py tests/singularity_test.py
```

These selected cases cover configuration, script rendering, mocked submission,
temporary local scripts, Experiment construction, arrays, staging, package creation
and Singularity URI handling. They do not comprise the complete upstream/vendor
suite. The two integration cases were deliberately deselected: no Docker/Singularity
containers, GPU jobs, real SSH/Slurm submissions or image pulls were performed.

The two warnings concern vendored XManager's deprecated asyncio child-watcher API.
They do not fail this Python 3.12 baseline and do not establish Python 3.14 support.

## Initial environment issues, not hidden passes

The first restricted-sandbox invocation stalled at the Experiment tests after 37
passes and was explicitly interrupted. The same two mocked Experiment tests passed
outside that sandbox in 1.95 seconds. The final selected suite above also ran
outside the restricted sandbox, under a timeout. No production code was changed
to make it pass; the precise sandbox restriction was not diagnosed here.

An initial packaging pass failed because the optional `pex` executable was absent.
Installing the declared optional dependency into the temporary environment and
putting its executables on PATH resolved that prerequisite. No test was removed
or weakened. Both temporary test processes completed or were explicitly stopped.

## Limits and provenance

Passing existing tests does not make the missing lifecycle, immutable-source,
output-retention or site capabilities implemented. Known code-level defects and
their required new regressions are listed in the [fork plan](fork-plan.md).
Real S3DF/NERSC/Vertex and pimm qualification must follow the relevant additions.
Existing exex evidence is a reference for those tests, not evidence that the fork
already implements their behavior.

The fork preserves upstream history and notices. `pyproject.toml` declares MIT;
the checkout includes the vendored XManager Apache-2.0 license and copyright
headers, but no top-level LICENSE file was found. No licensing text was invented,
removed or changed as part of this baseline.
