# Contributing

Keep the execution core independent of application frameworks. Use ordinary
subprocess and scheduler failures; do not add implicit retry or reconciliation.
Qualify changes with small jobs before running expensive workloads.

```bash
python -m pip install -e '.[wandb]' pytest pytest-golden pex
python -m pytest -m 'not integration'
```

The suite includes vendored XManager tests. Live integration tests require
explicit site configuration, credentials, and permission to allocate resources.
Never put credentials or generated artifacts in the repository.

## Source layout

| Directory | Responsibility |
| --- | --- |
| `exex/xm/` | Public XManager job vocabulary |
| `exex/xm_cluster/` | Experiment, executors, packaging, artifacts, and catalog |
| `exex/xm_cluster/execution/` | Worker scripts and execution-site helpers |
| `exex/clusters/` | Native scheduler and SSH commands |
| `exex/contrib/` | Optional integrations, including W&B |
| `exex/cli/` | Launch and inspection commands |
| `exex/_vendor/` | Vendored upstream code; preserve attribution |
| `tests/` | Unit and script-level regression tests |

The catalog belongs to the author process. Workers communicate through staged
files and receipts, not by opening the author's database. Slurm owns batch-job
lifetimes; an attached driver owns `salloc` lifetimes. The application owns
distributed process startup and checkpoint semantics.

## Documentation

User guides describe supported behavior and runnable examples. Keep development
history and native qualification IDs out of the README and ordinary guides.
Historical design proposals and qualification evidence are preserved in
[the development archive](docs/development/README.md); they are not API contracts.

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs /tmp/exex-docs
```

The distribution, import, and command are all `exex`. The rollout renamed the
former `LXM_*` environment to `EXEX_*` and `lxm.toml` to `exex.toml`; there are no
import aliases for the old implementation. Do not rewrite existing staged jobs
or move live catalogs as part of a source change. Point `EXEX_CONFIG` explicitly
at the desired storage and keep the old environment for historical launches.
