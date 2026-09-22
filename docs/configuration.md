# Hosts and storage

Exex uses an explicit named cluster. Configure transport and storage here;
resource requests, accounts, partitions, and QoS belong on the executor.

Save `exex.toml` in your working directory:

```toml
[local.storage]
staging = "/persistent/author/exex"

[[clusters]]
name = "mycluster"
server = "my-ssh-alias"

[clusters.storage]
staging = "/shared/cluster/exex"
```

Replace the paths and alias with your own. Omit `server` for on-site submission.
`user` is optional; SSH configuration can supply it. Slurm uses system OpenSSH,
including your existing agent, `ProxyJump`, and authentication configuration.
Credentials do not belong in this file or a job's recorded environment.

Lookup order is `--exex_config`, `EXEX_CONFIG`, `./exex.toml`, then
`~/.config/exex/config.toml`. Without configuration, local execution uses `.exex`.
`EXEX_CLUSTER` and `EXEX_PROJECT` supply optional defaults; explicit executor
cluster names remain preferable in reusable launchers.

```bash
export EXEX_CONFIG=/path/to/exex.toml
exex launch examples/quickstart/launch.py -- --cluster=mycluster
```

The author storage contains the SQLite catalog. Cluster storage holds staged
source, scripts, logs, and retained artifacts, and must be reachable from compute
nodes. Choose persistent/shared storage appropriate to the cluster's purge rules.
Workers do not connect to the author's catalog.

`Local(workdir_root=...)` and `Slurm(workdir_root=...)` select the execution-host
parent for temporary unpacking. Multi-node entrypoints need a shared path. Exex
runs the entrypoint once; your application can use `srun` or its framework's own
launcher to start workers. Node-local replication is not automatic.

SSH, submission and filesystem errors propagate normally. There is no automatic
resubmission after a lost connection. Inspect the scheduler when submission
outcome is uncertain before deciding to submit again.
