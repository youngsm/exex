# Native Slurm steps

Requesting multiple nodes allocates resources; it does not launch a worker on
each node. This standard-library example unpacks once into shared storage, then
its `driver.sh` uses `srun` to start two workers. The workers verify the shared
source and a token created by the driver.

With an explicit [cluster configuration](../../docs/configuration.md):

```sh
EXEX_CONFIG=/path/to/exex.toml exex launch examples/slurm_step/launch.py -- \
  --cluster=nersc --python=/usr/bin/python3 \
  --output_dir=/shared/probe/result-001 --workdir_root=/shared/probe/work \
  --resource=account=my-account --resource=qos=debug \
  --resource=constraint=cpu
```

The launcher requests two nodes, one CPU task per node, and two minutes. Choose
resources permitted by your site. Use a new output directory each time. Add
`--fail_worker` to check that a failed worker stops its peers and fails the job.
The source, work directory and output directory must be visible on both nodes.
This example does not test container placement or ML collectives.

Historical qualification evidence is retained in
[the development archive](../../docs/development/archive/slurm_step-qualification.md).
