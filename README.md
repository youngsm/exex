# exex

Run experiments locally or on Slurm, using explicit execution targets and
reusable environments. Package the code you have—including uncommitted changes—
then inspect jobs, retain outputs, and pass checkpoints to subsequent jobs.

Exex launches processes. It does not depend on PyTorch, TensorFlow, JAX, or a
particular training loop, and it does not choose a cluster for you.

## Get started

Requires Python 3.10–3.12 on the authoring host. Install this repository; there
is no exex PyPI release for this implementation yet.

```bash
git clone https://github.com/youngsm/exex.git
cd exex
python -m pip install -e .
exex launch examples/quickstart/launch.py
exex experiments
```

The example runs a standard-library Python worker locally, retains its output,
and prints the experiment ID. Local execution needs no cluster configuration.
Use a prepared Python environment or a prebuilt container for your own workload;
packaging source does not install application dependencies on compute nodes.

For Slurm, configure a named site once, then use the same example:

```bash
exex launch examples/quickstart/launch.py -- \
  --cluster=mycluster --resource=account=my-account --walltime=300
```

See [getting started](docs/getting-started.md) for the complete Python example
and [cluster configuration](docs/configuration.md) for SSH and storage setup.

## Guides

- [Source and containers](docs/packaging.md): dirty checkouts, existing SIF and Shifter images.
- [Job inspection](docs/inspection.md): status, logs, cancellation, and reopening experiments.
- [Artifacts](docs/artifacts.md): retained outputs, input bindings, and cross-site transfer.
- [Continuation](docs/continuation.md): checkpoint-aware batch requeue and attached `salloc` chains.
- [Weights & Biases](docs/wandb.md): native runs, history policy, and task links.
- [API reference](docs/api.rst).

Local and Slurm are the maintained execution paths. Slurm uses your system
OpenSSH configuration; Singularity/Apptainer and Shifter use existing runtimes.
Cloud execution, automatic placement, and background recovery
services are not implemented. See each guide for its supported scope.

## Contributing

Read the [contributor guide](CONTRIBUTING.md) for tests and the source layout.
Exex is derived from [LXM3](https://github.com/ethanluoyc/lxm3) and includes
vendored XManager components; see [attribution](NOTICE.md).
