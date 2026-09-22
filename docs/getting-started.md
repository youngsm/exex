# Your first experiment

Install exex from the repository, then save these two files in the same directory.
The worker uses only Python's standard library.

`worker.py`:

```python
import os
from pathlib import Path

(Path(os.environ["EXEX_OUTPUT_DIR"]) / "result.txt").write_text("hello\n")
print("Finished", flush=True)
```

`launch.py`:

```python
from pathlib import Path
from exex import xm, xm_cluster as xc

def main(_):
    executor = xc.Local()
    source = xc.SourceTree(
        xc.ModuleName("worker"), Path(__file__).parent, files=["worker.py"]
    )
    with xc.create_experiment("hello") as experiment:
        [executable] = experiment.package([xm.Packageable(source, executor.Spec())])
        experiment.add(
            xm.Job(executable, executor), outputs={"result": "result.txt"}
        )
    print(experiment.experiment_id)
```

Run `exex launch launch.py`. Local context exit waits for the job; nonzero
worker exits propagate. Source is unpacked in a temporary directory. Retained
outputs survive its removal. The default local catalog/staging root is `.exex`
in the author's working directory; keep it somewhere persistent.

To run on a configured cluster, replace the executor with:

```python
executor = xc.Slurm(
    cluster="mycluster", resources={"account": "my-account"}, walltime=300
)
```

The batch launcher returns after submission. Submission success is not workload
completion: use the printed experiment ID with `exex status EXPERIMENT_ID`.

The repository's `examples/quickstart/launch.py` accepts `--cluster`, repeated
`--resource=key=value`, and `--walltime` so the same example works on either target.
No scheduler resources are selected automatically. Dependencies and datasets must
already be available in the selected execution environment.
