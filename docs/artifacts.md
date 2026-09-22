# Retain and reuse outputs

Declare named outputs relative to `EXEX_OUTPUT_DIR`:

```python
experiment.add(
    xm.Job(executable, executor),
    outputs={"checkpoint": "checkpoint", "metrics": "metrics.json"},
)
```

The worker writes there using ordinary filesystem APIs. After successful exit,
exex captures the declared files/directories into immutable, content-addressed
artifacts. Missing outputs or capture errors fail the job; a process exit of
zero is not sufficient by itself.

Retrieve them after reopening:

```python
experiment = xc.get_experiment(experiment_id)
unit = experiment.work_units()[work_unit_id]
checkpoint = unit.artifacts()["checkpoint"]
checkpoint.fetch("/new/local/checkpoint")
```

The destination must not already exist. The retained artifact is independent of
the temporary application work directory, but still depends on the execution
site's storage retention.

Bind it to a new job:

```python
experiment.add(
    xm.Job(next_executable, next_executor),
    inputs={"checkpoint": checkpoint},
)
```

The worker reads `EXEX_INPUT_DIR/checkpoint`. Each task receives its own verified
copy. Same-site inputs reuse retained storage; cross-site inputs are transferred
through the author host before submission. This requires connectivity and enough
temporary disk on the author host. Exex does not infer dataset transfers or
change an application's checkpoint format.

For arrays, select a zero-based task with `unit.artifacts(task=...)`. For managed
continuation, `artifacts()` exposes the latest successfully retained set; an
earlier checkpoint survives a later failed attempt. Publication to external
registries is not implemented.
