# Weights & Biases

Install the optional integration with `pip install -e '.[wandb]'`. Credentials
must be available in the worker environment; do not put API keys in recorded
job arguments or environment dictionaries.

The extra pins W&B 0.28.0, the qualified SDK version. Upgrading the SDK is a
separate integration change; later releases removed APIs used by this helper.

The helper returns a native W&B Run and records its URL as a task link. Your
application retains responsibility for metric logging, rank-zero ownership,
checkpoint timing, and closing the run. Exex does not install training hooks.

See the public helper signatures in the API reference and the runnable
`examples/wandb/` examples. The three history policies are:

| Policy | Meaning |
| --- | --- |
| `new` | Start a new W&B run, optionally linked to checkpoint lineage |
| `append` | Continue an existing run without deleting server history |
| `fork` | Branch from checkpointed history; requires W&B permission |

Rewind is unsupported. Exex never silently falls back from fork to another
policy. Checkpoint the helper's tracking state together with application state;
execution continuation and W&B history selection are separate decisions.

`unit.get_links()` retrieves reported URLs while running or after reopening.
Any application can also call `exex.execution.link(name, url)` to record another
dashboard or external artifact URL. A link is metadata, not an artifact upload.
