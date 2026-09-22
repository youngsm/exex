# Prebuilt SIF and separately packaged source

This probe reads `value.txt` from an input directory and writes `result.json` to
an output directory. It uses PyTorch for a CUDA matrix product; exex itself does
not depend on PyTorch. The existing image and source archive are staged separately.

Prepare `value.txt` containing `3` and a [cluster configuration](../../docs/configuration.md), then run:

```sh
EXEX_CONFIG=/path/to/exex.toml exex launch examples/sif/launch.py -- \
  --cluster=s3df --image=/shared/runtime.sif \
  --input_dir=/shared/probe/input --output_dir=/shared/probe/result-001 \
  --workdir_root=/shared/probe/work \
  --resource=account=my-account --resource=partition=ampere
```

The launcher requests one GPU, two CPUs, 8 GiB memory and three minutes. The SIF
must be readable by the authoring host. This example creates its output directory
locally, so its input/output paths must be shared with the execution host. Use a
new output directory each time. Add `--fail` to test failure propagation; use
`--source_dir` with edited copies of `build.sh` and `probe.py` to test source-only
changes without rebuilding the image.

Historical qualification evidence is retained in
[the development archive](../../docs/development/archive/sif-qualification.md).
