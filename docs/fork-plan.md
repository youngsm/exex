# LXM3 fork: implementation delta

Status: minimal HPC launcher slice, 2026-09-20. Fork: [youngsm/lxm3](https://github.com/youngsm/lxm3).
Upstream: [ethanluoyc/lxm3](https://github.com/ethanluoyc/lxm3), commit
`30f55855e595ee7d8397daacb1475442265f37ef`. The
[API proposal](fork-api-proposal.md) defines the broader method-level changes.
The [minimal HPC slice](minimal-hpc-slice.md) supersedes its release gate: everything
beyond that boundary is deferred, not an implicit requirement for this patch.
Existing exex and pimm code, catalogs and live workflows remain intact.

## Implementation progress

The first implementation patch covers explicit-site routing and experiment-scoped
packaging. It is a subset of section 1, not completion of the entire roadmap.

- `Slurm(cluster=...)` and `GridEngine(cluster=...)` carry their selection into
  instance `Spec()` calls. Local's class/instance `Spec()` calls remain unchanged.
- `create_experiment(..., config=...)` snapshots config, environment defaults and
  local/on-site staging paths. Explicit project names do not mutate shared config.
- Each Experiment owns its packaging queue. Packaging, nested image caching and
  submission receive explicit settings/project; execution clients are not globally
  cached. The artifact-store factory uses its supplied arguments.
- Packaged AppBundles record their backend/staging destination. Submission to a
  different site, project, backend or changed staging profile raises before creating
  clients. Existing manually constructed AppBundles remain supported.
- No author database is introduced in this patch. The private packaging `store`
  parameter from the full proposal waits for the retained-source/catalog slice.

The second patch completes the narrow execution/staging work without adding public
API. Slurm commands and transfers use system OpenSSH; GridEngine retains its
upstream transport. Generated arguments, environment and mounts preserve literal
values; GPU flags follow the resource request. Local failures propagate, contexts
close on error, and local logs persist. Package files use content-derived names and
are uploaded to temporary paths before final exposure.

Regression tests execute generated scripts and exercise transfer/submission errors;
live qualification covers local execution and S3DF/NERSC paths. The
[scope and evidence record](minimal-hpc-slice.md) distinguishes scheduler acceptance
from completed workload evidence. No container builds, image pulls or training runs
are part of this slice. Source freezing, reopening/lifecycle APIs, retained outputs,
continuation, new runtimes/backends and pimm migration remain deferred.

The shared-storage multi-node path is now qualified without a new step API:
`Slurm(workdir_root=...)` unpacks once, and an ordinary packaged driver launches
workers with native `srun`. NERSC jobs `58674695` and `58674935` proved distinct
nodes, shared source, successful completion and failed-worker propagation. See
the [Slurm step example](../examples/slurm_step/README.md). Container execution
and GPU collectives are separate gates, not implied by this host-only result.

The prebuilt-container slice adds no public API. The [SIF example](../examples/sif/README.md)
qualified real GPU computation, input/output mounts, failure propagation and
source-only image reuse on S3DF. Its one-line library fix preserves Slurm's
`CUDA_VISIBLE_DEVICES` in the container environment file. The independent
[native Shifter example](shifter-slice.md) qualified a pinned NERSC image, read-only
input, task-level GPU visibility, computation and intentional failure, submitted
from S3DF. Both retain outputs and clean up temporary source. Combined regressions:
270 passed, 2 integration tests deselected. No first-class Shifter runtime, image
builder, multi-node container collective or pimm migration is claimed by these probes.

## Decision and scope

Extend LXM3, not exex under a different name. Keep its
`Experiment -> WorkUnit -> Job` model, `xm.Packageable -> AppBundle` packaging,
Python launchers, TOML site settings, arrays and optional W&B helper.
The existing WorkUnit is the logical research run; attempts are records beneath
it. There is no second public Run/Project/Executable hierarchy and no dependency
on exex at runtime.

The original commitments remain: dirty-source iteration, explicit Local/S3DF/
NERSC/Vertex targets, inspectable execution, retained outputs, checkpoint reuse,
publication and framework independence. Their implementation now follows LXM3's
extension points. The exex roadmap is a requirements/evidence reference, not a
list of modules to transplant or a parallel implementation backlog.

Keep the `lxm3` name and inherited history for now. Preserve copyright headers and
the vendored XManager license. Do not rename APIs, prune unrelated backends, upgrade
vendored XManager, change repository visibility or choose a new license in this work.

## Keep, extend, add

| Area | Decision | Concrete implementation seam |
| --- | --- | --- |
| Launch API and CLI | Keep async/sync contexts, JobGenerators, `xm.Job`, `ArrayJob`, `lxm3 launch`. Add inspection commands, not a second launch CLI. | `xm_cluster/experiment.py`, `cli/cli.py` |
| Python/source packaging | Keep `PythonPackage`, `UniversalPackage`, Fileset and AppBundle. Add raw-source capture and retained re-preparation. | `executable_specs.py`, `packaging/create_archive.py`, `packaging/router.py` |
| Containers | Qualify prebuilt Singularity and native Shifter composition first; retain source/dependency separation. Typed runtime additions and image builds are separate reviews. | `executable_specs.py`, `executors.py`, `execution/job_script_builder.py`, examples |
| Site configuration | Keep TOML. Make selected site/project explicit throughout packaging and dispatch. | `config.py`, executor specs, cached clients and artifact-store factory |
| SSH | Use system OpenSSH configuration for commands and transfers to the same alias; retain the storage interface. | `clusters/slurm.py`, `execution/job_script_builder.py`, `artifacts.py` |
| Persistent control | Add a small SQLite store and implement existing XM WorkUnit lifecycle methods. | `experiment.py`, `metadata.py`, native execution handles |
| Retained results | Extend the existing storage path with content identity, output capture, input reuse and publication receipts. | `artifacts.py`, job-script completion path, WorkUnit accessors |
| W&B | Keep `contrib.wandb.configure_wandb`; add actual run-link reporting and provenance. | `contrib/wandb.py`, WorkUnit metadata, application writer |
| Allocation ownership | Keep native task launch through entrypoints; borrowed-step and attached-allocation APIs remain separate future work. | `executors.py`, `execution/slurm.py`, `clusters/slurm.py` |
| Vertex | Add a native executor and durable object staging; it is not present in the vendored XM copy. | executor/spec, packaging route, execution adapter |
| pimm | Adapt authoring to XM jobs; keep training, checkpoint and logger semantics application-owned. | pimm integration, changed only with scoped approval |

Paths above are under `lxm3/`; packaging/execution/config paths are under
`lxm3/xm_cluster/` unless stated otherwise. Exact changes and gates follow.

## 1. Correct explicit-site routing and existing execution defects

This is the first slice, before adding more backends.

- Carry a selected cluster on `Slurm` and `SlurmSpec`. An executor's packaging
  spec must carry the same selection as dispatch; the current classmethod-only
  `Spec()` and empty SlurmSpec cannot do this.
- Resolve TOML settings and project explicitly. Remove zero-argument client caching
  or key it by resolved settings/project. Fix `create_artifact_store(project,
  settings)`, which currently discards both arguments, and its reversed callers.
  Two experiments/sites must not redirect each other through global configuration.
- Use one OpenSSH alias/configuration for command execution and staging/fetch.
  Keep direct subprocess execution for on-site submission. Use parsable native job
  IDs, correct quoting and ordinary subprocess errors, without a remote catalog,
  RPC service, lost-response search or submission retries.
- Fix narrow upstream defects with regressions: local child exit codes are ignored;
  bind/volume mappings are iterated without `.items()`; Slurm assumes every job
  needs GPU mode; generated arguments/environment need literal round-trip tests.

Gate: existing launchers still work; two sites and two projects in one process
cannot cross-route; nonzero children fail; spaces/quotes/dollar signs survive;
one SSH/submission error never causes a second submission. Test without live jobs
first, then bounded S3DF/NERSC probes.

## 2. Retain source and make prepared payload identity reliable

- Add raw `SourceTree` and a retained frozen-source value to `executable_specs.py`;
  export them and route them to the existing AppBundle. Preserve relative files,
  executable bits and explicit selection without requiring `pip install`.
- Capture dirty/new/deleted files without touching Git. Keep a target-independent
  content identity and manifest; the same capture can prepare another target after
  the original checkout changes or disappears. Git commit is provenance, not identity.
- Preserve `package_async` queue semantics. It queues specifications; capture occurs
  before build/upload when packaging runs. An explicit freeze operation gives the
  caller an earlier capture boundary. Do not rewrite AsyncPackager to hide this.
- Extend the existing archive/upload path with content-derived names, temporary
  transfer then final exposure, and verification. Size/mtime is not integrity proof;
  do not use the existing stat-keyed digest cache as one.
- Retain prepared package/source references for reopen, discovery and source/spec
  comparisons. Ordinary PythonPackage and UniversalPackage remain supported.

Gate: unchanged Git HEAD/index; identical selected bytes have identical identity
across checkout paths; post-capture mutations cannot change execution; interrupted/
corrupt transfers do not become executable packages. Re-prepare for another site
without the checkout. Reuse existing capture/verification algorithms only where useful.

## 3. Implement durable Experiment/WorkUnit control

- Add one author-side SQLite store for experiments, integer WorkUnit IDs, concrete
  invocation intent, attempts, provider handles and metadata. Transactions stay
  short; workers do not open it. No schema counters or compatibility framework.
  Any compatibility boundary uses the installed fork package version.
- Implement retrieval through `xm_cluster.get_experiment(...)`, recorded
  `work_units()`, and the existing `get_status()`, `stop()` and completion-wait methods.
  Add log access and CLI retrieval/status/logs/stop. Preserve context-manager and
  awaitable `add()` behavior, and persist `context.annotations` rather than returning
  a fresh empty metadata object each time.
- Implement keyed `add(..., identity=...)`: unchanged concrete intent reconnects;
  changed intent conflicts; no key means an intentional new WorkUnit. Comparing
  intent is an intentional strengthening of XM's existing add-if-present contract.
  Allocate identities before dispatch; save accepted native handles. Submission
  errors propagate, including lost acknowledgments, without discovery/resubmission.
  A failed submission without an accepted handle remains a recorded error;
  re-adding its identity must not implicitly submit again.
- Make identity conflicts non-destructive: XM's generic add-exception path calls
  `stop()`. Re-adding a conflicting request must never cancel an existing WorkUnit.
  Reopened units must not reserve a new ID or reuse an in-memory predictor sequence.
- Persist concrete jobs, never pickle generators. XM allows generators to run again
  on keyed add; deduplicate the resulting intent. Reopen/status/manual continuation
  must not rerun the launcher or a Python closure.
- Preserve ArrayJob as one WorkUnit with task-indexed execution and output state;
  aggregate actual outcomes. Preserve broadcasting, task offsets and singleton
  JobGroups. Multi-job groups are already unsupported; do not add a DAG engine.

Gate: submit, exit, reopen elsewhere on the author host, inspect/log/stop, and add
another unit. Concurrent identical keys submit once; conflicts leave original work
alone. Missing scheduler evidence is unknown, not success. Native job references
include site and execution generation so reused numbers cannot cross-cancel.

Keep Local's existing attached/wait-on-exit default. Detached local execution is
an explicit additional mode with durable logs/completion and safe process ownership,
not a replacement for Local or a requirement imposed on ordinary launchers.

## 4. Retain outputs, consume inputs and publish artifacts

- Extend existing ArtifactStore transport operations, rather than introduce a second
  storage engine. Add immutable content records, verified staging, completed-output
  capture and WorkUnit artifact retrieval. Declare inputs/outputs through the backend
  job/work-unit path; do not require a parallel public Job hierarchy.
- Capture before temporary working directories are removed. Required output capture
  failure prevents successful completion. Workers write site-local receipts, not the
  author database. Keep producer, job/array-task and attempt identity on each output.
- Reuse same-site content directly; fetch across sites explicitly initially. Normal
  outputs are retained without a user publish call. Preserve cancellation evidence
  and already completed artifacts; never copy a moving checkpoint as a completed one.
- Add `publish`/external-link support to retained artifact handles: local directory
  first, optional `huggingface_hub` publisher next. Record complete immutable remote
  revisions; respect explicit destination/privacy. Model export stays in applications.

Gate: a producer exits, its original outputs are removed, and a new process retrieves
and feeds verified retained content to a consumer. Cover corruption, interrupted
publication, task-index isolation and missing required output. HF publication is
independent of scheduling and requires separately authorized destinations.

## 5. Extend existing runtimes, not rebuild packaging

- Start with prebuilt images and the existing interfaces: `SingularityContainer`
  for an author-readable SIF; native Slurm `image`/`module` resources and a packaged
  `srun`/Shifter driver for NERSC. Resolve existing Shifter images to their native
  IDs before submission; do not equate those IDs with OCI digests. Keep the driver
  outside the container and resolve GPU visibility at the native task boundary.
  A first-class `ShifterContainer` and automatic preparation/import are separate
  proposals, not prerequisites for this qualification. Execution-site SIF paths
  must not be confused with author-local `SingularityContainer.image_path`.
- Keep actual SIF hashes and immutable cache blob paths, not mutable tag symlinks.
  Resolve mutable references during preparation. Reuse prepared site images rather
  than importing/converting in each rank or implementing another Shifter cache.
- Preserve LXM3 `PythonContainer` semantics: dependency image plus separately packaged
  source. Forward its and PDMProject's currently dropped `resources`, `extra_packages`
  and `pip_args`. Do not change it into XManager's full application-image abstraction.
- Extend the existing Docker build path with explicit builder/platform/output.
  A daemon consumer needs a loaded image; a remote consumer needs a pushed pinned
  reference. Current buildx invocation supplies neither `--load` nor `--push`.
  No silent builder fallback or publication. Frozen context and actual output identity
  must be recorded. Reuse existing caches and external builders.

Gate: source-only changes reuse dependency images, changed dependencies select a new
runtime, mutable tags cannot change prepared jobs, non-empty mounts work, and the
real S3DF/NERSC image paths are qualified. Existing compatible pinned images suffice
for initial HPC training; general builders are not a prerequisite.

## 6. Qualify distributed Slurm; add allocation ownership separately

- Native node/task resources describe the allocation; the batch entrypoint runs
  once. With an explicitly shared `workdir_root`, that entrypoint can use `srun`
  to launch workers against the single unpacked source tree. This path is qualified;
  no `SlurmStep`, automatic worker replication or per-node extraction is required.
- Applications own torchrun, TF_CONFIG, JAX or MPI setup and consume native topology.
  Qualify container placement and GPU collectives separately; a container around
  the batch driver is not automatically a container around each worker. Node-local
  source distribution remains outside the shared-storage contract.
- Add borrowed-step execution and an attached owned `salloc` context. Batch owns its
  allocation; a borrowed step never cancels its parent/siblings; context exit releases
  only its owned allocation. Keep native `srun`/`salloc` lifetimes explicit.
- Add native requeue/walltime/preemption continuation only after section 7 qualifies
  application readiness. A reused Slurm ID is not an attempt identity. Durable NERSC
  allocation chaining is conditional on native mechanisms failing the real workflow,
  and needs explicit approval, bounded ownership and cancellation of future slots.

Gate: the two-node host probe has proved distinct ranks and failed-rank propagation;
bounded GPU collective/training tests remain. Step cancellation leaves parent/siblings alive;
owned sessions release their allocations. No standing allocation, placement resolver
or implicit scron/supervisor installation.

## 7. Add checkpoint continuation; keep scientific policy in pimm

- A small execution context supplies restored-input paths, startup acknowledgment,
  immutable checkpoint publication and cooperative pause. Add explicit
  `ClusterWorkUnit.resume()` for a new attempt of the same frozen concrete invocation.
  Only a verified cooperative pause permits it; cancellation/budgets survive reopen.
- Qualify singleton jobs first. Reject array continuation until per-element resume
  semantics exist; never replay successful array elements implicitly. Failure/timeout
  without a valid pause remains failure/timeout, not automatic restart.
- Adapt pimm's lightweight authoring to LXM3 specs/jobs. Reuse its approved pause,
  checkpoint and RNG work; do not duplicate or discard it. Weights-only initialization
  and WSD/LR/beta branches are new WorkUnits consuming parent artifacts, with explicit
  restored state, optimizer-moment/counter treatment and effective configuration.
- The existing step-five trajectory divergence remains an open application gate.
  Forking the launcher does not resolve it. Qualify actual worker/AMP/hook/topology
  settings separately; do not silently widen numerical tolerance.

Gate: a non-ML counter pauses, reopens and resumes under one WorkUnit with distinct
attempts; then pimm proves declared state/history behavior. Real NERSC fresh training
precedes NERSC continuation and chain parity. Systems-level pimm edits stay separately
scoped and approved; the active checkout is not reset or swept into a commit.

## 8. Complete the existing W&B integration

- Keep `configure_wandb` and its Job/ArrayJob generator pattern. Preserve original job
  names and backend metadata when wrapping; the current implementation drops names.
  Test precedence and avoid forcing IDs that conflict with checkpoint logger state.
- Add generic execution-side URL reporting stored on the owning WorkUnit/attempt.
  Pimm reports its actual URL after initialization; no second logger or core W&B SDK.
- Record actual source/runtime identity, not a dirty-source hash presented as a Git
  commit. Qualify execution-site credential-file references and container visibility;
  do not serialize secret values. Test online, offline and restored history explicitly.

Gate: useful names, identity/links after reconnect, correct resume/branch history and
no tracking requirement when disabled. Logging policy remains application-owned.

## 9. Add Vertex without importing another framework

LXM3's vendored XM excludes cloud/xm_local; Vertex is new adapter work, not an enabled
option. Inspect XManager's provider translations for selective reuse, not its database,
registry and executable hierarchy as a second runtime system.

- Add an explicit Vertex executor/spec, packaging route and native handles under the
  existing Experiment/WorkUnit API. Use a prebuilt dependency image first; stage the
  separate source archive and inputs durably and bootstrap them at execution.
- Add object-storage inputs, retained outputs/receipts and provider identity/network/
  resource settings. Verify the actual worker/service credentials and restart/output
  behavior. Disable automatic create retries; lost acknowledgments raise errors.
- Qualify CPU, one GPU, then two-node coordinated execution/failure/completion. Include
  actual small TensorFlow/JAX examples before claiming equal distributed support.
  Keep SDKs optional; no GCP Batch, second tracking service or automatic Spot recovery.

Gate: launcher exit does not lose logs/results; cancel/reconnect works; outputs are
complete before the primary exits. The same non-ML application works on both Slurm
sites and Vertex. Full builders, HF and HPC chaining are not prerequisites.

## Delivery order and reuse boundaries

1. **Baseline:** fork unchanged upstream, isolated non-live tests, then commit this plan.
2. **Useful fork:** sections 1–3, narrowly sliced: explicit-site routing/correct local
   failures; retained source; persistent WorkUnits and native control. One installed
   launcher submits on S3DF, exits, reopens and adds a seed without rereading source.
3. **Cross-site research:** SSH/NERSC, existing SIF/new Shifter, retained outputs,
   W&B links and fresh pimm training. No full image builder prerequisite.
4. **Checkpoint and ownership parity:** manual continuation, scientific branches,
   distributed topology, borrowed steps/salloc, then qualified allocation-boundary
   continuation. Build/publish and Vertex are independently deliverable tracks.
5. **Migration:** retire old pimm/exex-backed paths only after per-workflow parity and
   explicit approval. HPC retirement does not wait for cloud; keep old evidence/data.

Reuse exex's proven capture/publication algorithms, native command details and
behavioral tests where they simplify these changes. Do not copy its catalog schema,
public Run model, worker/RPC hierarchy or entire modules wholesale. Preserve upstream
Local/Slurm/SGE launch regressions, async generators, arrays and singleton groups;
new lifecycle support is claimed only for individually qualified backends.

Each slice needs tests, a runnable example and an honest capability record. Keep
changes small, errors ordinary and validation limited to consequential boundaries.
No implementation is claimed by this plan; qualification results belong in the
[fork baseline record](fork-baseline.md) and subsequent evidence documents.
