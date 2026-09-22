API reference
=============

Import ``xm`` for the job vocabulary and ``xm_cluster`` for execution:

.. code-block:: python

   from exex import xm, xm_cluster as xc

Job vocabulary
--------------

``xm.Job(executable, executor, args=None, env_vars=None)`` binds a packaged
executable to an executor. ``xm.Packageable(spec, executor.Spec())`` is a packaging
request. ``xm.ShellSafeArg`` opts into shell expansion; ordinary arguments are quoted.

Experiments and WorkUnits
----------------------------------------

.. autofunction:: exex.xm_cluster.create_experiment
.. autofunction:: exex.xm_cluster.get_experiment
.. autofunction:: exex.xm_cluster.list_experiments

``Experiment.package(requests)`` returns packaged executables. ``freeze(source)``
retains a source snapshot. ``add(job, *, inputs=None, outputs=None,
continuation=None)`` schedules a WorkUnit; ``work_units()`` and ``sources()``
retrieve recorded work. In asynchronous launchers, await the result of ``add``.

WorkUnits expose ``get_status()``, ``get_logs(task=None, tail=200)``,
``get_links(task=None)``, ``get_script()``, ``artifacts(task=None)``, ``stop()``
and ``wait_until_complete()``. ``job`` and ``source`` expose saved provenance.
See the inspection and artifact guides for failure and retention semantics.

Executors and continuation
----------------------------------------

.. autoclass:: exex.xm_cluster.Local
.. autoclass:: exex.xm_cluster.Slurm
.. autoclass:: exex.xm_cluster.Continuation

Packaging
---------

.. autoclass:: exex.xm_cluster.SourceTree
.. autoclass:: exex.xm_cluster.FrozenSource
.. autoclass:: exex.xm_cluster.SingularityContainer
.. autoclass:: exex.xm_cluster.ShifterContainer

Existing alternatives include ``PythonPackage``, ``PythonContainer``,
``UniversalPackage``, ``PexBinary``, and ``DockerContainer``. Container executor
options use ``SingularityOptions``, ``ShifterOptions``, or ``DockerOptions``.

Worker helpers
--------------

.. autofunction:: exex.execution.pause_requested
.. autofunction:: exex.execution.mark_paused
.. autofunction:: exex.execution.link

Weights & Biases
----------------------------------------

.. autofunction:: exex.contrib.wandb.init
.. autofunction:: exex.contrib.wandb.checkpoint_state
.. autofunction:: exex.contrib.wandb.attach
.. autofunction:: exex.contrib.wandb.configure_wandb
