"""Tests for sky/serve/replica_managers.py.

Currently focused on `SkyPilotReplicaManager.__init__` startup ordering:
the daemon threads (especially `_job_status_fetcher`) must NOT race the
main thread for `self.lock` before `_recover_replica_operations` runs.
"""
from unittest import mock

from sky.serve import replica_managers


class TestSkyPilotReplicaManagerInitOrdering:
    """`SkyPilotReplicaManager.__init__` must run `_recover_replica_operations`
    BEFORE starting the `_job_status_fetcher` / `_thread_pool_refresher` /
    `_replica_prober` daemon threads.

    If the daemon threads start first, `_job_status_fetcher` will acquire
    `self.lock` (via the `@with_lock` decorator on `_fetch_job_status`)
    and perform a per-replica SSH/gRPC call to query job status. When a
    replica's head node is unreachable (pod / VM gone), each SSH connect
    hangs at the kernel TCP timeout (tens of seconds to minutes). The
    main thread then blocks on `_recover_replica_operations`'s
    `with self.lock:` for the full hang duration, never returns from
    `SkyPilotReplicaManager.__init__`, and `uvicorn.run` is never called.

    With HA recovery changes, `_wait_for_controller_ready`
    then times out (60s) → `_bail_on_boot_failure` → `os._exit(1)` →
    daemon retries → same race → infinite recovery loop.

    The fix: recovery first, daemon threads after.
    """

    def test_recover_called_before_threads_start(self):
        """Verify the call order: `_recover_replica_operations` first,
        then each daemon thread's `.start()`."""
        call_order = []

        def _record(name):

            def _fn(*_args, **_kwargs):
                call_order.append(name)

            return _fn

        # Patch the heavy deps so __init__ doesn't actually do work.
        # We only care about the call order.
        with mock.patch.object(
                replica_managers.ReplicaManager, '__init__',
                return_value=None), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state.get_yaml_content',
                 return_value='dummy: yaml'), \
             mock.patch(
                 'sky.serve.replica_managers.task_lib.Task.from_yaml_str',
                 return_value=mock.MagicMock()), \
             mock.patch(
                 'sky.serve.replica_managers.spot_placer.SpotPlacer.from_task',
                 return_value=None), \
             mock.patch.object(
                 replica_managers.SkyPilotReplicaManager,
                 '_recover_replica_operations',
                 _record('recover')), \
             mock.patch(
                 'sky.serve.replica_managers.thread_utils'
                 '.start_supervised_thread') as mock_supervised:
            # start_supervised_thread starts the supervisor thread
            # immediately, so its call time IS the thread start time.
            def _start_supervised(target, *_args, **_kwargs):
                target_name = getattr(target, '__name__', repr(target))
                call_order.append(f'thread_start:{target_name}')
                return mock.Mock()

            mock_supervised.side_effect = _start_supervised

            spec = mock.MagicMock()
            replica_managers.SkyPilotReplicaManager(service_name='svc',
                                                    spec=spec,
                                                    version=1)

        # `recover` must come before any `thread_start:*` entry. The
        # daemon threads themselves may be created in any order relative
        # to each other (we don't constrain that), but ALL of them must
        # appear after `recover`.
        assert 'recover' in call_order, (
            f'_recover_replica_operations was never called; '
            f'call_order={call_order}')
        recover_idx = call_order.index('recover')
        for i, name in enumerate(call_order):
            if name.startswith('thread_start:'):
                assert i > recover_idx, (
                    f'{name} happened at index {i} before recover at '
                    f'index {recover_idx}; call_order={call_order}. '
                    f'Daemon threads must NOT start until '
                    f'_recover_replica_operations has finished — '
                    f'see the docstring of '
                    f'TestSkyPilotReplicaManagerInitOrdering.')

    def test_all_three_daemon_threads_are_started(self):
        """Sanity: regardless of ordering, the three control-loop threads
        (_thread_pool_refresher / _job_status_fetcher / _replica_prober)
        still all start (supervised)."""
        started_targets = []

        with mock.patch.object(
                replica_managers.ReplicaManager, '__init__',
                return_value=None), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state.get_yaml_content',
                 return_value='dummy: yaml'), \
             mock.patch(
                 'sky.serve.replica_managers.task_lib.Task.from_yaml_str',
                 return_value=mock.MagicMock()), \
             mock.patch(
                 'sky.serve.replica_managers.spot_placer.SpotPlacer.from_task',
                 return_value=None), \
             mock.patch.object(
                 replica_managers.SkyPilotReplicaManager,
                 '_recover_replica_operations'), \
             mock.patch(
                 'sky.serve.replica_managers.thread_utils'
                 '.start_supervised_thread') as mock_supervised:

            def _start_supervised(target, *_args, **_kwargs):
                started_targets.append(getattr(target, '__name__', None))
                return mock.Mock()

            mock_supervised.side_effect = _start_supervised

            spec = mock.MagicMock()
            replica_managers.SkyPilotReplicaManager(service_name='svc',
                                                    spec=spec,
                                                    version=1)

        # Bound methods on the instance — verify by name.
        assert '_thread_pool_refresher' in started_targets
        assert '_job_status_fetcher' in started_targets
        assert '_replica_prober' in started_targets
