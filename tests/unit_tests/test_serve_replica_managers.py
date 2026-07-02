"""Tests for sky/serve/replica_managers.py.

Covers:
- `SkyPilotReplicaManager.__init__` startup ordering: the daemon threads
  (especially `_job_status_fetcher`) must NOT race the main thread for
  `self.lock` before `_recover_replica_operations` runs.
- `launch_cluster` retry scoping: resource availability (capacity) failures
  are capped by `availability_max_retry` while other errors keep the
  `max_retry` in-place attempts.
- `_launch_replica` passing `availability_max_retry=1` only for spot
  replicas managed by a spot placer.
"""
from unittest import mock

from sky import exceptions
from sky.serve import replica_managers
from sky.utils import thread_utils


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
                 'sky.serve.replica_managers.threading.Thread') as mock_thread:
            # Each Thread(target=...).start() records the target's name
            # via our side_effect on .start().
            def thread_factory(*_args, **kwargs):
                target = kwargs.get('target')
                t = mock.Mock()
                target_name = getattr(target, '__name__', repr(target))
                t.start.side_effect = _record(f'thread_start:{target_name}')
                return t

            mock_thread.side_effect = thread_factory

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
        """Sanity: regardless of ordering, the three daemon threads
        (_thread_pool_refresher / _job_status_fetcher / _replica_prober)
        still all start. The fix is purely a reorder, not a removal."""
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
                 'sky.serve.replica_managers.threading.Thread') as mock_thread:

            def thread_factory(*_args, **kwargs):
                target = kwargs.get('target')
                started_targets.append(getattr(target, '__name__', None))
                t = mock.Mock()
                return t

            mock_thread.side_effect = thread_factory

            spec = mock.MagicMock()
            replica_managers.SkyPilotReplicaManager(service_name='svc',
                                                    spec=spec,
                                                    version=1)

        # Bound methods on the instance — verify by name.
        assert '_thread_pool_refresher' in started_targets
        assert '_job_status_fetcher' in started_targets
        assert '_replica_prober' in started_targets


def _capacity_error() -> exceptions.ResourcesUnavailableError:
    return exceptions.ResourcesUnavailableError(
        'no capacity',
        failover_history=[
            exceptions.ResourcesUnavailableError('zone exhausted')
        ])


class TestLaunchClusterRetry:
    """`launch_cluster` must fail fast ONLY on resource availability
    (capacity) failures when `availability_max_retry` caps them; other
    (transient) errors must keep the `max_retry` in-place attempts."""

    def _run_launch_cluster(self, tmp_path, stream_side_effects, **kwargs):
        """Run launch_cluster with a mocked SDK.

        Each element of stream_side_effects is one launch attempt: an
        exception to raise from sdk.stream_and_get, or None for success.
        Returns (mock_sdk, mock_terminate, raised RuntimeError or None).
        """
        raised = None
        with mock.patch(
                'sky.serve.replica_managers.task_lib.Task.from_yaml_str',
                return_value=mock.MagicMock()), \
             mock.patch('sky.serve.replica_managers.usage_lib'), \
             mock.patch('sky.serve.replica_managers.sdk') as mock_sdk, \
             mock.patch('sky.serve.replica_managers.terminate_cluster'
                       ) as mock_terminate, \
             mock.patch('sky.serve.replica_managers.common_utils.Backoff'
                       ) as mock_backoff:
            # Skip the (up to 60s) backoff between attempts.
            mock_backoff.return_value.current_backoff.return_value = 0
            mock_sdk.launch.return_value = 'request-id'
            mock_sdk.stream_and_get.side_effect = stream_side_effects
            try:
                replica_managers.launch_cluster(
                    replica_id=1,
                    yaml_content='dummy: yaml',
                    cluster_name='svc-1',
                    log_file=str(tmp_path / 'launch.log'),
                    replica_to_request_id=thread_utils.ThreadSafeDict(),
                    replica_to_launch_cancelled=thread_utils.ThreadSafeDict(),
                    **kwargs)
            except RuntimeError as e:
                raised = e
        return mock_sdk, mock_terminate, raised

    def test_capacity_failure_fails_fast_with_availability_max_retry(
            self, tmp_path):
        """One capacity failure with availability_max_retry=1 must raise
        immediately (no in-place retry of the same exhausted location)."""
        mock_sdk, mock_terminate, raised = self._run_launch_cluster(
            tmp_path, [_capacity_error()] * 3, availability_max_retry=1)
        assert raised is not None
        assert 'after 1 attempt' in str(raised)
        assert mock_sdk.launch.call_count == 1
        assert mock_terminate.call_count == 1

    def test_transient_failures_keep_in_place_retries(self, tmp_path):
        """Transient (non-availability) errors must still be retried in
        place even when availability_max_retry=1, so a one-off blip does
        not poison the placer location."""
        mock_sdk, mock_terminate, raised = self._run_launch_cluster(
            tmp_path,
            [RuntimeError('transient'),
             RuntimeError('transient'), None],
            availability_max_retry=1)
        assert raised is None
        assert mock_sdk.launch.call_count == 3
        assert mock_sdk.stream_and_get.call_count == 3
        assert mock_terminate.call_count == 2

    def test_capacity_failures_default_to_max_retry(self, tmp_path):
        """Without availability_max_retry, capacity failures keep the
        default max_retry in-place attempts."""
        mock_sdk, _, raised = self._run_launch_cluster(tmp_path,
                                                       [_capacity_error()] * 3)
        assert raised is not None
        assert 'after 3 attempt' in str(raised)
        assert mock_sdk.launch.call_count == 3

    def test_capacity_failure_fails_fast_after_transient_failure(
            self, tmp_path):
        """A capacity failure must propagate immediately even if preceded
        by a transient failure."""
        mock_sdk, _, raised = self._run_launch_cluster(
            tmp_path, [RuntimeError('transient'),
                       _capacity_error()],
            availability_max_retry=1)
        assert raised is not None
        assert 'after 2 attempt' in str(raised)
        assert mock_sdk.launch.call_count == 2


class TestLaunchReplicaAvailabilityMaxRetry:
    """`_launch_replica` must cap availability failures at one attempt only
    for spot replicas managed by a spot placer."""

    def _launch_replica(self, use_spot: bool, with_placer: bool):
        # pylint: disable=protected-access
        manager = replica_managers.SkyPilotReplicaManager.__new__(
            replica_managers.SkyPilotReplicaManager)
        manager._service_name = 'svc'
        manager.yaml_content = 'dummy: yaml'
        manager.latest_version = 1
        manager._launch_thread_pool = thread_utils.ThreadSafeDict()
        manager._replica_to_request_id = thread_utils.ThreadSafeDict()
        manager._replica_to_launch_cancelled = thread_utils.ThreadSafeDict()
        placer = None
        if with_placer:
            placer = mock.Mock()
            location = mock.Mock()
            location.to_dict.return_value = {'zone': 'z'}
            placer.select_next_location.return_value = location
        manager._spot_placer = placer

        with mock.patch('sky.serve.replica_managers._should_use_spot',
                        return_value=use_spot), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state.get_replica_infos',
                 return_value=[]), \
             mock.patch(
                 'sky.serve.replica_managers.serve_state'
                 '.add_or_update_replica'), \
             mock.patch(
                 'sky.serve.replica_managers.serve_utils'
                 '.generate_replica_launch_log_file_name',
                 return_value='/tmp/launch.log'), \
             mock.patch('sky.serve.replica_managers._get_resources_ports',
                        return_value='8080'), \
             mock.patch('sky.serve.replica_managers.ReplicaInfo'), \
             mock.patch('sky.serve.replica_managers.thread_utils.SafeThread'
                       ) as mock_thread:
            manager._launch_replica(replica_id=1)
        return mock_thread.call_args

    def test_spot_with_placer_fails_fast_on_availability(self):
        call = self._launch_replica(use_spot=True, with_placer=True)
        assert call.kwargs['kwargs'] == {'availability_max_retry': 1}
        # retry_until_up must be False: failover is owned by the placer.
        assert call.kwargs['args'][-1] is False

    def test_spot_without_placer_keeps_default_retries(self):
        call = self._launch_replica(use_spot=True, with_placer=False)
        assert call.kwargs['kwargs'] == {'availability_max_retry': None}
        assert call.kwargs['args'][-1] is True

    def test_non_spot_keeps_default_retries(self):
        call = self._launch_replica(use_spot=False, with_placer=False)
        assert call.kwargs['kwargs'] == {'availability_max_retry': None}
        assert call.kwargs['args'][-1] is True
