"""Regression test: _fetch_job_status must not hold self.lock during SSH.

``ReplicaManager._fetch_job_status`` SSHes into every tracked replica's head
node to query job status. It used to be decorated ``@with_lock``, so the global
``self.lock`` was held across the entire serial SSH walk. When a replica is
unreachable (e.g. a preempted spot VM), its SSH connect hangs at the kernel TCP
timeout (tens of seconds to minutes) -- and during that hang the refresher /
prober / scaler (which all take ``self.lock``) are blocked, stalling autoscaling
exactly when the fleet is churning hardest.

The fix does the SSH walk WITHOUT the lock and re-takes it only on the
failure-handling paths. This test pins that: while a replica's
``get_job_status`` is blocked, ``self.lock`` must still be acquirable. It is
deterministic (uses Events, not sleeps).
"""
import threading

from sky import exceptions
from sky.serve import replica_managers
from sky.serve import serve_state
from sky.skylet import job_lib
from sky.utils import common_utils


def _tracked_replica(replica_id: int) -> replica_managers.ReplicaInfo:
    info = replica_managers.ReplicaInfo(replica_id=replica_id,
                                        cluster_name=f'c{replica_id}',
                                        replica_port='8080',
                                        is_spot=True,
                                        location=None,
                                        version=1,
                                        resources_override=None)
    # SUCCEEDED + no down -> should_track_service_status() is True.
    info.status_property.sky_launch_status = common_utils.ProcessStatus.SUCCEEDED
    return info


def _build_manager():
    mgr = object.__new__(replica_managers.SkyPilotReplicaManager)
    mgr.lock = threading.Lock()
    mgr._service_name = 'svc'
    mgr._is_pool = False
    return mgr


def test_fetch_job_status_releases_lock_during_ssh(monkeypatch):
    replicas = [_tracked_replica(i) for i in range(3)]
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda svc: list(replicas))
    # Avoid the real cluster lookup / handle assertion.
    monkeypatch.setattr(replica_managers.ReplicaInfo, 'handle',
                        lambda self, cluster_record=None: object())

    ssh_entered = threading.Event()
    ssh_release = threading.Event()

    def _blocking_get_job_status(self, handle, job_ids, stream_logs=False):
        ssh_entered.set()
        # Simulate an unreachable replica's hung SSH connect.
        assert ssh_release.wait(timeout=5)
        return {1: job_lib.JobStatus.RUNNING}

    monkeypatch.setattr(replica_managers.backends.CloudVmRayBackend,
                        'get_job_status', _blocking_get_job_status)

    mgr = _build_manager()
    fetch = threading.Thread(target=mgr._fetch_job_status)
    fetch.start()
    acquired = False
    try:
        assert ssh_entered.wait(2), 'job-status SSH walk did not start'
        # While the SSH is blocked, the lock MUST be acquirable. Pre-fix
        # (_fetch_job_status was @with_lock) the lock is held for the whole
        # walk and this times out -> the control loop would be wedged behind
        # an unreachable replica.
        acquired = mgr.lock.acquire(timeout=1.0)
        if acquired:
            mgr.lock.release()
    finally:
        ssh_release.set()
        fetch.join(timeout=5)

    assert acquired, (
        'self.lock was held during the get_job_status SSH walk; an unreachable '
        'replica would stall the refresher/prober/scaler')


def _raise_command_error(self, handle, job_ids, stream_logs=False):
    raise exceptions.CommandError(returncode=255,
                                  command='get_job_status',
                                  error_msg='ssh failed',
                                  detailed_reason=None)


def test_preemption_path_acts_on_fresh_replica_not_stale_snapshot(monkeypatch):
    """On CommandError, preemption handling must re-read the replica under the
    lock and act on the FRESH state, not the pre-SSH snapshot (which another
    thread may have mutated while we SSHed lock-free)."""
    stale = _tracked_replica(7)
    fresh = _tracked_replica(7)  # distinct object, same id
    monkeypatch.setattr(serve_state, 'get_replica_infos', lambda svc: [stale])
    monkeypatch.setattr(replica_managers.ReplicaInfo, 'handle',
                        lambda self, cluster_record=None: object())
    monkeypatch.setattr(replica_managers.backends.CloudVmRayBackend,
                        'get_job_status', _raise_command_error)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda svc, rid: fresh)

    seen = []
    mgr = _build_manager()
    mgr._handle_preemption = lambda info: (seen.append(info) or True)
    mgr._fetch_job_status()

    assert seen == [fresh], 'preemption must act on the re-read fresh replica'


def test_preemption_path_skips_vanished_replica(monkeypatch):
    """If the replica is gone (re-read returns None) after a CommandError, the
    preemption path must skip it, not crash or act on the stale snapshot."""
    stale = _tracked_replica(9)
    monkeypatch.setattr(serve_state, 'get_replica_infos', lambda svc: [stale])
    monkeypatch.setattr(replica_managers.ReplicaInfo, 'handle',
                        lambda self, cluster_record=None: object())
    monkeypatch.setattr(replica_managers.backends.CloudVmRayBackend,
                        'get_job_status', _raise_command_error)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda svc, rid: None)

    called = []
    mgr = _build_manager()
    mgr._handle_preemption = lambda info: called.append(info)
    mgr._fetch_job_status()  # must not raise

    assert called == [], 'vanished replica must be skipped, not handled'


def test_walk_skips_replica_whose_cluster_record_vanished(monkeypatch):
    """A replica whose cluster record vanished mid-walk (handle() is None)
    must be skipped without aborting the walk for the remaining replicas."""
    replicas = [_tracked_replica(1), _tracked_replica(2)]
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda svc: list(replicas))
    monkeypatch.setattr(
        replica_managers.ReplicaInfo, 'handle', lambda self, cluster_record=None:
        None if self.replica_id == 1 else object())

    probed = []

    def _get_job_status(self, handle, job_ids, stream_logs=False):
        probed.append(handle)
        return {1: job_lib.JobStatus.RUNNING}

    monkeypatch.setattr(replica_managers.backends.CloudVmRayBackend,
                        'get_job_status', _get_job_status)

    mgr = _build_manager()
    mgr._fetch_job_status()  # must not raise

    assert len(probed) == 1, 'the remaining replica must still be checked'


def test_user_failure_path_skips_replica_scheduled_down(monkeypatch):
    """A FAILED job status must not mark user_app_failed on (or re-terminate)
    a replica that was scheduled for scale-down while the SSH ran lock-free."""
    stale = _tracked_replica(3)
    fresh = _tracked_replica(3)  # distinct object, same id
    fresh.status_property.sky_down_status = common_utils.ProcessStatus.SCHEDULED
    monkeypatch.setattr(serve_state, 'get_replica_infos', lambda svc: [stale])
    monkeypatch.setattr(replica_managers.ReplicaInfo, 'handle',
                        lambda self, cluster_record=None: object())
    monkeypatch.setattr(
        replica_managers.backends.CloudVmRayBackend, 'get_job_status',
        lambda self, handle, job_ids, stream_logs=False:
        {1: job_lib.JobStatus.FAILED})
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda svc, rid: fresh)
    writes = []
    monkeypatch.setattr(serve_state, 'add_or_update_replica',
                        lambda svc, rid, info: writes.append(rid))

    terminated = []
    mgr = _build_manager()
    mgr._terminate_replica = lambda rid, **kwargs: terminated.append(rid)
    mgr._fetch_job_status()

    assert not fresh.status_property.user_app_failed
    assert writes == []
    assert terminated == []
