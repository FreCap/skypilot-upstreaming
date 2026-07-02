"""Regression tests for the SkyServe control-loop launch-budget scan cost.

Before the fix, ``ReplicaManager._refresh_thread_pool`` evaluated the launch
budget by calling ``controller_utils.can_provision`` / ``can_terminate`` once
*per* launching/terminating replica, and each call scanned and unpickled the
ENTIRE replica table twice (``serve_state.total_number_provisioning_replicas``
+ ``total_number_terminating_replicas``). That is O(K*N) ``pickle.loads`` per
refresh tick (measured ~1.7s at N=2000, K=140; grows with fleet size), burning
the refresh loop's CPU budget on bookkeeping instead of starting launches and
drains.

The fix hoists the budget read ONCE per tick via
``controller_utils.in_flight_launch_count`` and tracks the delta locally, so the
predicate accepts a pre-computed ``in_flight`` and does not re-scan.

These tests fail on the pre-fix code (the per-replica predicate has no
``in_flight`` parameter and the loop scans K times) and pass after it.
"""
import threading

from sky.serve import replica_managers
from sky.serve import serve_state
from sky.utils import common_utils
from sky.utils import controller_utils


class _NotStartedThread:
    """Stands in for a queued-but-not-started launch SafeThread."""
    format_exc = None

    def __init__(self):
        self.started = False

    def is_alive(self) -> bool:
        return False

    def start(self) -> None:
        self.started = True


def _pending_replica(replica_id: int) -> replica_managers.ReplicaInfo:
    # Default ReplicaStatusProperty -> sky_launch_status SCHEDULED -> PENDING.
    return replica_managers.ReplicaInfo(replica_id=replica_id,
                                        cluster_name=f'c{replica_id}',
                                        replica_port='8080',
                                        is_spot=True,
                                        location=None,
                                        version=1,
                                        resources_override=None)


def _build_manager(num_launching: int):
    # Bypass __init__: it spawns the refresher/prober/job-status daemon threads.
    mgr = object.__new__(replica_managers.SkyPilotReplicaManager)
    mgr.lock = threading.Lock()
    mgr._service_name = 'svc'
    mgr._is_pool = False
    mgr._spot_placer = None
    mgr.least_recent_version = 1
    mgr._launch_thread_pool = {
        rid: _NotStartedThread() for rid in range(num_launching)
    }
    mgr._down_thread_pool = {}
    mgr._replica_to_request_id = {}
    return mgr


def test_refresh_thread_pool_scans_budget_once_per_tick(monkeypatch, tmp_path):
    """With K launching replicas, the budget table is scanned O(1), not O(K)."""
    num_launching = 50
    replicas = {rid: _pending_replica(rid) for rid in range(num_launching)}
    scans = {'provisioning': 0, 'terminating': 0}

    def _count_provisioning() -> int:
        scans['provisioning'] += 1
        return sum(1 for info in replicas.values()
                   if info.status == serve_state.ReplicaStatus.PROVISIONING)

    def _count_terminating() -> int:
        scans['terminating'] += 1
        return 0

    monkeypatch.setattr(serve_state, 'total_number_provisioning_replicas',
                        _count_provisioning)
    monkeypatch.setattr(serve_state, 'total_number_terminating_replicas',
                        _count_terminating)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda svc, rid: replicas[rid])
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda svc: list(replicas.values()))
    monkeypatch.setattr(
        serve_state, 'add_or_update_replica',
        lambda svc, rid, info: replicas.__setitem__(rid, info))
    # Ample budget so every pending replica is allowed to launch; also avoids
    # the memory-probing path inside _get_request_parallelism.
    monkeypatch.setattr(controller_utils, '_get_request_parallelism',
                        lambda pool: 10_000)
    monkeypatch.setattr(controller_utils, 'get_resources_lock_path',
                        lambda: str(tmp_path / 'resources.lock'))

    mgr = _build_manager(num_launching)
    mgr._refresh_thread_pool()

    # Correctness preserved: every pending replica got launched...
    assert all(t.started for t in mgr._launch_thread_pool.values())
    assert all(
        info.status_property.sky_launch_status == common_utils.ProcessStatus.
        RUNNING for info in replicas.values())
    # ...and the whole-table budget scan happened at most ONCE for the tick,
    # not once per launching replica (the O(K*N) bug -> would be num_launching).
    assert scans['provisioning'] <= 1, (
        f'budget table scanned {scans["provisioning"]}x for {num_launching} '
        'launching replicas; expected a single hoisted scan per tick')
    assert scans['terminating'] <= 1


def test_down_admission_uses_hoisted_budget(monkeypatch, tmp_path):
    """With K terminating replicas, the budget table is scanned O(1)."""
    num_terminating = 20
    replicas = {}
    for rid in range(num_terminating):
        info = _pending_replica(rid)
        info.status_property.sky_down_status = (
            common_utils.ProcessStatus.SCHEDULED)
        replicas[rid] = info
    scans = {'n': 0}

    def _scan() -> int:
        scans['n'] += 1
        return 0

    monkeypatch.setattr(serve_state, 'total_number_provisioning_replicas',
                        _scan)
    monkeypatch.setattr(serve_state, 'total_number_terminating_replicas',
                        _scan)
    monkeypatch.setattr(serve_state, 'get_replica_info_from_id',
                        lambda svc, rid: replicas[rid])
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda svc: list(replicas.values()))
    monkeypatch.setattr(
        serve_state, 'add_or_update_replica',
        lambda svc, rid, info: replicas.__setitem__(rid, info))
    monkeypatch.setattr(controller_utils, '_get_request_parallelism',
                        lambda pool: 10_000)
    monkeypatch.setattr(controller_utils, 'get_resources_lock_path',
                        lambda: str(tmp_path / 'resources.lock'))

    mgr = _build_manager(num_launching=0)
    mgr._down_thread_pool = {
        rid: _NotStartedThread() for rid in range(num_terminating)
    }
    mgr._refresh_thread_pool()

    assert all(t.started for t in mgr._down_thread_pool.values())
    assert all(
        info.status_property.sky_down_status == common_utils.ProcessStatus.
        RUNNING for info in replicas.values())
    # One in_flight_launch_count read = one scan of each of the two tables.
    assert scans['n'] <= 2


def test_idle_tick_performs_no_budget_scan(monkeypatch):
    """A tick with nothing to admit must not scan the budget tables."""
    scans = {'n': 0}

    def _scan() -> int:
        scans['n'] += 1
        return 0

    monkeypatch.setattr(serve_state, 'total_number_provisioning_replicas',
                        _scan)
    monkeypatch.setattr(serve_state, 'total_number_terminating_replicas',
                        _scan)
    monkeypatch.setattr(serve_state, 'get_replica_infos', lambda svc: [])

    mgr = _build_manager(num_launching=0)
    mgr._refresh_thread_pool()

    assert scans['n'] == 0


def test_can_provision_with_precomputed_in_flight_skips_db_scan(monkeypatch):
    """can_provision/can_terminate must honor a pre-computed in_flight count
    without touching the (expensive) whole-table scan functions."""
    scanned = {'n': 0}

    def _boom():
        scanned['n'] += 1
        return 0

    monkeypatch.setattr(serve_state, 'total_number_provisioning_replicas',
                        _boom)
    monkeypatch.setattr(serve_state, 'total_number_terminating_replicas', _boom)
    monkeypatch.setattr(controller_utils, '_get_request_parallelism',
                        lambda pool: 100)

    assert controller_utils.can_provision(False, in_flight=3) is True
    assert controller_utils.can_terminate(False, in_flight=3) is True
    assert scanned['n'] == 0

    # And when in_flight is NOT supplied it falls back to scanning (one each).
    assert controller_utils.can_terminate(False) is True
    assert scanned['n'] == 2
