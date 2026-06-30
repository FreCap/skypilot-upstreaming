"""Regression test: _cleanup must remove the HA recovery script LAST.

``sky/serve/service.py::_cleanup`` used to delete the
``serve_ha_recovery_script`` row on its very first line, *before* the
(seconds-to-minutes) replica teardown.
If the controller pod was then killed mid-teardown (HA pod move / node drain),
the durable service row survived but its recovery script was gone, so
``ha_recovery_for_consolidation_mode`` logged 'recovery script does not exist.
Skipping recovery' forever and stranded the service with replicas still
consuming resources.

The fix moves the ``remove_ha_recovery_script`` call to the END of ``_cleanup``,
after all destructive teardown. These tests pin the ordering invariant: the
script must outlive replica teardown so a crash partway through leaves recovery
able to respawn the controller and re-run cleanup.
"""
import threading
import types

import pytest

from sky import exceptions
from sky.serve import replica_managers
from sky.serve import serve_state
from sky.serve import service
from sky.utils import common_utils
from sky.utils import controller_utils


def _replica(replica_id: int) -> replica_managers.ReplicaInfo:
    return replica_managers.ReplicaInfo(replica_id=replica_id,
                                        cluster_name=f'c{replica_id}',
                                        replica_port='8080',
                                        is_spot=False,
                                        location=None,
                                        version=1,
                                        resources_override=None)


def _patch_common(monkeypatch, events, replicas):
    """Wire up _cleanup's collaborators to record an ordered event log."""
    monkeypatch.setattr(service.time, 'sleep', lambda *_a, **_k: None)
    monkeypatch.setattr(serve_state, 'get_service_from_name', lambda svc: None)
    monkeypatch.setattr(serve_state, 'get_replica_infos',
                        lambda svc: list(replicas))
    monkeypatch.setattr(service.global_user_state,
                        'get_cluster_names_start_with',
                        lambda prefix: [r.cluster_name for r in replicas])
    monkeypatch.setattr(serve_state, 'add_or_update_replica',
                        lambda *a, **k: None)
    monkeypatch.setattr(serve_state, 'remove_replica', lambda *a, **k: None)
    monkeypatch.setattr(serve_state, 'get_service_versions', lambda svc: [])
    monkeypatch.setattr(controller_utils, 'can_terminate',
                        lambda pool, in_flight=None: True)
    monkeypatch.setattr(
        serve_state, 'remove_ha_recovery_script',
        lambda svc: events.append('remove_recovery_script'))


def test_recovery_script_removed_after_replica_teardown(monkeypatch):
    """The recovery script must be deleted only AFTER replica teardown runs."""
    events = []

    def _terminate(cluster_name, log_file_name):
        events.append(f'teardown:{cluster_name}')

    monkeypatch.setattr(replica_managers, 'terminate_cluster', _terminate)
    _patch_common(monkeypatch, events, [_replica(1)])

    failed = service._cleanup('svc', pool=False)

    assert failed is False
    assert events == ['teardown:c1', 'remove_recovery_script'], (
        'recovery script must be removed only after replica teardown; '
        f'got order {events}')


def test_recovery_script_survives_until_all_replicas_torn_down(monkeypatch):
    """With multiple replicas, the script removal must come after the LAST
    teardown -- a crash before that point leaves recovery able to retry."""
    events = []
    lock = threading.Lock()

    def _terminate(cluster_name, log_file_name):
        with lock:
            events.append(f'teardown:{cluster_name}')

    monkeypatch.setattr(replica_managers, 'terminate_cluster', _terminate)
    _patch_common(monkeypatch, events, [_replica(i) for i in range(3)])

    service._cleanup('svc', pool=False)

    assert events[-1] == 'remove_recovery_script', (
        f'recovery script removed before all teardowns finished: {events}')
    assert sorted(events[:-1]) == ['teardown:c0', 'teardown:c1', 'teardown:c2']


def test_recovery_script_removed_once_on_clean_service(monkeypatch):
    """No replicas: the script is still removed exactly once at the end."""
    events = []
    monkeypatch.setattr(replica_managers, 'terminate_cluster',
                        lambda *a, **k: events.append('teardown'))
    _patch_common(monkeypatch, events, [])

    service._cleanup('svc', pool=False)

    assert events == ['remove_recovery_script']


# --- recovery must resume teardown, not resurrect a torn-down service ---


def _svc(status):
    return {'status': status}


def test_resume_teardown_for_user_downed_service():
    """A recovery of a SHUTTING_DOWN service must resume teardown (else the
    preserved recovery script would resurrect a service the user `down`ed)."""
    assert service._should_resume_teardown(
        True, _svc(serve_state.ServiceStatus.SHUTTING_DOWN)) is True


def test_resume_teardown_for_failed_cleanup_service():
    assert service._should_resume_teardown(
        True, _svc(serve_state.ServiceStatus.FAILED_CLEANUP)) is True


def test_no_resume_for_ready_service():
    """A controller that died while the service was healthy (READY) must be
    recovered normally (brought back up), NOT torn down."""
    assert service._should_resume_teardown(
        True, _svc(serve_state.ServiceStatus.READY)) is False


def test_no_resume_on_fresh_run():
    assert service._should_resume_teardown(False, None) is False
    assert service._should_resume_teardown(
        False, _svc(serve_state.ServiceStatus.SHUTTING_DOWN)) is False


def _patch_finalize(monkeypatch, calls):
    monkeypatch.setattr(serve_state, 'set_service_status_and_active_versions',
                        lambda *a, **k: calls.append(('failed_cleanup', a)))
    monkeypatch.setattr(serve_state, 'remove_service_completely',
                        lambda name: calls.append(('removed', name)))
    monkeypatch.setattr(serve_state, 'remove_ha_recovery_script',
                        lambda name: calls.append(('remove_script', name)))
    monkeypatch.setattr(service.shutil, 'rmtree', lambda *a, **k: None)
    monkeypatch.setattr(service, '_cleanup_task_run_script', lambda jid: None)


def test_finalize_removes_service_on_clean_teardown(monkeypatch):
    calls = []
    monkeypatch.setattr(service, '_cleanup', lambda name, pool: False)
    _patch_finalize(monkeypatch, calls)

    service._run_cleanup_and_finalize('svc', types.SimpleNamespace(pool=False),
                                      '/tmp/svc', 1)

    assert ('removed', 'svc') in calls
    assert not any(c[0] == 'failed_cleanup' for c in calls)
    # _cleanup itself removed the script on its clean path; finalize must not.
    assert not any(c[0] == 'remove_script' for c in calls)


def test_finalize_marks_failed_cleanup_when_teardown_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(service, '_cleanup', lambda name, pool: True)
    _patch_finalize(monkeypatch, calls)

    service._run_cleanup_and_finalize('svc', types.SimpleNamespace(pool=False),
                                      '/tmp/svc', 1)

    assert any(c[0] == 'failed_cleanup' for c in calls)
    assert not any(c[0] == 'removed' for c in calls)
    # _cleanup completed (returned True) and removed its own script; a normal
    # failure must NOT be turned into a recovery loop -- but the explicit
    # removal here is only for the EXCEPTION path, so finalize must not call it.
    assert not any(c[0] == 'remove_script' for c in calls)


def test_finalize_contains_cleanup_exception_and_breaks_recovery_loop(
        monkeypatch):
    """A _cleanup that RAISES must be contained, leave the service
    FAILED_CLEANUP, AND remove the HA recovery script -- otherwise a persistent
    cleanup error loops forever (FAILED_CLEANUP is a resume status and the
    script was never reached for removal inside _cleanup)."""
    calls = []

    def _boom(name, pool):
        raise RuntimeError('cleanup blew up')

    monkeypatch.setattr(service, '_cleanup', _boom)
    _patch_finalize(monkeypatch, calls)

    service._run_cleanup_and_finalize('svc', types.SimpleNamespace(pool=False),
                                      '/tmp/svc', 1)

    assert any(c[0] == 'failed_cleanup' for c in calls)
    assert ('remove_script', 'svc') in calls, (
        'a caught cleanup exception must remove the HA script to avoid a '
        'recovery loop')


def test_handle_signal_persists_shutting_down_before_consuming_signal(
        monkeypatch, tmp_path):
    """The terminate signal must not be consumed before SHUTTING_DOWN is durably
    set: otherwise a crash in that window loses the teardown intent and HA
    recovery would bring the (user-downed) service back up serving."""
    sig = tmp_path / 'svc.signal'
    sig.write_text('terminate')
    monkeypatch.setattr(service.constants, 'SIGNAL_FILE_PATH',
                        str(tmp_path / '{}.signal'))
    observed = []

    def _record_status(name, status):
        # The signal file must still exist when we persist SHUTTING_DOWN.
        observed.append((status, sig.exists()))

    monkeypatch.setattr(serve_state, 'set_service_status_and_active_versions',
                        _record_status)

    with pytest.raises(exceptions.ServeUserTerminatedError):
        service._handle_signal('svc')

    assert observed, 'SHUTTING_DOWN must be persisted on a terminate signal'
    status, signal_existed_at_status_time = observed[0]
    assert status == serve_state.ServiceStatus.SHUTTING_DOWN
    assert signal_existed_at_status_time is True, (
        'status must be set BEFORE the signal file is consumed')
    assert not sig.exists(), 'signal file is consumed after status is persisted'
