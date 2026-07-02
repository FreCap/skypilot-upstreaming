"""Regression test: don't FAILED_CONTROLLER a job whose controller is restarting.

``update_managed_jobs_statuses`` checks the controller-restart signal file once
at the TOP, then iterates jobs and, for any whose controller pid is dead,
terminates the job's cluster and marks it FAILED_CONTROLLER. Marking many jobs
takes time, so a controller restart that begins in that window (creating the
signal file) races the loop: a job that is actually being restarted under it
gets its cluster torn down and is failed terminally instead of resuming.

The fix re-checks the restart signal immediately before the destructive action.
These tests pin that: a restart that appears mid-cycle defers the job's
FAILED_CONTROLLER, while a genuinely dead controller (no restart) still fails.
"""
import pytest

from sky.jobs import state as managed_job_state
from sky.jobs import utils
from sky.skylet import job_lib


def _make_task():
    return {
        'schedule_state': managed_job_state.ManagedJobScheduleState.ALIVE,
        'controller_pid': 123,
        'controller_pid_started_at': None,
        'status': managed_job_state.ManagedJobStatus.RUNNING,
        'job_name': 'job',
        'pool': None,
    }


def _make_status_check_info():
    """The slim per-job shape update_managed_jobs_statuses now reads."""
    return {
        1: {
            'schedule_state': managed_job_state.ManagedJobScheduleState.ALIVE,
            'controller_pid': 123,
            'controller_pid_started_at': None,
            'pool': None,
            'tasks': [{
                'task_id': 0,
                'status': managed_job_state.ManagedJobStatus.RUNNING,
                'job_name': 'job',
            }],
        }
    }


def _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls):
    monkeypatch.setattr(managed_job_state,
                        'get_jobs_to_check_status',
                        lambda job_id=None: [1])
    monkeypatch.setattr(managed_job_state, 'get_jobs_status_check_info',
                        lambda job_ids: _make_status_check_info())
    # _cleanup_job_clusters still re-fetches full task rows in this change.
    monkeypatch.setattr(managed_job_state, 'get_managed_job_tasks',
                        lambda job_id: [_make_task()])
    monkeypatch.setattr(utils, 'controller_process_alive',
                        lambda record, job_id: False)
    monkeypatch.setattr(
        managed_job_state, 'get_job_schedule_state',
        lambda job_id: managed_job_state.ManagedJobScheduleState.ALIVE)
    monkeypatch.setattr(utils.global_user_state, 'get_handle_from_cluster_name',
                        lambda name: None)
    monkeypatch.setattr(managed_job_state, 'set_failed',
                        lambda *a, **k: set_failed_calls.append((a, k)))
    monkeypatch.setattr(utils.scheduler, 'job_done',
                        lambda *a, **k: job_done_calls.append((a, k)))


def test_defers_failed_controller_when_restart_begins_midcycle(monkeypatch):
    set_failed_calls, job_done_calls = [], []
    _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls)
    # Restart signal: absent at the top-of-function check, present at the
    # re-check just before the destructive action.
    seq = iter([False, True])
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: next(seq))

    utils.update_managed_jobs_statuses(job_id=1)

    assert set_failed_calls == [], (
        'a job must not be FAILED_CONTROLLER while its controller is restarting'
    )
    assert job_done_calls == []


def test_marks_failed_controller_when_no_restart(monkeypatch):
    set_failed_calls, job_done_calls = [], []
    _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls)
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: False)

    utils.update_managed_jobs_statuses(job_id=1)

    assert len(set_failed_calls) == 1, (
        'a genuinely dead controller (no restart) must still fail the job')
    assert len(job_done_calls) == 1


def test_defers_terminal_write_when_restart_begins_during_cleanup(monkeypatch):
    """A restart beginning during cluster teardown must defer set_failed.

    The restart signal is absent at the top-of-function check and at the
    pre-cleanup re-check, and appears while the (potentially minutes-long)
    cluster teardown runs. Cleanup has already happened, but the terminal
    set_failed / job_done must not.
    """
    set_failed_calls, job_done_calls = [], []
    _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls)
    seq = iter([False, False, True])
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: next(seq))
    cleanup_reads = []
    monkeypatch.setattr(
        managed_job_state, 'get_managed_job_tasks',
        lambda job_id: cleanup_reads.append(job_id) or [_make_task()])

    utils.update_managed_jobs_statuses(job_id=1)

    assert cleanup_reads == [1], 'cluster cleanup should have started'
    assert set_failed_calls == []
    assert job_done_calls == []


def test_defers_when_job_reset_for_recovery_midcycle(monkeypatch):
    """A job reset after the snapshot (WAITING, pid cleared) must be deferred.

    The batched sweep snapshot judges the job dead (ALIVE, dead pid), but the
    fresh per-job re-read taken just before the destructive action shows it was
    reset for recovery. Neither the cluster teardown nor the terminal
    set_failed may run; the next sweep re-judges the job from fresh state.
    """
    set_failed_calls, job_done_calls = [], []
    _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls)
    reset_info = {
        1: {
            'schedule_state': managed_job_state.ManagedJobScheduleState.WAITING,
            'controller_pid': None,
            'controller_pid_started_at': None,
            'pool': None,
            'tasks': [{
                'task_id': 0,
                'status': managed_job_state.ManagedJobStatus.PENDING,
                'job_name': 'job',
            }],
        }
    }
    # First call: the sweep-wide snapshot (dead controller). Second call: the
    # fresh per-job re-read, showing the job was reset for recovery.
    snapshots = iter([_make_status_check_info(), reset_info])
    monkeypatch.setattr(managed_job_state, 'get_jobs_status_check_info',
                        lambda job_ids: next(snapshots))
    cleanup_reads = []
    monkeypatch.setattr(
        managed_job_state, 'get_managed_job_tasks',
        lambda job_id: cleanup_reads.append(job_id) or [_make_task()])
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: False)

    utils.update_managed_jobs_statuses(job_id=1)

    assert cleanup_reads == [], (
        'cluster cleanup must not start for a job reset for recovery')
    assert set_failed_calls == []
    assert job_done_calls == []


def _make_pending_status_check_info(schedule_state):
    """A pending job whose controller process has not started (pid is None)."""
    return {
        1: {
            'schedule_state': schedule_state,
            'controller_pid': None,
            'controller_pid_started_at': None,
            'pool': None,
            'tasks': [{
                'task_id': 0,
                'status': managed_job_state.ManagedJobStatus.PENDING,
                'job_name': 'job',
            }],
        }
    }


@pytest.mark.parametrize('schedule_state', [
    managed_job_state.ManagedJobScheduleState.INACTIVE,
    managed_job_state.ManagedJobScheduleState.WAITING,
])
def test_pending_job_skips_controller_status_read(monkeypatch, schedule_state):
    """A pid-None INACTIVE/WAITING job is skipped before the skylet controller
    status is read.

    Two things are pinned: (1) the per-job filelock + SQLite read in
    ``job_lib.get_status`` is avoided for the pending backlog, and (2) a stale
    skylet ``FAILED_SETUP`` row cannot strand a recovering job — a WAITING job
    (e.g. one reset by ``reset_jobs_for_recovery``) must be left for the
    scheduler to relaunch, not marked FAILED_CONTROLLER.
    """
    get_status_calls = []
    set_failed_calls = []

    def _record_get_status(job_id):
        get_status_calls.append(job_id)
        return job_lib.JobStatus.FAILED_SETUP

    monkeypatch.setattr(managed_job_state,
                        'get_jobs_to_check_status',
                        lambda job_id=None: [1])
    monkeypatch.setattr(
        managed_job_state, 'get_jobs_status_check_info',
        lambda job_ids: _make_pending_status_check_info(schedule_state))
    monkeypatch.setattr(job_lib, 'get_status', _record_get_status)
    monkeypatch.setattr(managed_job_state, 'set_failed',
                        lambda *a, **k: set_failed_calls.append((a, k)))
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: False)

    utils.update_managed_jobs_statuses(job_id=1)

    assert get_status_calls == [], (
        'controller status must not be read for a pid-None pending job')
    assert set_failed_calls == [], (
        'a pending (pre-controller) job must not be FAILED_CONTROLLER, even if '
        'a stale skylet status would read FAILED_SETUP')
