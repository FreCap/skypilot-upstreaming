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
from sky.jobs import state as managed_job_state
from sky.jobs import utils


def _make_task():
    return {
        'schedule_state': managed_job_state.ManagedJobScheduleState.ALIVE,
        'controller_pid': 123,
        'controller_pid_started_at': None,
        'status': managed_job_state.ManagedJobStatus.RUNNING,
        'job_name': 'job',
        'pool': None,
    }


def _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls):
    monkeypatch.setattr(managed_job_state, 'get_jobs_to_check_status',
                        lambda job_id=None: [1])
    monkeypatch.setattr(managed_job_state, 'get_managed_job_tasks',
                        lambda job_id: [_make_task()])
    monkeypatch.setattr(utils, 'controller_process_alive',
                        lambda record, job_id: False)
    monkeypatch.setattr(
        managed_job_state, 'get_job_schedule_state',
        lambda job_id: managed_job_state.ManagedJobScheduleState.ALIVE)
    monkeypatch.setattr(utils.global_user_state,
                        'get_handle_from_cluster_name', lambda name: None)
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
        'a job must not be FAILED_CONTROLLER while its controller is restarting')
    assert job_done_calls == []


def test_marks_failed_controller_when_no_restart(monkeypatch):
    set_failed_calls, job_done_calls = [], []
    _wire_dead_controller(monkeypatch, set_failed_calls, job_done_calls)
    monkeypatch.setattr(utils, '_controller_is_restarting', lambda: False)

    utils.update_managed_jobs_statuses(job_id=1)

    assert len(set_failed_calls) == 1, (
        'a genuinely dead controller (no restart) must still fail the job')
    assert len(job_done_calls) == 1


def test_controller_is_restarting_reflects_signal_file(monkeypatch, tmp_path):
    sig = tmp_path / 'restart.signal'
    monkeypatch.setattr(utils.constants,
                        'PERSISTENT_RUN_RESTARTING_SIGNAL_FILE', str(sig))
    assert utils._controller_is_restarting() is False
    sig.write_text('')
    assert utils._controller_is_restarting() is True
