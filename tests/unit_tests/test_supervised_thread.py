"""Regression test: serve control-loop threads must be supervised.

The serve autoscaler (controller.py) and the replica manager's
refresher / prober / job-status fetcher (replica_managers.py) ran as bare
``threading.Thread``s whose bodies catch ``Exception`` but not
``BaseException``. A ``BaseException`` escaping one of those loops (or the loop
returning) silently ended the thread while the controller process kept serving
HTTP -- a "half-dead" controller where the dead duty never runs again until a
full process restart, which in VM mode nothing triggers.

``thread_utils.start_supervised_thread`` wraps such a duty so any exit -- a
normal return OR a ``BaseException`` -- is logged and the duty is restarted.
These tests pin that contract deterministically.
"""
import threading

from sky.utils import thread_utils


def test_restarts_after_baseexception():
    """A BaseException (e.g. SystemExit) escaping the target must restart it,
    not silently kill the thread."""
    stop = threading.Event()
    calls = []
    reached_three = threading.Event()

    def target():
        calls.append(1)
        if len(calls) >= 3:
            reached_three.set()
            stop.set()  # let the supervisor exit cleanly after this run
            return
        # SystemExit is a BaseException, NOT an Exception -- the pre-fix bare
        # thread would have died here and never come back.
        raise SystemExit('simulated non-Exception thread death')

    t = thread_utils.start_supervised_thread(target,
                                             'test-baseexc',
                                             restart_backoff_seconds=0.01,
                                             stop_event=stop)

    assert reached_three.wait(5), 'target was not restarted after SystemExit'
    t.join(timeout=5)
    assert not t.is_alive(), 'supervisor did not stop on stop_event'
    assert len(calls) >= 3, 'target should have been restarted at least twice'


def test_restarts_after_normal_return():
    """Even a normal return (the duty loops are infinite, so returning is
    unexpected) must restart the target."""
    stop = threading.Event()
    calls = []

    def target():
        calls.append(1)
        if len(calls) >= 3:
            stop.set()
        return

    t = thread_utils.start_supervised_thread(target,
                                             'test-return',
                                             restart_backoff_seconds=0.01,
                                             stop_event=stop)

    t.join(timeout=5)
    assert not t.is_alive()
    assert len(calls) >= 3, 'target should have been restarted after return'


def test_stop_event_halts_supervision():
    """A pre-set stop_event means the target never runs and the thread exits."""
    stop = threading.Event()
    stop.set()
    calls = []

    t = thread_utils.start_supervised_thread(lambda: calls.append(1),
                                             'test-stop',
                                             restart_backoff_seconds=0.01,
                                             stop_event=stop)

    t.join(timeout=5)
    assert not t.is_alive()
    assert not calls, 'a pre-set stop_event must prevent the target running'
