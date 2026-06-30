"""Regression test: the serve controller subprocess hard-exits when its
uvicorn server returns, so the parent watchdog can respawn it.

``SkyServeController.run()`` starts non-daemon supervised control-loop threads
(autoscaler, replica refresher / prober / job-status fetcher) and then calls
``uvicorn.run()``. If ``uvicorn.run()`` ever returns -- a clean shutdown, a
child-only SIGINT raising ``KeyboardInterrupt``, or any other exit -- the HTTP
control plane is dead but the non-daemon threads keep looping, so the
interpreter cannot exit and the process lingers. The parent ``_start`` watchdog
only respawns the controller when ``controller_process.is_alive()`` is False,
so a lingering child is never respawned and the service is stuck with no working
controller. ``run()`` therefore force-terminates the subprocess once
``uvicorn.run()`` returns (in a ``finally`` so it also fires on an exception).
"""
# pylint: disable=protected-access
import fastapi
import pytest

from sky.serve import controller as controller_mod


def _make_controller(monkeypatch):
    """Build a controller without its heavy __init__ and stub the server bits
    so ``run()`` exercises only the exit behavior."""
    ctrl = controller_mod.SkyServeController.__new__(
        controller_mod.SkyServeController)
    ctrl._app = fastapi.FastAPI()
    ctrl._host = '127.0.0.1'
    ctrl._port = 20010
    # Don't actually start a control-loop thread during the test.
    monkeypatch.setattr(controller_mod.thread_utils, 'start_supervised_thread',
                        lambda *a, **k: None)
    exits = []

    def _fake_exit(code):
        exits.append(code)
        # Real os._exit never returns; emulate that so run() stops here and the
        # test can observe the call.
        raise SystemExit(code)

    monkeypatch.setattr(controller_mod.os, '_exit', _fake_exit)
    return ctrl, exits


def test_hard_exit_when_uvicorn_returns(monkeypatch):
    ctrl, exits = _make_controller(monkeypatch)
    monkeypatch.setattr(controller_mod.uvicorn, 'run', lambda *a, **k: None)

    with pytest.raises(SystemExit):
        ctrl.run()

    assert exits == [1], 'controller must hard-exit when uvicorn.run() returns'


def test_hard_exit_when_uvicorn_raises(monkeypatch):
    ctrl, exits = _make_controller(monkeypatch)

    def _boom(*a, **k):
        raise KeyboardInterrupt()

    monkeypatch.setattr(controller_mod.uvicorn, 'run', _boom)

    with pytest.raises(SystemExit):
        ctrl.run()

    assert exits == [1], ('controller must hard-exit even when uvicorn.run() '
                          'exits via an exception (e.g. child-only SIGINT)')
