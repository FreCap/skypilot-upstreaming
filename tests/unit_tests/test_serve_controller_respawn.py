"""Regression test: a dead serve controller is re-created (fresh port) + LB.

``sky/serve/service.py::_start`` runs the controller server as a child process,
but its supervision loop historically only checked DB row ownership -- never the
controller child's liveness. HA recovery only re-creates a controller on parent
``controller_pid`` loss / pod move; it does not cover the controller child dying
while the parent stays alive, and in VM mode nothing does.

``_respawn_controller_and_lb`` re-creates the controller on a FRESH port (chosen
free under the port-selection lock, which avoids cross-wiring to another service
that might have taken a reused port) and restarts the load balancer pointing at
it on the same public port. ``_ensure_load_balancer`` keeps the LB up. These
tests pin their contracts, including that failures are contained (never raised
into ``_start``'s destructive ``_cleanup``) and the data plane is preserved on a
failed controller bring-up.
"""
import types

from sky.serve import serve_state
from sky.serve import service
from sky.utils import common_utils
from sky.utils import subprocess_utils

_PORT = 20005


class _FakeProc:

    def __init__(self, alive: bool, pid: int = 100):
        self._alive = alive
        self.pid = pid
        self.exitcode = None if alive else 1

    def is_alive(self) -> bool:
        return self._alive


class _DummyLock:

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _spec(pool=False):
    return types.SimpleNamespace(pool=pool)


def _setup(monkeypatch, *, new_controller, new_lb=None, ready=True,
           latest_version=None, latest_spec=None, killed=None,
           owns_row=True):
    """Wire the respawn collaborators. new_controller/new_lb may be _FakeProc,
    None, or an Exception instance to raise from the spawn."""
    monkeypatch.setattr(service.filelock, 'FileLock', _DummyLock)
    monkeypatch.setattr(common_utils, 'find_free_port', lambda start: _PORT)
    monkeypatch.setattr(serve_state, 'set_service_controller_port_if_owner',
                        lambda name, pid, port: owns_row)
    monkeypatch.setattr(serve_state, 'get_latest_version',
                        lambda name: latest_version)
    monkeypatch.setattr(serve_state, 'get_spec',
                        lambda name, ver: latest_spec)

    spawn_ctrl_calls = []

    def _spawn_controller(service_name, spec, version, host, port):
        spawn_ctrl_calls.append(dict(spec=spec, version=version, port=port))
        if isinstance(new_controller, BaseException):
            raise new_controller
        return new_controller

    monkeypatch.setattr(service, '_spawn_controller', _spawn_controller)

    def _spawn_lb(controller_addr, lb_port, spec, log_file):
        if isinstance(new_lb, BaseException):
            raise new_lb
        return new_lb

    monkeypatch.setattr(service, '_spawn_load_balancer', _spawn_lb)

    if ready:
        monkeypatch.setattr(service, '_wait_for_controller_ready',
                            lambda *a, **k: None)
    else:
        def _not_ready(*a, **k):
            raise RuntimeError('not ready')

        monkeypatch.setattr(service, '_wait_for_controller_ready', _not_ready)

    if killed is None:
        killed = []
    monkeypatch.setattr(
        subprocess_utils, 'kill_children_processes',
        lambda parent_pids, force=False: killed.extend(parent_pids))
    return spawn_ctrl_calls, killed


# --- _respawn_controller_and_lb ---


def test_respawn_recreates_controller_on_fresh_port_and_restarts_lb(
        monkeypatch):
    dead_ctrl = _FakeProc(alive=False, pid=111)
    old_lb = _FakeProc(alive=True, pid=222)
    new_ctrl = _FakeProc(alive=True, pid=333)
    new_lb = _FakeProc(alive=True, pid=444)
    killed = []
    _setup(monkeypatch, new_controller=new_ctrl, new_lb=new_lb, killed=killed)

    result = service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1',
                                                30001, '/tmp/lb.log', dead_ctrl,
                                                old_lb)

    assert result == (new_ctrl, new_lb, _PORT)
    # The dead controller and the stale LB are reaped after success.
    assert 111 in killed and 222 in killed


def test_respawn_reloads_latest_version_and_spec(monkeypatch):
    latest_spec = _spec()
    spawn_calls, _ = _setup(monkeypatch,
                            new_controller=_FakeProc(alive=True, pid=333),
                            new_lb=_FakeProc(alive=True, pid=444),
                            latest_version=7, latest_spec=latest_spec)

    service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1', 30001,
                                       '/tmp/lb.log',
                                       _FakeProc(alive=False, pid=111), None)

    assert spawn_calls[0]['version'] == 7
    assert spawn_calls[0]['spec'] is latest_spec


def test_respawn_controller_failure_preserves_old_lb(monkeypatch):
    """If the replacement controller never becomes live, return None and leave
    the OLD load balancer running so the data plane survives retries."""
    dead_ctrl = _FakeProc(alive=False, pid=111)
    old_lb = _FakeProc(alive=True, pid=222)
    not_live = _FakeProc(alive=False, pid=333)  # exited during startup
    killed = []
    _setup(monkeypatch, new_controller=not_live, killed=killed)

    result = service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1',
                                                30001, '/tmp/lb.log', dead_ctrl,
                                                old_lb)

    assert result is None
    assert 222 not in killed, 'the old LB must be preserved on a failed respawn'
    assert 333 in killed, 'the not-live replacement controller must be reaped'


def test_respawn_controller_spawn_raises_is_contained(monkeypatch):
    killed = []
    _setup(monkeypatch, new_controller=OSError('no memory'), killed=killed)

    result = service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1',
                                                30001, '/tmp/lb.log',
                                                _FakeProc(False, 111),
                                                _FakeProc(True, 222))

    assert result is None  # contained, not raised
    assert 222 not in killed, 'old LB preserved when controller spawn fails'


def test_respawn_lost_row_ownership_discards_replacement(monkeypatch):
    """If another instance took over the DB row while the replacement was
    booting, the guarded port write reports lost ownership: the replacement
    must be discarded (not returned) and the old LB preserved."""
    old_lb = _FakeProc(alive=True, pid=222)
    new_ctrl = _FakeProc(alive=True, pid=333)
    killed = []
    _setup(monkeypatch, new_controller=new_ctrl, killed=killed, owns_row=False)

    result = service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1',
                                                30001, '/tmp/lb.log',
                                                _FakeProc(False, 111), old_lb)

    assert result is None
    assert 333 in killed, 'the disowned replacement controller must be reaped'
    assert 222 not in killed, 'the old LB must be preserved on lost ownership'


def test_respawn_lb_failure_returns_live_controller_with_no_lb(monkeypatch):
    """If the controller comes up but the LB restart fails, keep the controller
    (the LB is retried by the ensure-LB path) -- don't tear down a good
    controller over an LB hiccup."""
    new_ctrl = _FakeProc(alive=True, pid=333)
    _setup(monkeypatch, new_controller=new_ctrl, new_lb=OSError('lb boom'))

    result = service._respawn_controller_and_lb('svc', _spec(), 1, '127.0.0.1',
                                                30001, '/tmp/lb.log',
                                                _FakeProc(False, 111), None)

    assert result == (new_ctrl, None, _PORT)


# --- _ensure_load_balancer ---


def test_ensure_lb_noop_for_pool(monkeypatch):
    monkeypatch.setattr(service, '_spawn_load_balancer',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError('pool has no LB')))
    lb = _FakeProc(alive=True, pid=1)
    assert service._ensure_load_balancer(lb, 'http://h:1', 30001, _spec(
        pool=True), '/tmp/lb.log') is lb


def test_ensure_lb_keeps_live_lb(monkeypatch):
    spawned = []
    monkeypatch.setattr(service, '_spawn_load_balancer',
                        lambda *a, **k: spawned.append(1))
    lb = _FakeProc(alive=True, pid=1)
    assert service._ensure_load_balancer(lb, 'http://h:1', 30001, _spec(),
                                         '/tmp/lb.log') is lb
    assert spawned == []


def test_ensure_lb_restarts_dead_lb(monkeypatch):
    new_lb = _FakeProc(alive=True, pid=2)
    killed = []
    monkeypatch.setattr(service, '_spawn_load_balancer', lambda *a, **k: new_lb)
    monkeypatch.setattr(
        subprocess_utils, 'kill_children_processes',
        lambda parent_pids, force=False: killed.extend(parent_pids))
    out = service._ensure_load_balancer(_FakeProc(alive=False, pid=1),
                                        'http://h:1', 30001, _spec(),
                                        '/tmp/lb.log')
    assert out is new_lb
    assert 1 in killed


def test_ensure_lb_starts_missing_lb(monkeypatch):
    new_lb = _FakeProc(alive=True, pid=2)
    monkeypatch.setattr(service, '_spawn_load_balancer', lambda *a, **k: new_lb)
    assert service._ensure_load_balancer(None, 'http://h:1', 30001, _spec(),
                                         '/tmp/lb.log') is new_lb


def test_ensure_lb_spawn_failure_is_contained(monkeypatch):
    def _boom(*a, **k):
        raise OSError('cannot start')

    monkeypatch.setattr(service, '_spawn_load_balancer', _boom)
    assert service._ensure_load_balancer(None, 'http://h:1', 30001, _spec(),
                                         '/tmp/lb.log') is None
