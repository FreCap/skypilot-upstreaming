"""Regression test: in-flight requests dropped on API-server restart must be
surfaced, not silently lost.

``requests.reset_db_and_logs`` runs on every API-server startup and wipes the
request DB + logs; the executor child processes that ran those requests died
with the previous process. A request still PENDING/WAITING/RUNNING -- e.g. a
long provisioning launch -- was therefore silently dropped: the caller saw the
request vanish, and any half-provisioned cluster leaked until a later status
refresh. The boltz long-worker-pool widening admits more concurrent launches,
so each restart orphans proportionally more of them.

``_log_orphaned_inflight_requests`` is the minimal mitigation: detect and loudly
log the orphaned requests before the wipe so the drop is alertable and the
leaked clusters reconcilable. These tests pin that behavior, including that a
scan failure never blocks startup.
"""
import types

from sky.server.requests import requests as requests_lib
from sky.server.requests import storage as request_storage


class _FakeReq:

    def __init__(self, request_id, name, status_value, cluster_name=None):
        self.request_id = request_id
        self.name = name
        self.status = types.SimpleNamespace(value=status_value)
        self.cluster_name = cluster_name


class _FakeBackend:

    def __init__(self, reqs):
        self._reqs = reqs
        self.filters = []

    def query_requests(self, req_filter):
        self.filters.append(req_filter)
        return self._reqs


def _capture_warnings(monkeypatch):
    warnings = []
    monkeypatch.setattr(requests_lib.logger, 'warning',
                        lambda msg, *a, **k: warnings.append(str(msg)))
    return warnings


def test_logs_each_orphaned_inflight_request(monkeypatch):
    reqs = [
        _FakeReq('req-1', 'sky.launch', 'RUNNING', 'cluster-a'),
        _FakeReq('req-2', 'sky.down', 'PENDING'),
    ]
    backend = _FakeBackend(reqs)
    monkeypatch.setattr(request_storage, 'get_request_backend',
                        lambda: backend)
    warnings = _capture_warnings(monkeypatch)

    requests_lib._log_orphaned_inflight_requests()

    joined = '\n'.join(warnings)
    assert 'req-1' in joined and 'req-2' in joined
    assert 'cluster-a' in joined
    assert any('2 request' in w for w in warnings), 'must report the count'
    # The scan must filter to only active (in-flight) statuses.
    assert backend.filters[0].status == requests_lib.RequestStatus.active_statuses()


def test_no_orphans_no_warning(monkeypatch):
    monkeypatch.setattr(request_storage, 'get_request_backend',
                        lambda: _FakeBackend([]))
    warnings = _capture_warnings(monkeypatch)

    requests_lib._log_orphaned_inflight_requests()

    assert warnings == [], 'no in-flight requests -> no warning'


def test_scan_failure_does_not_block_startup(monkeypatch):
    def _boom():
        raise RuntimeError('request DB schema incompatible')

    monkeypatch.setattr(request_storage, 'get_request_backend', _boom)
    warnings = _capture_warnings(monkeypatch)

    # Must not raise -- a scan failure may never block API-server startup.
    requests_lib._log_orphaned_inflight_requests()

    assert warnings == []
