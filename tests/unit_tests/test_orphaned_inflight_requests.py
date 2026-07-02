"""Regression test: in-flight requests dropped on API-server restart must be
surfaced, not silently lost.

``requests.reset_db_and_logs`` runs on every API-server startup and wipes the
request DB + logs; the executor child processes that ran those requests died
with the previous process. A request still PENDING/WAITING/RUNNING -- e.g. a
long provisioning launch -- was therefore silently dropped: the caller saw the
request vanish, and any half-provisioned cluster leaked until a later status
refresh.

``_log_orphaned_inflight_requests`` is the minimal mitigation: detect and loudly
log the orphaned requests before the wipe so the drop is alertable and the
leaked clusters reconcilable. These tests pin that behavior, including that a
scan failure never blocks startup.
"""
# pylint: disable=protected-access
import types
import unittest.mock as mock

import pytest

from sky.server import daemons
from sky.server.requests import payloads
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
    monkeypatch.setattr(request_storage, 'get_request_backend', lambda: backend)
    warnings = _capture_warnings(monkeypatch)

    requests_lib._log_orphaned_inflight_requests()

    # One summary warning plus one warning per orphaned request.
    assert len(warnings) == 1 + len(reqs)
    # The scan must filter to only active (in-flight) statuses, and must not
    # select the pickled columns, whose decode can fail across an upgrade.
    req_filter = backend.filters[0]
    assert req_filter.status == requests_lib.RequestStatus.active_statuses()
    assert set(
        req_filter.fields) == {'request_id', 'name', 'status', 'cluster_name'}


def test_daemon_requests_are_not_reported(monkeypatch):
    # Internal daemon requests sit in RUNNING for the server's whole life and
    # are recreated on every startup: they are not dropped work.
    daemon_reqs = [
        _FakeReq(daemon.id, daemon.name, 'RUNNING')
        for daemon in daemons.INTERNAL_REQUEST_DAEMONS
    ]
    monkeypatch.setattr(request_storage, 'get_request_backend',
                        lambda: _FakeBackend(daemon_reqs))
    warnings = _capture_warnings(monkeypatch)

    requests_lib._log_orphaned_inflight_requests()

    assert not warnings


def test_no_orphans_no_warning(monkeypatch):
    monkeypatch.setattr(request_storage, 'get_request_backend',
                        lambda: _FakeBackend([]))
    warnings = _capture_warnings(monkeypatch)

    requests_lib._log_orphaned_inflight_requests()

    assert not warnings, 'no in-flight requests -> no warning'


def test_scan_failure_does_not_block_startup(monkeypatch):

    def _boom():
        raise RuntimeError('request DB schema incompatible')

    monkeypatch.setattr(request_storage, 'get_request_backend', _boom)
    warnings = _capture_warnings(monkeypatch)

    # Must not raise -- a scan failure may never block API-server startup.
    requests_lib._log_orphaned_inflight_requests()

    assert not warnings


def _dummy():
    return None


@pytest.fixture()
def isolated_database(tmp_path):
    temp_db_path = tmp_path / 'requests.db'
    temp_log_path = tmp_path / 'logs'
    temp_log_path.mkdir()
    with mock.patch('sky.server.constants.API_SERVER_REQUEST_DB_PATH',
                    str(temp_db_path)):
        with mock.patch('sky.server.constants.REQUEST_LOG_PATH_PREFIX',
                        str(temp_log_path)):
            requests_lib._DB = None
            yield
            requests_lib._DB = None


def _make_request(request_id: str,
                  status: requests_lib.RequestStatus) -> requests_lib.Request:
    return requests_lib.Request(request_id=request_id,
                                name='sky.launch',
                                entrypoint=_dummy,
                                request_body=payloads.RequestBody(),
                                status=status,
                                created_at=0.0,
                                user_id='test-user')


@pytest.mark.asyncio
async def test_reset_db_and_logs_reinitializes_backend(isolated_database,
                                                       tmp_path, monkeypatch):
    # The pre-wipe orphan scan initializes the DB handle against the old
    # database file; reset_db_and_logs must not leave that handle bound to
    # the unlinked file, or post-reset reads would serve phantom pre-restart
    # rows and the fresh database would never be created on this thread.
    monkeypatch.setattr(requests_lib, 'LEGACY_REQUEST_LOG_PATH_PREFIX',
                        str(tmp_path / 'legacy_logs'))
    monkeypatch.setattr(requests_lib.bs, 'get_blob_storage', mock.Mock)
    warnings = _capture_warnings(monkeypatch)
    assert await requests_lib.create_if_not_exists_async(
        _make_request('req-old', requests_lib.RequestStatus.RUNNING))

    requests_lib.reset_db_and_logs()

    # The scan ran before the wipe: summary + one orphaned request.
    assert len(warnings) == 2
    # The backend now serves the re-created, empty database, and new
    # requests can be written to it.
    filt = requests_lib.RequestTaskFilter()
    assert requests_lib.get_request_tasks(filt) == []
    assert await requests_lib.create_if_not_exists_async(
        _make_request('req-new', requests_lib.RequestStatus.PENDING))
