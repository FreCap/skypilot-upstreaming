"""Regression test: interrupt_request_for_retry must skip terminal requests.

During graceful shutdown, ``_wait_requests`` snapshots the active requests and
then calls ``interrupt_request_for_retry`` per id; a request can finish in the
gap between the snapshot and the interrupt. Without a terminal-status guard the
interrupt would:
  (a) overwrite a SUCCEEDED/FAILED result with CANCELLED + should_retry --
      losing the recorded return value and making the client re-run an
      operation that already completed; and
  (b) ``os.kill`` a stale ``pid`` -- finished requests do not clear ``pid`` and
      the worker pool reuses PIDs, so the signal could hit an unrelated
      in-flight request.
Every other kill-path skips ``status > RUNNING``; this pins that
``interrupt_request_for_retry`` does too, without breaking the normal
non-terminal interrupt.
"""
# pylint: disable=protected-access
import signal
import unittest.mock as mock

import pytest

from sky.server import uvicorn as uvicorn_module
from sky.server.requests import payloads
from sky.server.requests import requests
from sky.server.requests.requests import RequestStatus


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
            requests._DB = None
            yield
            requests._DB = None


def _make_request(request_id: str, status: RequestStatus,
                  pid: int) -> requests.Request:
    return requests.Request(request_id=request_id,
                            name='sky.launch',
                            entrypoint=_dummy,
                            request_body=payloads.RequestBody(),
                            status=status,
                            created_at=0.0,
                            user_id='test-user',
                            pid=pid)


def _server() -> uvicorn_module.Server:
    # interrupt_request_for_retry uses no instance state; build a bare instance
    # to avoid uvicorn.Config wiring.
    return uvicorn_module.Server.__new__(uvicorn_module.Server)


@pytest.mark.asyncio
async def test_skips_succeeded_request(isolated_database, monkeypatch):
    assert await requests.create_if_not_exists_async(
        _make_request('req-done', RequestStatus.SUCCEEDED, pid=999999))
    kills = []
    monkeypatch.setattr(uvicorn_module.os, 'kill',
                        lambda pid, sig: kills.append((pid, sig)))

    _server().interrupt_request_for_retry('req-done')

    # No signal sent to the (now possibly reused) pid...
    assert kills == []
    # ...and the terminal result is left untouched.
    record = requests.get_request('req-done')
    assert record.status == RequestStatus.SUCCEEDED
    assert record.should_retry is False


@pytest.mark.asyncio
async def test_still_interrupts_running_request(isolated_database, monkeypatch):
    assert await requests.create_if_not_exists_async(
        _make_request('req-live', RequestStatus.RUNNING, pid=4242))
    kills = []
    monkeypatch.setattr(uvicorn_module.os, 'kill',
                        lambda pid, sig: kills.append((pid, sig)))

    _server().interrupt_request_for_retry('req-live')

    # The normal path is preserved: the live worker is signalled and the
    # request is flagged for client retry.
    assert kills == [(4242, signal.SIGTERM)]
    record = requests.get_request('req-live')
    assert record.status == RequestStatus.CANCELLED
    assert record.should_retry is True
