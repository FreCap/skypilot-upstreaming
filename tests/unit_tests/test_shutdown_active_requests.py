"""Regression test: graceful shutdown must preserve WAITING requests.

``get_shutdown_active_requests`` returns the requests the graceful-shutdown
coordinator waits for and (for retryable ones) marks ``should_retry`` so the
client re-issues them after restart. ``WAITING`` is a non-terminal status (a
request parked on a retry backoff / external continue-condition, with its
resume timer living only in an in-memory monitor thread). If the shutdown query
omits WAITING, a parked request is neither waited for nor flagged for retry, its
in-memory timer dies with the process, and ``reset_db_and_logs`` wipes the row
on the next boot -- silently dropping the request on a *clean* restart even
though the whole ``should_retry`` machinery exists. These tests pin that
WAITING is treated like the other active statuses.
"""
import unittest.mock as mock

import pytest

from sky.server.requests import payloads
from sky.server.requests import requests
from sky.server.requests import storage as request_storage
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


def _make_request(request_id: str, status: RequestStatus) -> requests.Request:
    return requests.Request(request_id=request_id,
                            name='sky.launch',
                            entrypoint=_dummy,
                            request_body=payloads.RequestBody(),
                            status=status,
                            created_at=0.0,
                            user_id='test-user')


@pytest.mark.asyncio
async def test_shutdown_active_requests_includes_waiting(isolated_database):
    backend = request_storage.get_request_backend()
    for request_id, status in [
        ('req-pending', RequestStatus.PENDING),
        ('req-waiting', RequestStatus.WAITING),
        ('req-running', RequestStatus.RUNNING),
        ('req-succeeded', RequestStatus.SUCCEEDED),
        ('req-failed', RequestStatus.FAILED),
    ]:
        assert await requests.create_if_not_exists_async(
            _make_request(request_id, status))

    active_ids = {rid for rid, _ in backend.get_shutdown_active_requests()}

    # A WAITING request must be waited for / retried on graceful shutdown,
    # exactly like PENDING and RUNNING -- it is the regression this pins.
    assert 'req-waiting' in active_ids
    assert active_ids == {'req-pending', 'req-waiting', 'req-running'}


@pytest.mark.asyncio
async def test_shutdown_active_requests_matches_active_statuses(
        isolated_database):
    # The shutdown query must cover every non-terminal status, so no in-flight
    # request can slip past graceful shutdown unpreserved.
    backend = request_storage.get_request_backend()
    expected_ids = set()
    for i, status in enumerate(RequestStatus.active_statuses()):
        request_id = f'req-{i}'
        assert await requests.create_if_not_exists_async(
            _make_request(request_id, status))
        expected_ids.add(request_id)

    active_ids = {rid for rid, _ in backend.get_shutdown_active_requests()}

    # Set equality, not just count: a query that swapped one active status for
    # an equal number of others (the exact drift this guards) would still fail.
    assert active_ids == expected_ids
