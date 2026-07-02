"""Regression test: graceful shutdown must preserve WAITING requests.

``get_shutdown_active_requests`` returns the requests the graceful-shutdown
coordinator waits for and (for retryable ones) marks ``should_retry`` so the
client re-issues them after restart. ``WAITING`` is a non-terminal status (a
request parked on a retry backoff / external continue-condition, with its
resume timer living only in an in-memory monitor thread). If the shutdown query
omits WAITING, a parked request is neither waited for nor flagged for retry, its
in-memory timer dies with the process, and ``reset_db_and_logs`` wipes the row
on the next boot -- silently dropping the request on a *clean* restart even
though the whole ``should_retry`` machinery exists. This test pins that
WAITING is treated like the other active statuses.
"""
# pylint: disable=protected-access
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
async def test_shutdown_query_returns_exactly_the_active_requests(
        isolated_database):
    backend = request_storage.get_request_backend()
    for status in RequestStatus:
        assert await requests.create_if_not_exists_async(
            _make_request(f'req-{status.value.lower()}', status))

    active_ids = {rid for rid, _ in backend.get_shutdown_active_requests()}

    # A WAITING request must be waited for / retried on graceful shutdown --
    # it is the regression this pins.
    assert 'req-waiting' in active_ids
    # And the query must cover exactly the active (non-terminal) statuses, so
    # no in-flight request can slip past graceful shutdown unpreserved.
    assert active_ids == {
        f'req-{status.value.lower()}'
        for status in RequestStatus.active_statuses()
    }
