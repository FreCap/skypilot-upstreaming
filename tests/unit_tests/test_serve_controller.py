"""Tests for sky/serve/controller.py.

Currently focused on `SkyServeController._get_lb_replica_info`, which builds
the `/controller/load_balancer_sync` response. Resolving a replica's url and
gpu_type is expensive (cluster handle fetch + endpoint query), so both must
be resolved at most once per replica lifetime and cached; the cache must be
pruned when a replica leaves the ready set.
"""
from typing import Dict, Optional
from unittest import mock

from sky.serve import controller
from sky.serve import serve_state


class _FakeHandle:
    """Stub for the resource handle returned by ReplicaInfo.handle()."""

    def __init__(self, accelerators: Optional[Dict[str, int]]) -> None:
        self.launched_resources = mock.Mock()
        self.launched_resources.accelerators = accelerators


class _FakeReplicaInfo:
    """ReplicaInfo stub that counts expensive url/handle resolutions."""

    def __init__(self,
                 replica_id: int,
                 status: serve_state.ReplicaStatus,
                 version: int = 1,
                 url: Optional[str] = None,
                 accelerators: Optional[Dict[str, int]] = None,
                 handle_is_none: bool = False) -> None:
        self.replica_id = replica_id
        self.status = status
        self.version = version
        self._url = url
        self._accelerators = accelerators
        self._handle_is_none = handle_is_none
        self.url_resolutions = 0
        self.handle_resolutions = 0

    @property
    def url(self) -> Optional[str]:
        self.url_resolutions += 1
        return self._url

    def handle(self) -> Optional[_FakeHandle]:
        self.handle_resolutions += 1
        if self._handle_is_none:
            return None
        return _FakeHandle(self._accelerators)


def _make_controller() -> controller.SkyServeController:
    # Bypass __init__: it builds a real replica manager and autoscaler.
    ctrl = controller.SkyServeController.__new__(controller.SkyServeController)
    ctrl._service_name = 'svc'  # pylint: disable=protected-access
    ctrl._lb_replica_cache = {}  # pylint: disable=protected-access
    return ctrl


def _sync(ctrl: controller.SkyServeController, infos,
          active_versions=(1,)) -> Dict[str, Dict[str, str]]:
    record = {'active_versions': list(active_versions)}
    with mock.patch.object(controller.serve_state,
                           'get_service_from_name',
                           return_value=record), \
         mock.patch.object(controller.serve_state,
                           'get_replica_infos',
                           return_value=infos):
        return ctrl._get_lb_replica_info()  # pylint: disable=protected-access


class TestGetLbReplicaInfo:
    """Tests for the (url, gpu_type) per-replica cache behind the
    /controller/load_balancer_sync response."""

    def test_resolves_url_and_gpu_type_for_ready_replicas(self):
        ctrl = _make_controller()
        infos = [
            _FakeReplicaInfo(1,
                             serve_state.ReplicaStatus.READY,
                             url='http://1.1.1.1:8080',
                             accelerators={'L4': 1}),
            _FakeReplicaInfo(2,
                             serve_state.ReplicaStatus.READY,
                             url='http://2.2.2.2:8080',
                             accelerators={'A100': 8}),
        ]
        assert _sync(ctrl, infos) == {
            'http://1.1.1.1:8080': {
                'gpu_type': 'L4'
            },
            'http://2.2.2.2:8080': {
                'gpu_type': 'A100'
            },
        }

    def test_resolution_happens_at_most_once_per_replica(self):
        ctrl = _make_controller()
        info = _FakeReplicaInfo(1,
                                serve_state.ReplicaStatus.READY,
                                url='http://1.1.1.1:8080',
                                accelerators={'L4': 1})
        first = _sync(ctrl, [info])
        second = _sync(ctrl, [info])
        assert first == second
        # url and handle must be resolved on the first sync only.
        assert info.url_resolutions == 1
        assert info.handle_resolutions == 1

    def test_not_ready_replicas_are_never_resolved(self):
        ctrl = _make_controller()
        provisioning = _FakeReplicaInfo(1,
                                        serve_state.ReplicaStatus.PROVISIONING)
        ready = _FakeReplicaInfo(2,
                                 serve_state.ReplicaStatus.READY,
                                 url='http://2.2.2.2:8080',
                                 accelerators={'L4': 1})
        result = _sync(ctrl, [provisioning, ready])
        assert result == {'http://2.2.2.2:8080': {'gpu_type': 'L4'}}
        assert provisioning.url_resolutions == 0
        assert provisioning.handle_resolutions == 0

    def test_inactive_version_replicas_are_excluded(self):
        ctrl = _make_controller()
        outdated = _FakeReplicaInfo(1,
                                    serve_state.ReplicaStatus.READY,
                                    version=1,
                                    url='http://1.1.1.1:8080',
                                    accelerators={'L4': 1})
        current = _FakeReplicaInfo(2,
                                   serve_state.ReplicaStatus.READY,
                                   version=2,
                                   url='http://2.2.2.2:8080',
                                   accelerators={'L4': 1})
        result = _sync(ctrl, [outdated, current], active_versions=(2,))
        assert result == {'http://2.2.2.2:8080': {'gpu_type': 'L4'}}
        assert outdated.url_resolutions == 0

    def test_unknown_gpu_type_when_handle_is_missing(self):
        ctrl = _make_controller()
        info = _FakeReplicaInfo(1,
                                serve_state.ReplicaStatus.READY,
                                url='http://1.1.1.1:8080',
                                handle_is_none=True)
        assert _sync(ctrl, [info]) == {
            'http://1.1.1.1:8080': {
                'gpu_type': 'unknown'
            }
        }

    def test_unknown_gpu_type_when_no_accelerators(self):
        ctrl = _make_controller()
        info = _FakeReplicaInfo(1,
                                serve_state.ReplicaStatus.READY,
                                url='http://1.1.1.1:8080',
                                accelerators=None)
        assert _sync(ctrl, [info]) == {
            'http://1.1.1.1:8080': {
                'gpu_type': 'unknown'
            }
        }

    def test_cache_pruned_when_replica_leaves_ready_set(self):
        ctrl = _make_controller()
        info = _FakeReplicaInfo(1,
                                serve_state.ReplicaStatus.READY,
                                url='http://1.1.1.1:8080',
                                accelerators={'L4': 1})
        _sync(ctrl, [info])
        assert 1 in ctrl._lb_replica_cache  # pylint: disable=protected-access

        # The replica gets preempted: it must be dropped from the response
        # and pruned from the cache.
        preempted = _FakeReplicaInfo(1,
                                     serve_state.ReplicaStatus.NOT_READY,
                                     url='http://1.1.1.1:8080')
        assert not _sync(ctrl, [preempted])
        assert not ctrl._lb_replica_cache  # pylint: disable=protected-access

        # The replica recovers with a new endpoint: it must be re-resolved
        # instead of served from a stale cache entry.
        recovered = _FakeReplicaInfo(1,
                                     serve_state.ReplicaStatus.READY,
                                     url='http://3.3.3.3:8080',
                                     accelerators={'L4': 1})
        assert _sync(ctrl, [recovered]) == {
            'http://3.3.3.3:8080': {
                'gpu_type': 'L4'
            }
        }
        assert recovered.url_resolutions == 1
