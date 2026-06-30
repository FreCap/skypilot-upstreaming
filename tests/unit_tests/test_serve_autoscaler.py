"""Unit tests for sky.serve.autoscalers."""
import types
import unittest
from unittest import mock

from sky.serve import autoscalers
from sky.serve import constants
from sky.serve import replica_managers
from sky.serve import serve_state


class TestSelectNonterminalReplicasToScaleDown(unittest.TestCase):
    """Test cases for _select_nonterminal_replicas_to_scale_down."""

    def setUp(self):
        """Set up test fixtures."""
        self.service_name = 'test-service'

        # Create mock ReplicaInfo objects
        self.replica1 = mock.Mock(spec=replica_managers.ReplicaInfo)
        self.replica1.replica_id = 1
        self.replica1.cluster_name = 'test-cluster-1'
        self.replica1.version = 1
        self.replica1.status = serve_state.ReplicaStatus.READY

        self.replica2 = mock.Mock(spec=replica_managers.ReplicaInfo)
        self.replica2.replica_id = 2
        self.replica2.cluster_name = 'test-cluster-2'
        self.replica2.version = 1
        self.replica2.status = serve_state.ReplicaStatus.READY

        self.replica3 = mock.Mock(spec=replica_managers.ReplicaInfo)
        self.replica3.replica_id = 3
        self.replica3.cluster_name = 'test-cluster-3'
        self.replica3.version = 1
        self.replica3.status = serve_state.ReplicaStatus.READY

    @mock.patch('sky.serve.autoscalers.managed_job_state.'
                'get_nonterminal_job_counts_by_pool')
    def test_select_replicas_with_job_counts(self, mock_get_counts):
        """Test that replicas with fewer jobs are selected first."""

        # Mock job counts: replica1 has 2 jobs, replica2 has 0 jobs,
        # replica3 has 1 job
        mock_get_counts.return_value = {
            'test-cluster-1': 2,
            'test-cluster-3': 1,
            # test-cluster-2 absent means 0 jobs
        }

        replica_infos = [self.replica1, self.replica2, self.replica3]

        # Select 2 replicas to scale down
        result = autoscalers._select_nonterminal_replicas_to_scale_down(
            2, replica_infos, self.service_name)

        # Should select replica2 (0 jobs) and replica3 (1 job) first
        # Order should be: replica2 (0 jobs), replica3 (1 job), replica1
        # (2 jobs). Since we're selecting 2, we should get [2, 3]
        self.assertEqual(len(result), 2)
        self.assertEqual(result, [2, 3])

        # Verify the function was called once with the service name
        mock_get_counts.assert_called_once_with(self.service_name)

    @mock.patch('sky.serve.autoscalers.managed_job_state.'
                'get_nonterminal_job_counts_by_pool')
    def test_select_replicas_with_same_job_counts(self, mock_get_counts):
        """Test that when job counts are equal, other sorting criteria apply."""
        # All replicas have the same number of jobs
        mock_get_counts.return_value = {
            'test-cluster-1': 1,
            'test-cluster-2': 1,
            'test-cluster-3': 1,
        }

        replica_infos = [self.replica1, self.replica2, self.replica3]

        # Select 2 replicas to scale down
        result = autoscalers._select_nonterminal_replicas_to_scale_down(
            2, replica_infos, self.service_name)

        # When job counts are equal, should fall back to replica_id
        # descending order. So replica3 (id=3) and replica2 (id=2)
        # should be selected.
        self.assertEqual(len(result), 2)
        self.assertEqual(result, [3, 2])

    @mock.patch('sky.serve.autoscalers.managed_job_state.'
                'get_nonterminal_job_counts_by_pool')
    def test_select_replicas_with_status_priority(self, mock_get_counts):
        """Test that status priority is still respected."""
        # Create replicas with different statuses
        replica_provisioning = mock.Mock(spec=replica_managers.ReplicaInfo)
        replica_provisioning.replica_id = 1
        replica_provisioning.cluster_name = 'test-cluster-1'
        replica_provisioning.version = 1
        replica_provisioning.status = serve_state.ReplicaStatus.PROVISIONING

        replica_ready = mock.Mock(spec=replica_managers.ReplicaInfo)
        replica_ready.replica_id = 2
        replica_ready.cluster_name = 'test-cluster-2'
        replica_ready.version = 1
        replica_ready.status = serve_state.ReplicaStatus.READY

        # PROVISIONING replica has more jobs, but should still be selected
        # first
        mock_get_counts.return_value = {
            'test-cluster-1': 3,
            'test-cluster-2': 1,
        }

        replica_infos = [replica_provisioning, replica_ready]

        # Select 1 replica to scale down
        result = autoscalers._select_nonterminal_replicas_to_scale_down(
            1, replica_infos, self.service_name)

        # Should select PROVISIONING replica first despite having more jobs
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [1])

    @mock.patch('sky.serve.autoscalers.managed_job_state.'
                'get_nonterminal_job_counts_by_pool')
    def test_select_replicas_with_version_priority(self, mock_get_counts):
        """Test that version priority is still respected."""
        # Create replicas with different versions
        replica_old = mock.Mock(spec=replica_managers.ReplicaInfo)
        replica_old.replica_id = 1
        replica_old.cluster_name = 'test-cluster-1'
        replica_old.version = 1
        replica_old.status = serve_state.ReplicaStatus.READY

        replica_new = mock.Mock(spec=replica_managers.ReplicaInfo)
        replica_new.replica_id = 2
        replica_new.cluster_name = 'test-cluster-2'
        replica_new.version = 2
        replica_new.status = serve_state.ReplicaStatus.READY

        # New version replica has fewer jobs, but old version should be
        # selected first
        mock_get_counts.return_value = {
            'test-cluster-1': 2,
            # test-cluster-2 absent means 0 jobs
        }

        replica_infos = [replica_old, replica_new]

        # Select 1 replica to scale down
        result = autoscalers._select_nonterminal_replicas_to_scale_down(
            1, replica_infos, self.service_name)

        # Should select old version replica first despite having more jobs
        self.assertEqual(len(result), 1)
        self.assertEqual(result, [1])


class TestAutoscalerVersionInitialization(unittest.TestCase):
    """The autoscaler must be constructed at the recovered service version.

    On an API-server restart/respawn the controller is rebuilt and passes the
    durable latest version to `Autoscaler.from_spec`. If the autoscaler instead
    reset to INITIAL_VERSION (1) while live replicas are at version >= 2 (any
    service updated at least once), its version filters would treat every
    running replica as outdated and drive permanent replica churn. These tests
    pin that `version` is honored at construction and forwarded by `from_spec`.
    """

    def test_base_init_sets_latest_version_from_arg(self):
        spec = types.SimpleNamespace(min_replicas=1,
                                     max_replicas=3,
                                     num_overprovision=None)
        autoscaler = object.__new__(autoscalers.Autoscaler)
        autoscalers.Autoscaler.__init__(autoscaler, 'svc', spec, version=5)
        self.assertEqual(autoscaler.latest_version, 5)
        # Must stay one below latest so the unrecoverable-failure early-return
        # only arms once a replica at the latest version becomes ready.
        self.assertEqual(autoscaler.latest_version_ever_ready, 4)

    def test_base_init_defaults_to_initial_version(self):
        spec = types.SimpleNamespace(min_replicas=1,
                                     max_replicas=3,
                                     num_overprovision=None)
        autoscaler = object.__new__(autoscalers.Autoscaler)
        autoscalers.Autoscaler.__init__(autoscaler, 'svc', spec)
        self.assertEqual(autoscaler.latest_version, constants.INITIAL_VERSION)
        self.assertEqual(autoscaler.latest_version_ever_ready,
                         constants.INITIAL_VERSION - 1)

    def _route_spec(self, pool=False, use_ondemand_fallback=False,
                    target_qps_per_replica=2.0):
        return types.SimpleNamespace(
            pool=pool,
            use_ondemand_fallback=use_ondemand_fallback,
            target_qps_per_replica=target_qps_per_replica)

    def test_from_spec_forwards_version_request_rate(self):
        spec = self._route_spec()
        with mock.patch.object(autoscalers,
                               'RequestRateAutoscaler') as mock_cls:
            autoscalers.Autoscaler.from_spec('svc', spec, version=7)
        mock_cls.assert_called_once_with('svc', spec, 7)

    def test_from_spec_forwards_version_queue_length(self):
        spec = self._route_spec(pool=True)
        with mock.patch.object(autoscalers,
                               'QueueLengthAutoscaler') as mock_cls:
            autoscalers.Autoscaler.from_spec('svc', spec, version=7)
        mock_cls.assert_called_once_with('svc', spec, 7)

    def test_from_spec_forwards_version_fallback(self):
        spec = self._route_spec(use_ondemand_fallback=True)
        with mock.patch.object(autoscalers,
                               'FallbackRequestRateAutoscaler') as mock_cls:
            autoscalers.Autoscaler.from_spec('svc', spec, version=7)
        mock_cls.assert_called_once_with('svc', spec, 7)

    def test_from_spec_forwards_version_instance_aware(self):
        spec = self._route_spec(target_qps_per_replica={'A100': 2.0})
        with mock.patch.object(
                autoscalers,
                'InstanceAwareRequestRateAutoscaler') as mock_cls:
            autoscalers.Autoscaler.from_spec('svc', spec, version=7)
        mock_cls.assert_called_once_with('svc', spec, 7)

    def test_from_spec_defaults_to_initial_version(self):
        spec = self._route_spec()
        with mock.patch.object(autoscalers,
                               'RequestRateAutoscaler') as mock_cls:
            autoscalers.Autoscaler.from_spec('svc', spec)
        mock_cls.assert_called_once_with('svc', spec, constants.INITIAL_VERSION)

    def test_controller_passes_recovered_version_to_autoscaler(self):
        # Regression for the actual bug boundary: SkyServeController must hand
        # the recovered service version to the autoscaler factory (not drop it),
        # so a controller rebuilt on restart at version >= 2 doesn't churn.
        from sky.serve import controller as serve_controller
        with mock.patch.object(serve_controller.replica_managers,
                               'SkyPilotReplicaManager'), \
             mock.patch.object(serve_controller.autoscalers.Autoscaler,
                               'from_spec') as mock_from_spec:
            serve_controller.SkyServeController('svc',
                                                mock.MagicMock(),
                                                version=7,
                                                host='localhost',
                                                port=8000)
        mock_from_spec.assert_called_once()
        # version is the 3rd positional arg (service_name, service_spec, version)
        assert mock_from_spec.call_args.args[2] == 7


if __name__ == '__main__':
    unittest.main()
