"""SkyServeController: the central controller of SkyServe.

Responsible for autoscaling and replica management.
"""
import contextlib
import logging
import os
import threading
import time
import traceback
from typing import Any, Dict, List, Tuple

import colorama
import fastapi
from fastapi import responses
import uvicorn

from sky import serve
from sky import sky_logging
from sky.serve import autoscalers
from sky.serve import replica_managers
from sky.serve import serve_state
from sky.serve import serve_utils
from sky.skylet import constants
from sky.utils import common_utils
from sky.utils import context_utils
from sky.utils import ux_utils

logger = sky_logging.init_logger(__name__)


class AutoscalerInfoFilter(logging.Filter):

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ('GET' in message and '200' in message and
                    '/autoscaler/info' in message)


class SkyServeController:
    """SkyServeController: control everything about replica.

    This class is responsible for:
        - Starting and terminating the replica monitor and autoscaler.
        - Providing the HTTP Server API for SkyServe to communicate with.
    """

    def __init__(self, service_name: str, service_spec: serve.SkyServiceSpec,
                 version: int, host: str, port: int) -> None:
        self._service_name = service_name
        self._replica_manager: replica_managers.ReplicaManager = (
            replica_managers.SkyPilotReplicaManager(service_name=service_name,
                                                    spec=service_spec,
                                                    version=version))
        self._autoscaler: autoscalers.Autoscaler = (
            autoscalers.Autoscaler.from_spec(service_name, service_spec))
        self._host = host
        self._port = port
        # Cache of replica_id -> (url, gpu_type) for the load_balancer_sync
        # response. Both fields require a cluster handle fetch (and, for the
        # url, an endpoint query) and are fixed for a replica's lifetime once
        # it is READY, so they are resolved at most once per replica. The
        # cache is rebuilt from the currently active replicas on every sync,
        # which prunes replicas that are no longer READY; a replica that
        # recovers with a new endpoint is thus re-resolved.
        self._lb_replica_cache: Dict[int, Tuple[str, str]] = {}
        self._app = fastapi.FastAPI(lifespan=self.lifespan)

    @contextlib.asynccontextmanager
    async def lifespan(self, _: fastapi.FastAPI):
        uvicorn_access_logger = logging.getLogger('uvicorn.access')
        for handler in uvicorn_access_logger.handlers:
            handler.setFormatter(sky_logging.FORMATTER)
            handler.addFilter(AutoscalerInfoFilter())
        yield

    def _get_lb_replica_info(self) -> Dict[str, Dict[str, str]]:
        """Build the url -> replica info mapping for load_balancer_sync.

        Resolving a replica's url and gpu_type is expensive (a cluster handle
        fetch plus, for the url, an endpoint query against a database the
        launch threads contend on), so both are cached per replica for the
        replica's lifetime: only newly-READY replicas are resolved on a sync.
        A brand-new replica whose gpu_type cannot be resolved yet is reported
        as 'unknown' until it is.
        """
        record = serve_state.get_service_from_name(self._service_name)
        assert record is not None, ('No service record found for '
                                    f'{self._service_name}')
        active_versions = set(record['active_versions'])
        replica_cache: Dict[int, Tuple[str, str]] = {}
        replica_info: Dict[str, Dict[str, str]] = {}
        for info in serve_state.get_replica_infos(self._service_name):
            if (info.status != serve_state.ReplicaStatus.READY or
                    info.version not in active_versions):
                continue
            cached = self._lb_replica_cache.get(info.replica_id)
            if cached is None:
                url = info.url
                assert url is not None, info
                # gpu_type is used by instance-aware load balancing policies.
                # It derives from the replica's launched accelerators, which
                # are fixed for the replica's lifetime.
                gpu_type = 'unknown'
                handle = info.handle()
                if handle is not None:
                    accelerators = handle.launched_resources.accelerators
                    if accelerators:
                        gpu_type = list(accelerators.keys())[0]
                cached = (url, gpu_type)
            replica_cache[info.replica_id] = cached
            url, gpu_type = cached
            replica_info[url] = {'gpu_type': gpu_type}
        # Replacing the cache with this sync's active replicas prunes the
        # replicas that are no longer READY.
        self._lb_replica_cache = replica_cache
        return replica_info

    def _run_autoscaler(self):
        logger.info('Starting autoscaler.')
        while True:
            try:
                replica_infos = serve_state.get_replica_infos(
                    self._service_name)
                # Use the active versions set by replica manager to make
                # sure we only scale down the outdated replicas that are
                # not used by the load balancer.
                record = serve_state.get_service_from_name(self._service_name)
                assert record is not None, ('No service record found for '
                                            f'{self._service_name}')
                active_versions = record['active_versions']
                logger.info(f'All replica info for autoscaler: {replica_infos}')

                # Autoscaler now extracts GPU type info directly from
                # replica_infos in generate_scaling_decisions method
                # for better decoupling.
                scaling_options = self._autoscaler.generate_scaling_decisions(
                    replica_infos, active_versions)
                for scaling_option in scaling_options:
                    logger.info(f'Scaling option received: {scaling_option}')
                    if (scaling_option.operator ==
                            autoscalers.AutoscalerDecisionOperator.SCALE_UP):
                        assert (scaling_option.target is None or isinstance(
                            scaling_option.target, dict)), scaling_option
                        self._replica_manager.scale_up(scaling_option.target)
                    else:
                        assert isinstance(scaling_option.target,
                                          int), scaling_option
                        self._replica_manager.scale_down(scaling_option.target)
            except Exception as e:  # pylint: disable=broad-except
                # No matter what error happens, we should keep the
                # monitor running.
                logger.error('Error in autoscaler: '
                             f'{common_utils.format_exception(e)}')
                with ux_utils.enable_traceback():
                    logger.error(f'  Traceback: {traceback.format_exc()}')
            time.sleep(self._autoscaler.get_decision_interval())

    def run(self) -> None:

        @self._app.get('/autoscaler/info')
        async def get_autoscaler_info() -> fastapi.Response:
            return responses.JSONResponse(content=self._autoscaler.info(),
                                          status_code=200)

        @self._app.post('/controller/load_balancer_sync')
        async def load_balancer_sync(
                request: fastapi.Request) -> fastapi.Response:
            request_data = await request.json()
            # TODO(MaoZiming): Check aggregator type.
            request_aggregator: Dict[str, Any] = request_data.get(
                'request_aggregator', {})
            timestamps: List[int] = request_aggregator.get('timestamps', [])
            logger.info(f'Received {len(timestamps)} inflight requests.')
            self._autoscaler.collect_request_information(request_aggregator)

            return responses.JSONResponse(
                content={'replica_info': self._get_lb_replica_info()},
                status_code=200)

        @self._app.post('/controller/update_service')
        async def update_service(request: fastapi.Request) -> fastapi.Response:
            request_data = await request.json()
            try:
                version = request_data.get('version', None)
                if version is None:
                    return responses.JSONResponse(
                        content={'message': 'Error: version is not specified.'},
                        status_code=400)
                update_mode_str = request_data.get(
                    'mode', serve_utils.DEFAULT_UPDATE_MODE.value)
                update_mode = serve_utils.UpdateMode(update_mode_str)
                logger.info(f'Update to new version {version} with '
                            f'update_mode {update_mode}.')
                # The yaml with the name latest_task_yaml will be synced
                # See sky/serve/core.py::update
                latest_task_yaml = serve_utils.generate_task_yaml_file_name(
                    self._service_name, version)
                with open(latest_task_yaml, 'r', encoding='utf-8') as f:
                    yaml_content = f.read()
                service = serve.SkyServiceSpec.from_yaml_str(yaml_content)
                serve_state.add_or_update_version(self._service_name, version,
                                                  service, yaml_content)
                logger.info(
                    f'Update to new version version {version}: {service}')

                self._replica_manager.update_version(version,
                                                     service,
                                                     update_mode=update_mode)
                new_autoscaler = autoscalers.Autoscaler.from_spec(
                    self._service_name, service)
                if not isinstance(self._autoscaler, type(new_autoscaler)):
                    logger.info('Autoscaler type changed to '
                                f'{type(new_autoscaler)}, updating autoscaler.')
                    old_autoscaler = self._autoscaler
                    self._autoscaler = new_autoscaler
                    self._autoscaler.load_dynamic_states(
                        old_autoscaler.dump_dynamic_states())
                self._autoscaler.update_version(version,
                                                service,
                                                update_mode=update_mode)
                return responses.JSONResponse(content={'message': 'Success'},
                                              status_code=200)
            except Exception as e:  # pylint: disable=broad-except
                exception_str = common_utils.format_exception(e)
                logger.error(f'Error in update_service: {exception_str}')
                return responses.JSONResponse(content={
                    'message': 'Error',
                    'exception': exception_str,
                    'traceback': traceback.format_exc()
                },
                                              status_code=500)

        @self._app.post('/controller/terminate_replica')
        async def terminate_replica(
                request: fastapi.Request) -> fastapi.Response:
            request_data = await request.json()
            replica_id = request_data['replica_id']
            assert isinstance(replica_id,
                              int), 'Error: replica ID must be an integer.'
            purge = request_data['purge']
            assert isinstance(purge, bool), 'Error: purge must be a boolean.'
            replica_info = serve_state.get_replica_info_from_id(
                self._service_name, replica_id)
            assert replica_info is not None, (f'Error: replica '
                                              f'{replica_id} does not exist.')
            replica_status = replica_info.status

            if replica_status == serve_state.ReplicaStatus.SHUTTING_DOWN:
                return responses.JSONResponse(
                    status_code=409,
                    content={
                        'message':
                            f'Replica {replica_id} of service '
                            f'{self._service_name!r} is already in the process '
                            f'of terminating. Skip terminating now.'
                    })

            if (replica_status in serve_state.ReplicaStatus.failed_statuses()
                    and not purge):
                return responses.JSONResponse(
                    status_code=409,
                    content={
                        'message': f'{colorama.Fore.YELLOW}Replica '
                                   f'{replica_id} of service '
                                   f'{self._service_name!r} is in failed '
                                   f'status ({replica_info.status}). '
                                   f'Skipping its termination as it could '
                                   f'lead to a resource leak. '
                                   f'(Use `sky serve down '
                                   f'{self._service_name!r} --replica-id '
                                   f'{replica_id} --purge` to '
                                   'forcefully terminate the replica.)'
                                   f'{colorama.Style.RESET_ALL}'
                    })

            self._replica_manager.scale_down(replica_id, purge=purge)

            action = 'terminated' if not purge else 'purged'
            message = (f'{colorama.Fore.GREEN}Replica {replica_id} of service '
                       f'{self._service_name!r} is scheduled to be '
                       f'{action}.{colorama.Style.RESET_ALL}\n'
                       f'Please use {ux_utils.BOLD}sky serve status '
                       f'{self._service_name}{ux_utils.RESET_BOLD} '
                       f'to check the latest status.')
            return responses.JSONResponse(status_code=200,
                                          content={'message': message})

        @self._app.exception_handler(Exception)
        async def validation_exception_handler(
                request: fastapi.Request, exc: Exception) -> fastapi.Response:
            with ux_utils.enable_traceback():
                logger.error(f'Error in controller: {exc!r}')
            return responses.JSONResponse(
                status_code=500,
                content={
                    'message':
                        (f'Failed method {request.method} at URL {request.url}.'
                         f' Exception message is {exc!r}.')
                },
            )

        threading.Thread(target=self._run_autoscaler).start()

        logger.info('SkyServe Controller started on '
                    f'http://{self._host}:{self._port}. PID: {os.getpid()}')

        uvicorn.run(self._app, host=self._host, port=self._port)


# TODO(tian): Probably we should support service that will stop the VM in
# specific time period.
def run_controller(service_name: str, service_spec: serve.SkyServiceSpec,
                   version: int, controller_host: str, controller_port: int):
    os.environ[constants.OVERRIDE_CONSOLIDATION_MODE] = 'true'
    # Hijack sys.stdout/stderr to be context aware.
    context_utils.hijack_sys_attrs()
    controller = SkyServeController(service_name, service_spec, version,
                                    controller_host, controller_port)
    controller.run()
