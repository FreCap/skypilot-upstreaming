"""Utility functions for threads."""

import threading
import time
from typing import Any, Callable, Dict, Generic, Optional, overload, TypeVar

from sky import sky_logging
from sky.utils import common_utils

logger = sky_logging.init_logger(__name__)

_DEFAULT_THREAD_RESTART_BACKOFF_SECONDS = 5.0


def start_supervised_thread(
    target: Callable[[], Any],
    name: str,
    restart_backoff_seconds: float = _DEFAULT_THREAD_RESTART_BACKOFF_SECONDS,
    stop_event: Optional[threading.Event] = None,
) -> threading.Thread:
    """Run ``target`` in a background thread, restarting it if it ever exits.

    Several long-lived control-loop duties (e.g. the serve autoscaler and the
    replica manager's refresher / prober / job-status fetcher) run as bare
    ``threading.Thread``s whose bodies are ``while True: try/except Exception``.
    They survive ordinary exceptions, but a ``BaseException`` escaping (or the
    target returning at all) silently ends the thread while the host process
    keeps serving HTTP -- a "half-dead" controller where, for example,
    autoscaling never happens again, failed replicas keep getting traffic, or
    spot preemptions stop being reaped, and only a full process restart clears
    it.

    This wraps ``target`` so any exit -- a normal return OR a ``BaseException``
    -- is logged and the target is re-run after a short backoff, so the duty
    cannot silently stop. Returns the supervisor thread.

    ``stop_event``: if provided, the supervisor stops restarting (and the
    thread exits) once it is set, checked between restarts. The serve call
    sites pass none -- they restart forever and are torn down with the process
    -- but it makes graceful shutdown and testing possible.
    """

    def _keep_running() -> bool:
        return stop_event is None or not stop_event.is_set()

    def _supervise() -> None:
        while _keep_running():
            try:
                target()
                if not _keep_running():
                    break
                # The wrapped duties are infinite loops; a normal return is
                # itself unexpected, so restart it too.
                logger.error(
                    f'Supervised thread {name!r} returned unexpectedly (its '
                    f'loop should never exit); restarting after '
                    f'{restart_backoff_seconds}s.')
            except BaseException as e:  # pylint: disable=broad-except
                if not _keep_running():
                    break
                # NOTE: not common_utils.format_exception, whose signature does
                # not admit an arbitrary BaseException.
                logger.error(f'Supervised thread {name!r} died with {e!r}; '
                             f'restarting after {restart_backoff_seconds}s.')
            # Interruptible backoff so a stop is honored promptly.
            if stop_event is not None:
                stop_event.wait(restart_backoff_seconds)
            else:
                time.sleep(restart_backoff_seconds)

    # Match the non-daemon semantics of the bare threads this replaces; the
    # serve controller is torn down via signal / os._exit (force-killed), not a
    # clean interpreter shutdown, so daemon-ness is moot for the real exit path.
    thread = threading.Thread(target=_supervise, name=f'supervised-{name}')
    thread.start()
    return thread


class SafeThread(threading.Thread):
    """A thread that can catch exceptions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exc = None

    def run(self):
        try:
            super().run()
        except BaseException as e:  # pylint: disable=broad-except
            self._exc = e

    @property
    def format_exc(self) -> Optional[str]:
        if self._exc is None:
            return None
        return common_utils.format_exception(self._exc)


# pylint: disable=invalid-name
KeyType = TypeVar('KeyType')
ValueType = TypeVar('ValueType')


# Google style guide: Do not rely on the atomicity of built-in types.
# Our launch and down process pool will be used by multiple threads,
# therefore we need to use a thread-safe dict.
# see https://google.github.io/styleguide/pyguide.html#218-threading
class ThreadSafeDict(Generic[KeyType, ValueType]):
    """A thread-safe dict."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._dict: Dict[KeyType, ValueType] = dict(*args, **kwargs)
        self._lock = threading.Lock()

    def __getitem__(self, key: KeyType) -> ValueType:
        with self._lock:
            return self._dict.__getitem__(key)

    def __setitem__(self, key: KeyType, value: ValueType) -> None:
        with self._lock:
            return self._dict.__setitem__(key, value)

    def __delitem__(self, key: KeyType) -> None:
        with self._lock:
            return self._dict.__delitem__(key)

    def __len__(self) -> int:
        with self._lock:
            return self._dict.__len__()

    def __contains__(self, key: KeyType) -> bool:
        with self._lock:
            return self._dict.__contains__(key)

    def items(self):
        with self._lock:
            return self._dict.items()

    def values(self):
        with self._lock:
            return self._dict.values()

    @overload
    def get(self, key: KeyType, default: ValueType) -> ValueType:
        ...

    @overload
    def get(self,
            key: KeyType,
            default: Optional[ValueType] = None) -> Optional[ValueType]:
        ...

    def get(self,
            key: KeyType,
            default: Optional[ValueType] = None) -> Optional[ValueType]:
        with self._lock:
            return self._dict.get(key, default)

    def pop(self, key: KeyType) -> Optional[ValueType]:
        with self._lock:
            return self._dict.pop(key, None)
