"""Unit tests for sky.utils.annotations."""

import threading
import time

from sky.utils import annotations


class TestLruCache:
    """Tests for lru_cache decorator."""

    def test_caching_works(self):
        """Test that lru_cache decorator caches function results."""
        call_count = 0

        @annotations.lru_cache(scope='global', maxsize=5)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        expensive_func.cache_clear()

        # First call
        result1 = expensive_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call with same arg should use cache
        result2 = expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented

        # Call with different arg should not use cache
        result3 = expensive_func(10)
        assert result3 == 20
        assert call_count == 2

    def test_request_scope_cache_clear(self):
        """Test that request-scope cache is registered for clearing."""

        @annotations.lru_cache(scope='request', maxsize=5)
        def request_scoped_func(x):
            return x + 1

        # The function should be in the _FUNCTIONS_NEED_RELOAD_CACHE list
        assert any(f().__name__ == 'request_scoped_func' or
                   hasattr(f(), '__wrapped__')
                   for f in annotations._FUNCTIONS_NEED_RELOAD_CACHE
                   if f() is not None)

    def test_cache_clear_method(self):
        """Test that cache_clear method works."""
        call_count = 0

        @annotations.lru_cache(scope='global', maxsize=5)
        def func_to_clear(x):
            nonlocal call_count
            call_count += 1
            return x

        func_to_clear.cache_clear()

        # First call
        result1 = func_to_clear(5)
        assert result1 == 5
        assert call_count == 1

        # Use cache
        result2 = func_to_clear(5)
        assert result2 == 5
        assert call_count == 1

        # Clear and call again
        func_to_clear.cache_clear()
        result3 = func_to_clear(5)
        assert result3 == 5
        assert call_count == 2  # Called again after clear


class TestTtlCache:
    """Tests for ttl_cache decorator."""

    def test_caching_works(self):
        """Test that ttl_cache decorator caches function results."""
        call_count = 0

        @annotations.ttl_cache(scope='global',
                               timer=time.time,
                               maxsize=5,
                               ttl=60)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        assert expensive_func(5) == 10
        assert call_count == 1

        # Second call with same arg should use cache
        assert expensive_func(5) == 10
        assert call_count == 1  # Not incremented

        # Call with different arg should not use cache
        assert expensive_func(10) == 20
        assert call_count == 2

    def test_cache_is_lock_guarded(self):
        """Test that the shared TTLCache is guarded by a lock.

        cachetools' memoizing decorators are not thread-safe without a lock,
        and ttl_cache'd functions may be called concurrently (e.g. the
        per-cloud catalog fan-out in subprocess_utils.run_in_parallel).
        """

        @annotations.ttl_cache(scope='global',
                               timer=time.time,
                               maxsize=5,
                               ttl=60)
        def cached_func(x):
            return x

        assert cached_func.cache_lock is not None

    def test_concurrent_access(self):
        """Concurrent calls with expiring entries must not corrupt the cache.

        Without a lock, interleaved TTLCache writes/expirations can corrupt
        the cache's internal linked list and raise KeyError.
        """

        @annotations.ttl_cache(scope='global',
                               timer=time.time,
                               maxsize=4,
                               ttl=0.01)
        def cached_func(x):
            return x

        errors = []

        def worker(offset):
            try:
                for i in range(500):
                    key = (offset + i) % 8
                    assert cached_func(key) == key
            except Exception as e:  # pylint: disable=broad-except
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,)) for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors, f'Concurrent cache access failed: {errors}'

    def test_request_scope_cache_clear(self):
        """Test that request-scope ttl_cache is cleared with the request."""
        call_count = 0

        @annotations.ttl_cache(scope='request',
                               timer=time.time,
                               maxsize=5,
                               ttl=60)
        def request_scoped_func(x):
            nonlocal call_count
            call_count += 1
            return x + 1

        assert request_scoped_func(1) == 2
        assert request_scoped_func(1) == 2
        assert call_count == 1

        annotations.clear_request_level_cache()
        assert request_scoped_func(1) == 2
        assert call_count == 2  # Called again after clear
