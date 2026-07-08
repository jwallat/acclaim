"""Unit tests for the parallel_map thread-pool helper."""

from __future__ import annotations

import threading
import time

import pytest

from acclaim.concurrency import parallel_map


def test_sequential_preserves_order_max_workers_1() -> None:
    assert parallel_map(lambda x: x * 2, [1, 2, 3], max_workers=1) == [2, 4, 6]


def test_parallel_preserves_order_max_workers_4() -> None:
    assert parallel_map(lambda x: x * 2, [1, 2, 3, 4], max_workers=4) == [2, 4, 6, 8]


def test_empty_input() -> None:
    assert parallel_map(lambda x: x, [], max_workers=4) == []


def test_max_workers_1_does_not_spawn_threads() -> None:
    """Sequential path must not touch ThreadPoolExecutor at all."""
    seen_threads = set()

    def _record(x: int) -> int:
        seen_threads.add(threading.current_thread().ident)
        return x

    parallel_map(_record, [1, 2, 3], max_workers=1)
    assert seen_threads == {threading.current_thread().ident}


def test_parallel_speedup_with_artificial_delay() -> None:
    """Concurrent execution of slow calls should be faster than sequential."""
    delay = 0.05

    def _slow(x: int) -> int:
        time.sleep(delay)
        return x

    items = list(range(8))

    start = time.perf_counter()
    parallel_map(_slow, items, max_workers=1)
    sequential_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    parallel_map(_slow, items, max_workers=8)
    parallel_elapsed = time.perf_counter() - start

    assert parallel_elapsed < sequential_elapsed / 2


def test_exception_propagates() -> None:
    def _maybe_raise(x: int) -> int:
        if x == 2:
            raise ValueError("boom")
        return x

    with pytest.raises(ValueError, match="boom"):
        parallel_map(_maybe_raise, [1, 2, 3], max_workers=4)
