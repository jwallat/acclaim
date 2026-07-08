"""
Shared thread-pool helper for parallelizing independent, synchronous LLM
judge calls.

litellm.completion() calls are synchronous and I/O-bound, so a thread pool
(not asyncio) is sufficient to get concurrency without rewriting the judge
call stack as async.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    max_workers: int = 1,
    *,
    desc: str | None = None,
) -> list[R]:
    """
    Apply ``fn`` to each item in ``items``, preserving input order in the output.

    When ``max_workers <= 1`` (or there's at most one item), falls back to a
    plain sequential loop — no thread pool is created, so this is identical
    in behavior and stack traces to a hand-written ``for`` loop.

    Args:
        fn:          Callable to apply to each item. Must be safe to call
                     concurrently from multiple threads.
        items:       Input items, processed in this order.
        max_workers: Number of worker threads. ``<= 1`` means sequential.
        desc:        Progress bar label. A bar is only shown when this is
                     given — callers that don't pass ``desc`` (e.g. nested
                     per-claim/per-citation calls) stay silent so a batch
                     run shows exactly one bar, not one per nesting level.

    Returns:
        Results in the same order as ``items``.
    """
    show_progress = desc is not None
    if max_workers <= 1 or len(items) <= 1:
        return [
            fn(item) for item in tqdm(items, desc=desc, disable=not show_progress)
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            tqdm(
                executor.map(fn, items),
                total=len(items),
                desc=desc,
                disable=not show_progress,
            )
        )
