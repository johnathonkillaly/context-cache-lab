"""Wall-clock timing that is actually correct on MPS.

MPS dispatch is asynchronous. Reading `time.perf_counter()` without synchronizing first
measures how long it took to *enqueue* the work, which on short prefills can be off by
an order of magnitude. Every timed region in this repo goes through `sync()`.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from typing import Callable, Iterator, TypeVar

T = TypeVar("T")


def sync() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        elif torch.cuda.is_available():  # pragma: no cover - no CUDA on this box
            torch.cuda.synchronize()
    except ImportError:  # pragma: no cover
        pass


@contextmanager
def timed(out: dict, key: str) -> Iterator[None]:
    """Record seconds elapsed in `out[key]`, synchronizing on both edges."""
    sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sync()
        out[key] = time.perf_counter() - t0


def measure(fn: Callable[[], T], *, trials: int = 5, warmup: int = 2) -> tuple[T, dict]:
    """Median-of-`trials` wall clock after `warmup` untimed runs.

    Median, not mean: the first timed run on MPS often pays for kernel specialization
    and shader caching, and one such outlier badly skews a mean over 5 samples.
    """
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    result: T | None = None
    for _ in range(trials):
        sync()
        t0 = time.perf_counter()
        result = fn()
        sync()
        samples.append(time.perf_counter() - t0)
    stats = {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "max_s": max(samples),
        "n_trials": trials,
        "n_warmup": warmup,
        "samples_s": samples,
    }
    return result, stats  # type: ignore[return-value]
