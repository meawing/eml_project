"""CUDA timing helper.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


@torch.no_grad()
def time_ms(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    repeats: int = 50,
    batch: int = 1,
) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    ts = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(batch):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / batch)

    arr = np.asarray(ts, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25, 75])
    return {
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "iqr_ms": float(q75 - q25),
    }
