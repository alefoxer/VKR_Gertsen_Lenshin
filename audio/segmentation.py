from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from utils.config import AudioConfig


@dataclass
class TimeWindow:
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float


def build_time_windows(num_samples: int, config: AudioConfig) -> List[TimeWindow]:
    window_size = max(1, int(config.window_sec * config.sample_rate))
    hop_size = max(1, int(config.hop_sec * config.sample_rate))
    windows = []
    start = 0
    while start < num_samples:
        end = min(num_samples, start + window_size)
        windows.append(TimeWindow(start, end, start / config.sample_rate, end / config.sample_rate))
        if end >= num_samples:
            break
        start += hop_size
    return windows


def build_patch_positions(num_samples: int, sample_rate: int, patch_duration_sec: float, patch_stride_sec: float) -> list[int]:
    patch_len = max(1, int(sample_rate * patch_duration_sec))
    stride = max(1, int(sample_rate * patch_stride_sec))
    if patch_len >= num_samples:
        return [0]
    return list(range(0, max(1, num_samples - patch_len + 1), stride))


def window_means(values: np.ndarray, windows: Iterable[TimeWindow]) -> np.ndarray:
    out = []
    for window in windows:
        frag = values[window.start_sample:window.end_sample]
        out.append(float(frag.mean()) if frag.size else 0.0)
    return np.asarray(out, dtype=np.float32)
