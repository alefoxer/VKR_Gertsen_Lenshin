from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return values.astype(np.float32)
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def compute_saliency_map(adapter, audio: np.ndarray, target_word: str, smooth_kernel: int = 11) -> np.ndarray:
    gradient = adapter.compute_gradient(audio, target_word)
    saliency = np.abs(gradient).astype(np.float32)
    max_val = float(saliency.max()) if saliency.size else 0.0
    if max_val > 1e-8:
        saliency /= max_val
    return moving_average(saliency, smooth_kernel)
