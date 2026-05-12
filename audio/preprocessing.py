from __future__ import annotations

from typing import Any, Dict

import numpy as np

from utils.config import AudioConfig


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        return (audio / peak).astype(np.float32)
    return audio.astype(np.float32)


def clip_audio(audio: np.ndarray, min_value: float = -1.0, max_value: float = 1.0) -> np.ndarray:
    return np.clip(audio, min_value, max_value).astype(np.float32)


def describe_preprocessing(config: AudioConfig, source_sample_rate: int | None = None, input_mode: str = "") -> Dict[str, Any]:
    return {
        "input_mode": input_mode,
        "source_sample_rate": source_sample_rate,
        "target_sample_rate": config.sample_rate,
        "mono": config.mono,
        "normalize_peak": config.normalize,
        "min_duration_sec": config.min_duration_sec,
        "default_duration_sec": config.default_duration_sec,
        "window_sec": config.window_sec,
        "hop_sec": config.hop_sec,
        "model_input": "raw waveform in [-1, 1]",
    }
