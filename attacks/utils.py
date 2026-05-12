from __future__ import annotations

import numpy as np
import torch


def normalize_map(values: np.ndarray) -> np.ndarray:
    arr = np.abs(values).astype(np.float32)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1e-8:
        arr /= max_val
    return arr


def clamp_delta(delta_tensor: torch.Tensor, max_delta: float) -> torch.Tensor:
    return torch.clamp(delta_tensor, -max_delta, max_delta)


def create_seed_audio(config, audio_config, uploaded_audio=None):
    import numpy as np
    n = int(audio_config.sample_rate * audio_config.default_duration_sec)
    if config.input_mode == "attack_uploaded_audio" and uploaded_audio is not None:
        return uploaded_audio.astype(np.float32)
    if config.input_mode == "maximize_from_silence":
        return np.zeros(n, dtype=np.float32)
    if config.input_mode == "maximize_from_noise":
        rng = np.random.default_rng(config.seed)
        return (config.seed_noise_scale * rng.standard_normal(n)).astype(np.float32)
    return np.zeros(n, dtype=np.float32)
