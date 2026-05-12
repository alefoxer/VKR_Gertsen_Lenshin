from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import librosa
import numpy as np
import soundfile as sf

from utils.config import AudioConfig


class AudioProcessingError(RuntimeError):
    pass


def prepare_audio_waveform(waveform: np.ndarray, config: AudioConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    waveform = waveform.astype(np.float32)
    original_samples = int(waveform.size)
    peak_before = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if waveform.size == 0:
        raise AudioProcessingError("Пустое аудио.")
    if config.normalize and peak_before > 1e-8:
        waveform = waveform / peak_before
    min_samples = int(config.min_duration_sec * config.sample_rate)
    padded_samples = max(0, min_samples - len(waveform))
    if padded_samples:
        waveform = np.pad(waveform, (0, padded_samples))
    metadata = {
        "original_samples": original_samples,
        "final_samples": int(waveform.size),
        "duration_sec": float(waveform.size / config.sample_rate),
        "peak_before_normalize": peak_before,
        "peak_after_normalize": float(np.max(np.abs(waveform))) if waveform.size else 0.0,
        "padded_samples": int(padded_samples),
        "trimmed_samples": 0,
    }
    return waveform.astype(np.float32), metadata


def load_audio(file_path: str | Path, config: AudioConfig) -> Tuple[np.ndarray, int]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise AudioProcessingError(f"Файл не найден: {file_path}")
    waveform, sr = librosa.load(str(file_path), sr=config.sample_rate, mono=config.mono)
    waveform, _ = prepare_audio_waveform(waveform, config)
    return waveform, sr


def save_audio(file_path: str | Path, waveform: np.ndarray, sample_rate: int) -> str:
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), waveform.astype(np.float32), sample_rate)
    return str(file_path)
