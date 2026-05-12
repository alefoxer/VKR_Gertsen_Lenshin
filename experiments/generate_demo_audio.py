from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import numpy as np

from audio.io import save_audio
from utils.config import DEFAULT_AUDIO_CONFIG
from utils.russian_targets import RUSSIAN_TARGETS


def generate_signal(target: str, variant: int = 0) -> np.ndarray:
    sr = DEFAULT_AUDIO_CONFIG.sample_rate
    body_len = sr
    t = np.linspace(0, 1, body_len, endpoint=False, dtype=np.float32)
    hann = np.hanning(body_len).astype(np.float32)
    pulse = np.exp(-((t - 0.25) ** 2) / 0.01).astype(np.float32) + np.exp(-((t - 0.7) ** 2) / 0.012).astype(np.float32)
    triple = (
        np.exp(-((t - 0.18) ** 2) / 0.006).astype(np.float32)
        + np.exp(-((t - 0.5) ** 2) / 0.01).astype(np.float32)
        + np.exp(-((t - 0.82) ** 2) / 0.006).astype(np.float32)
    )
    base = {
        "да": 0.9 * np.sin(2 * np.pi * (2.0 + 0.1 * variant) * t).astype(np.float32) * hann,
        "нет": 0.9 * np.sin(2 * np.pi * (3.0 + 0.1 * variant) * t).astype(np.float32) * hann,
        "стоп": 0.45 * np.sign(np.sin(2 * np.pi * (8.0 + 0.2 * variant) * t)).astype(np.float32) * hann,
        "вперёд": (np.linspace(-0.9, 0.9, body_len, dtype=np.float32) + 0.2 * np.sin(2 * np.pi * 2 * t)).astype(np.float32),
        "назад": (np.linspace(0.9, -0.9, body_len, dtype=np.float32) + 0.2 * np.sin(2 * np.pi * 2 * t)).astype(np.float32),
        "привет": 0.75 * pulse * np.sin(2 * np.pi * 4 * t).astype(np.float32),
        "включи": 0.8 * triple * np.sin(2 * np.pi * 2.5 * t).astype(np.float32),
        "выключи": 0.8 * triple[::-1].copy() * np.sin(2 * np.pi * 3.5 * t).astype(np.float32),
    }[target]
    noise = (0.015 + 0.005 * variant) * np.random.randn(body_len).astype(np.float32)
    silence = np.zeros(sr // 2, dtype=np.float32)
    waveform = np.concatenate([silence, base + noise, silence]).astype(np.float32)
    peak = float(np.max(np.abs(waveform)))
    if peak > 1e-8:
        waveform /= peak
    return waveform


def generate_dataset(output_dir: Path, variants_per_word: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in RUSSIAN_TARGETS:
        for variant in range(variants_per_word):
            waveform = generate_signal(target, variant)
            save_audio(output_dir / f"{target}_{variant}.wav", waveform, DEFAULT_AUDIO_CONFIG.sample_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация demo-аудио")
    parser.add_argument("--target", type=str, default="да", choices=RUSSIAN_TARGETS)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_target.wav"))
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs/demo_dataset"))
    args = parser.parse_args()

    if args.dataset:
        generate_dataset(args.dataset_dir)
        print(f"Dataset saved to: {args.dataset_dir}")
    else:
        waveform = generate_signal(args.target)
        save_audio(args.output, waveform, DEFAULT_AUDIO_CONFIG.sample_rate)
        print(f"Saved: {args.output}")
