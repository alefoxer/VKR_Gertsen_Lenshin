from __future__ import annotations

from pathlib import Path

import torch

from adapters.mock_common import MockRussianBaseAdapter


class MockRussianKWSAdapterA(MockRussianBaseAdapter):
    model_name = "mock_ru_kws_a"

    def __init__(self, sample_rate: int = 16000, device: str = "cpu") -> None:
        super().__init__(sample_rate=sample_rate, device=device, feature_bins=256, scale=16.0)

    def load_model(self, model_path: str | Path | None = None) -> None:
        del model_path
        t = torch.linspace(0, 1, steps=self.feature_bins, device=self.device)
        hann = torch.hann_window(self.feature_bins, device=self.device)
        pulse = torch.exp(-((t - 0.25) ** 2) / 0.01) + torch.exp(-((t - 0.7) ** 2) / 0.012)
        triple = (
            torch.exp(-((t - 0.18) ** 2) / 0.006)
            + torch.exp(-((t - 0.5) ** 2) / 0.01)
            + torch.exp(-((t - 0.82) ** 2) / 0.006)
        )
        templates = {
            "да": 0.95 * torch.sin(2 * torch.pi * 2 * t) * hann,
            "нет": 0.95 * torch.sin(2 * torch.pi * 3 * t) * hann,
            "стоп": 0.55 * torch.sign(torch.sin(2 * torch.pi * 9 * t)) * hann,
            "вперёд": torch.linspace(-1, 1, steps=self.feature_bins, device=self.device) + 0.3 * torch.sin(2 * torch.pi * 2 * t),
            "назад": torch.linspace(1, -1, steps=self.feature_bins, device=self.device) + 0.3 * torch.sin(2 * torch.pi * 2 * t),
            "привет": pulse * (0.8 * torch.sin(2 * torch.pi * 4 * t)),
            "включи": triple * (0.9 * torch.sin(2 * torch.pi * 2.5 * t)),
            "выключи": triple.flip(0) * (0.85 * torch.sin(2 * torch.pi * 3.5 * t)),
        }
        self.templates = {k: self._normalize(v.float()) for k, v in templates.items()}
        self.loaded = True
