from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from adapters.base import BaseRussianSpeechAdapter
from utils.russian_targets import RUSSIAN_TARGETS


class MockRussianKWSAdapterB(BaseRussianSpeechAdapter):
    model_name = "mock_ru_kws_b"

    def __init__(self, sample_rate: int = 16000, device: str = "cpu", feature_bins: int = 192, scale: float = 15.0) -> None:
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.feature_bins = feature_bins
        self.vocabulary = list(RUSSIAN_TARGETS)
        self.scale = scale
        self.templates = {}
        self.loaded = False

    @staticmethod
    def _normalize(x):
        return x / (torch.norm(x) + 1e-6)

    def load_model(self, model_path: str | Path | None = None) -> None:
        del model_path
        t = torch.linspace(0, 1, steps=self.feature_bins, device=self.device)
        hann = torch.hann_window(self.feature_bins, device=self.device)
        spike_a = torch.exp(-((t - 0.22) ** 2) / 0.004)
        spike_b = torch.exp(-((t - 0.62) ** 2) / 0.009)
        templates = {
            "да": 0.85 * torch.sin(2 * torch.pi * 2.2 * t) * hann,
            "нет": 0.85 * torch.sin(2 * torch.pi * 3.5 * t) * hann,
            "стоп": 0.6 * torch.sign(torch.sin(2 * torch.pi * 7 * t)) * hann,
            "вперёд": 0.8 * torch.linspace(-1, 1, steps=self.feature_bins, device=self.device) + 0.25 * spike_a,
            "назад": 0.8 * torch.linspace(1, -1, steps=self.feature_bins, device=self.device) + 0.25 * spike_b,
            "привет": 0.55 * (spike_a + spike_b) * torch.sin(2 * torch.pi * 5 * t),
            "включи": 0.6 * (spike_a + spike_b + torch.exp(-((t - 0.85) ** 2) / 0.004)) * torch.sin(2 * torch.pi * 2.8 * t),
            "выключи": 0.75 * torch.flip(hann, dims=[0]) * torch.sin(2 * torch.pi * 3.6 * t),
        }
        self.templates = {k: self._normalize(v.float()) for k, v in templates.items()}
        self.loaded = True

    def _prepare_tensor(self, audio):
        if hasattr(audio, 'detach'):
            x = audio.float().to(self.device)
        else:
            x = torch.from_numpy(audio).float().to(self.device)
        return x.flatten()

    def _extract_features(self, audio_tensor):
        x = audio_tensor.view(1,1,-1)
        pooled = F.adaptive_max_pool1d(torch.abs(x), self.feature_bins).view(-1)
        coarse = F.adaptive_avg_pool1d(x, self.feature_bins).view(-1)
        feat = 0.6 * coarse + 0.4 * torch.sign(coarse + 1e-6) * pooled
        return self._normalize(feat)

    def _logits(self, audio_tensor):
        if not self.loaded:
            self.load_model()
        feat = self._extract_features(audio_tensor)
        energy = torch.mean(torch.abs(audio_tensor))
        diff_energy = torch.mean(torch.abs(torch.diff(audio_tensor))) if audio_tensor.numel() > 1 else torch.tensor(0.0, device=self.device)
        vals = []
        for word in self.vocabulary:
            sim = torch.dot(feat, self.templates[word])
            bias = 0.015 * (diff_energy if word in {"вперёд","назад"} else energy)
            vals.append(self.scale * sim + bias)
        return torch.stack(vals)

    def get_vocabulary(self):
        if not self.loaded:
            self.load_model()
        return list(self.vocabulary)

    def predict(self, audio):
        p = self.predict_proba(audio)
        return max(p, key=p.get)

    def predict_proba(self, audio):
        logits = self._logits(self._prepare_tensor(audio))
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return {w: float(p) for w,p in zip(self.vocabulary, probs)}

    def get_target_score(self, audio, target_text):
        logits = self._logits(self._prepare_tensor(audio))
        idx = self.vocabulary.index(target_text)
        return float(torch.softmax(logits, dim=0)[idx].detach().cpu().item())

    def target_score_tensor(self, audio_tensor, target_text):
        logits = self._logits(self._prepare_tensor(audio_tensor))
        idx = self.vocabulary.index(target_text)
        return torch.softmax(logits, dim=0)[idx]

    def target_logit_tensor(self, audio_tensor, target_text):
        logits = self._logits(self._prepare_tensor(audio_tensor))
        idx = self.vocabulary.index(target_text)
        return logits[idx]

    def compute_gradient(self, audio, target_text):
        x = self._prepare_tensor(audio).clone().detach().requires_grad_(True)
        score = self.target_score_tensor(x, target_text)
        score.backward()
        return x.grad.detach().cpu().numpy().astype('float32')

    def supports_targeted_attack(self):
        return True

    def get_reference_pattern(self, target_text: str, num_samples: int, sample_rate: int):
        if not self.loaded:
            self.load_model()
        template = self.templates[target_text].detach().cpu().numpy().astype('float32')
        import numpy as np
        t_src = np.linspace(0,1,len(template),endpoint=False,dtype=np.float32)
        t_dst = np.linspace(0,1,num_samples,endpoint=False,dtype=np.float32)
        resampled = np.interp(t_dst,t_src,template).astype(np.float32)
        peak = float(np.max(np.abs(resampled)))
        if peak > 1e-8:
            resampled /= peak
        return resampled
