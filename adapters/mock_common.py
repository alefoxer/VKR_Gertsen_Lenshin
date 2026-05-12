from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from adapters.base import BaseRussianSpeechAdapter
from utils.russian_targets import RUSSIAN_TARGETS


class MockRussianBaseAdapter(BaseRussianSpeechAdapter):
    def __init__(self, sample_rate: int = 16000, device: str = "cpu", feature_bins: int = 256, scale: float = 14.0) -> None:
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.feature_bins = feature_bins
        self.vocabulary = list(RUSSIAN_TARGETS)
        self.templates: Dict[str, torch.Tensor] = {}
        self.loaded = False
        self.scale = scale

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return x / (torch.norm(x) + 1e-6)

    def _prepare_tensor(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            x = torch.from_numpy(audio).float().to(self.device)
        else:
            x = audio.float().to(self.device)
        return x.flatten()

    def _extract_features(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        x = audio_tensor.view(1, 1, -1)
        pooled = F.adaptive_avg_pool1d(x, self.feature_bins).view(-1)
        deriv = torch.diff(pooled, prepend=pooled[:1])
        feat = 0.8 * pooled + 0.2 * deriv
        return self._normalize(feat)

    def _score_dict(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.loaded:
            self.load_model()
        feat = self._extract_features(audio_tensor)
        energy = torch.mean(torch.abs(audio_tensor))
        zcr = torch.mean(torch.abs(torch.diff(torch.sign(audio_tensor + 1e-6)))) if audio_tensor.numel() > 1 else torch.tensor(0.0, device=self.device)
        result: Dict[str, torch.Tensor] = {}
        for word, template in self.templates.items():
            similarity = torch.dot(feat, template)
            if word in {"стоп", "нет", "выключи"}:
                bias = 0.02 * zcr
            else:
                bias = 0.02 * energy
            result[word] = self.scale * similarity + bias
        return result

    def _logits_tensor(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        scores = self._score_dict(audio_tensor)
        return torch.stack([scores[w] for w in self.vocabulary])

    def get_vocabulary(self) -> List[str]:
        if not self.loaded:
            self.load_model()
        return list(self.vocabulary)

    def predict(self, audio: np.ndarray) -> str:
        probs = self.predict_proba(audio)
        return max(probs, key=probs.get)

    def predict_proba(self, audio: np.ndarray) -> Dict[str, float]:
        logits = self._logits_tensor(self._prepare_tensor(audio))
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return {w: float(p) for w, p in zip(self.vocabulary, probs)}

    def get_target_score(self, audio: np.ndarray, target_text: str) -> float:
        logits = self._logits_tensor(self._prepare_tensor(audio))
        idx = self.vocabulary.index(target_text)
        prob = torch.softmax(logits, dim=0)[idx]
        return float(prob.detach().cpu().item())

    def target_score_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = self._logits_tensor(self._prepare_tensor(audio_tensor))
        idx = self.vocabulary.index(target_text)
        return torch.softmax(logits, dim=0)[idx]

    def target_logit_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = self._logits_tensor(self._prepare_tensor(audio_tensor))
        idx = self.vocabulary.index(target_text)
        return logits[idx]

    def compute_gradient(self, audio: np.ndarray, target_text: str) -> np.ndarray:
        x = self._prepare_tensor(audio).clone().detach().requires_grad_(True)
        score = self.target_score_tensor(x, target_text)
        score.backward()
        return x.grad.detach().cpu().numpy().astype(np.float32)

    def supports_targeted_attack(self) -> bool:
        return True

    def get_reference_pattern(self, target_text: str, num_samples: int, sample_rate: int) -> np.ndarray | None:
        if not self.loaded:
            self.load_model()
        template = self.templates[target_text].detach().cpu().numpy().astype(np.float32)
        t_src = np.linspace(0, 1, len(template), endpoint=False, dtype=np.float32)
        t_dst = np.linspace(0, 1, num_samples, endpoint=False, dtype=np.float32)
        resampled = np.interp(t_dst, t_src, template).astype(np.float32)
        peak = float(np.max(np.abs(resampled)))
        if peak > 1e-8:
            resampled /= peak
        return resampled
