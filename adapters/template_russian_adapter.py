from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from adapters.base import BaseRussianSpeechAdapter


class TemplateRussianAdapter(BaseRussianSpeechAdapter):
    model_name = "template_real_ru_model"

    def __init__(self, vocabulary: List[str], device: str = "cpu") -> None:
        self.vocabulary = vocabulary
        self.device = torch.device(device)
        self.model: torch.nn.Module | None = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        raise NotImplementedError

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(audio).float().to(self.device).unsqueeze(0)

    def _forward_logits(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, audio: np.ndarray) -> str:
        probs = self.predict_proba(audio)
        return max(probs, key=probs.get)

    def predict_proba(self, audio: np.ndarray) -> Dict[str, float]:
        logits = self._forward_logits(self._preprocess_audio(audio)).squeeze(0)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return {w: float(p) for w, p in zip(self.vocabulary, probs)}

    def get_vocabulary(self) -> List[str]:
        return self.vocabulary

    def get_target_score(self, audio: np.ndarray, target_text: str) -> float:
        logits = self._forward_logits(self._preprocess_audio(audio)).squeeze(0)
        idx = self.vocabulary.index(target_text)
        return float(torch.softmax(logits, dim=0)[idx].detach().cpu().item())

    def target_score_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = self._forward_logits(audio_tensor.unsqueeze(0)).squeeze(0)
        idx = self.vocabulary.index(target_text)
        return torch.softmax(logits, dim=0)[idx]

    def compute_gradient(self, audio: np.ndarray, target_text: str) -> np.ndarray:
        x = torch.from_numpy(audio).float().to(self.device).requires_grad_(True)
        score = self.target_score_tensor(x, target_text)
        score.backward()
        return x.grad.detach().cpu().numpy().astype(np.float32)

    def supports_targeted_attack(self) -> bool:
        return True
