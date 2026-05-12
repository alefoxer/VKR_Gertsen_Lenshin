from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


class BaseRussianSpeechAdapter(ABC):
    model_name: str = "base_ru_model"

    @abstractmethod
    def load_model(self, model_path: str | Path | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, audio: np.ndarray) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, audio: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_vocabulary(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_target_score(self, audio: np.ndarray, target_text: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_gradient(self, audio: np.ndarray, target_text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def target_score_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        raise NotImplementedError

    def target_logit_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        score = torch.clamp(self.target_score_tensor(audio_tensor, target_text), min=1e-8)
        return torch.log(score)

    @abstractmethod
    def supports_targeted_attack(self) -> bool:
        raise NotImplementedError

    def supports_gradients(self) -> bool:
        return self.supports_targeted_attack()

    def language(self) -> str:
        return "ru"

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "language": self.language(),
            "supports_targeted_attack": str(self.supports_targeted_attack()),
            "supports_gradients": str(self.supports_gradients()),
        }

    def get_reference_pattern(self, target_text: str, num_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        return None
