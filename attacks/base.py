from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class AttackOutcome:
    method: str
    target_word: str
    input_mode: str
    original_score: float
    final_score: float
    adversarial_audio: np.ndarray
    delta: np.ndarray
    change_map: np.ndarray
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    early_stopping_reason: str = "completed"

    @property
    def score_gain(self) -> float:
        return float(self.final_score - self.original_score)

    @property
    def optimized_audio(self) -> np.ndarray:
        return self.adversarial_audio
