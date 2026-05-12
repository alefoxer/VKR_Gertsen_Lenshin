from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class SegmentAttribution:
    window_index: int
    start_sec: float
    end_sec: float
    saliency_mean: float
    occlusion_drop: float
    attack_change_mean: float
    isolated_score: float
    combined_score: float
    audio_path: str | None = None
    original_audio_path: str | None = None
    optimized_audio_path: str | None = None
    original_segment_score: float = 0.0
    optimized_segment_score: float = 0.0
    segment_score_gain: float = 0.0
    contribution_to_gain: float = 0.0
    predicted_label: str = ""
    target_probability: float = 0.0
    is_exact_target_match: bool = False
    overlap_group_id: int = -1
    rank_type: str = "similar_fragment"
    segment_role: str = "supporting"

    @property
    def signal_change_mean(self) -> float:
        return self.attack_change_mean


@dataclass
class FullRunResult:
    model_name: str
    target_word: str
    attack_method: str
    input_mode: str
    original_prediction: str
    adversarial_prediction: str
    original_score: float
    final_score: float
    score_gain: float
    success: bool
    goal_reached: bool
    time_axis: np.ndarray
    waveform: np.ndarray
    adversarial_waveform: np.ndarray
    delta_waveform: np.ndarray
    saliency_map: np.ndarray
    change_map: np.ndarray
    segments: List[SegmentAttribution] = field(default_factory=list)
    exact_segments: List[SegmentAttribution] = field(default_factory=list)
    similar_segments: List[SegmentAttribution] = field(default_factory=list)
    probabilities_before: Dict[str, float] = field(default_factory=dict)
    probabilities_after: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    textual_explanation: str = ""
    method_explanation: str = ""

    @property
    def optimization_method(self) -> str:
        return self.attack_method

    @property
    def baseline_waveform(self) -> np.ndarray:
        return self.waveform

    @property
    def maximized_waveform(self) -> np.ndarray:
        return self.adversarial_waveform

    def summary_dict(self) -> Dict[str, Any]:
        class_image = self.metadata.get("class_image", {})
        return {
            "model_name": self.model_name,
            "target_word": self.target_word,
            "attack_method": self.attack_method,
            "optimization_method": self.attack_method,
            "input_mode": self.input_mode,
            "original_prediction": self.original_prediction,
            "adversarial_prediction": self.adversarial_prediction,
            "original_score": self.original_score,
            "baseline_score": self.original_score,
            "final_score": self.final_score,
            "maximized_score": self.final_score,
            "score_gain": self.score_gain,
            "success": self.success,
            "goal_reached": self.goal_reached,
            "class_image": class_image,
            "class_image_audio_path": class_image.get("class_image_audio_path"),
            "class_image_score": class_image.get("class_image_score"),
            "class_image_prediction": class_image.get("class_image_prediction"),
            "class_image_interpretation": class_image.get("class_image_interpretation"),
            "is_synthesized_from_noise": class_image.get("is_synthesized_from_noise", False),
            "is_uploaded_audio_attack": class_image.get("is_uploaded_audio_attack", False),
            "segments": [
                {
                    "start_sec": seg.start_sec,
                    "end_sec": seg.end_sec,
                    "predicted_label": seg.predicted_label,
                    "target_probability": seg.target_probability,
                    "is_exact_target_match": seg.is_exact_target_match,
                    "rank_type": seg.rank_type,
                    "overlap_group_id": seg.overlap_group_id,
                    "combined_score": seg.combined_score,
                    "segment_role": seg.segment_role,
                    "target_probability_before": seg.original_segment_score,
                    "target_probability_after": seg.optimized_segment_score,
                    "gain": seg.segment_score_gain,
                }
                for seg in self.segments
            ],
            "probabilities_before": self.probabilities_before,
            "probabilities_after": self.probabilities_after,
            "textual_explanation": self.textual_explanation,
            "method_explanation": self.method_explanation,
            "metadata": self.metadata,
        }
