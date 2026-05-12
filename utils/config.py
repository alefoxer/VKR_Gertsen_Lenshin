from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class AudioConfig:
    sample_rate: int = 16_000
    mono: bool = True
    normalize: bool = True
    window_sec: float = 0.25
    hop_sec: float = 0.05
    min_duration_sec: float = 1.0
    default_duration_sec: float = 2.0


@dataclass
class AttackConfig:
    method: str = "gradient_ascent"
    input_mode: str = "attack_uploaded_audio"
    num_steps: int = 120
    learning_rate: float = 0.08
    max_delta: float = 0.25
    l2_weight: float = 0.005
    tv_weight: float = 0.001
    mask_sparsity_weight: float = 0.02
    mask_smoothness_weight: float = 0.01
    patch_duration_sec: float = 0.35
    patch_stride_sec: float = 0.05
    success_margin: float = 0.05
    goal_score: float = 0.99
    seed_noise_scale: float = 0.02
    explicit_attack: bool = True
    prototype_emphasis: float = 1.35
    reference_mix_min: float = 0.75
    objective: str = "probability"
    seed: int | None = 42


@dataclass
class AnalysisConfig:
    top_n: int = 5
    saliency_smooth_kernel: int = 11
    occlusion_fill_value: float = 0.0
    exact_match_threshold: float = 0.95
    segment_overlap_threshold: float = 0.55
    merge_gap_sec: float = 0.04


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    outputs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs")
    runs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "outputs" / "runs")
    supported_extensions: List[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg"])


def dataclass_to_json_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return dataclass_to_json_dict(asdict(value))
    if isinstance(value, dict):
        return {str(k): dataclass_to_json_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_json_dict(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


DEFAULT_AUDIO_CONFIG = AudioConfig()
DEFAULT_ATTACK_CONFIG = AttackConfig()
DEFAULT_ANALYSIS_CONFIG = AnalysisConfig()
DEFAULT_APP_CONFIG = AppConfig()
