from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from analysis.types import SegmentAttribution


def make_run_dir(base_dir: Path) -> Path:
    run_dir = base_dir / uuid.uuid4().hex[:12]
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def segments_to_dataframe(segments: Iterable[SegmentAttribution]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments, start=1):
        rows.append(
            {
                "rank": idx,
                "start_sec": round(seg.start_sec, 4),
                "end_sec": round(seg.end_sec, 4),
                "duration_sec": round(seg.end_sec - seg.start_sec, 4),
                "saliency_mean": round(seg.saliency_mean, 6),
                "occlusion_drop": round(seg.occlusion_drop, 6),
                "attack_change_mean": round(seg.attack_change_mean, 6),
                "signal_change_mean": round(seg.signal_change_mean, 6),
                "isolated_score": round(seg.isolated_score, 6),
                "original_segment_score": round(seg.original_segment_score, 6),
                "optimized_segment_score": round(seg.optimized_segment_score, 6),
                "segment_score_gain": round(seg.segment_score_gain, 6),
                "contribution_to_gain": round(seg.contribution_to_gain, 6),
                "predicted_label": seg.predicted_label,
                "target_probability": round(seg.target_probability, 6),
                "target_probability_before": round(seg.original_segment_score, 6),
                "target_probability_after": round(seg.optimized_segment_score, 6),
                "gain": round(seg.segment_score_gain, 6),
                "is_exact_target_match": bool(seg.is_exact_target_match),
                "overlap_group_id": seg.overlap_group_id,
                "rank_type": seg.rank_type,
                "segment_role": seg.segment_role,
                "combined_score": round(seg.combined_score, 6),
                "before_audio_path": seg.original_audio_path,
                "after_audio_path": seg.optimized_audio_path,
                "audio_path": seg.audio_path,
            }
        )
    return pd.DataFrame(rows)


def save_json(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
