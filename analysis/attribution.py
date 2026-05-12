from __future__ import annotations

import numpy as np

from analysis.types import SegmentAttribution


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - arr.min()
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 1e-8:
        arr /= max_val
    return arr


def build_ranked_segments(windows, saliency_means, occlusion_drops, isolated_scores, attack_change_means):
    ns = _normalize(saliency_means)
    no = _normalize(occlusion_drops)
    ni = _normalize(isolated_scores)
    nc = _normalize(attack_change_means)
    segments = []
    for idx, window in enumerate(windows):
        combined = 0.35 * ns[idx] + 0.25 * no[idx] + 0.15 * ni[idx] + 0.25 * nc[idx]
        segments.append(
            SegmentAttribution(
                window_index=idx,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                saliency_mean=float(saliency_means[idx]),
                occlusion_drop=float(occlusion_drops[idx]),
                attack_change_mean=float(attack_change_means[idx]),
                isolated_score=float(isolated_scores[idx]),
                combined_score=float(combined),
            )
        )
    segments.sort(key=lambda x: x.combined_score, reverse=True)
    return segments
