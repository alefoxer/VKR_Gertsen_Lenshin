from __future__ import annotations

import numpy as np


def compute_occlusion_metrics(adapter, audio: np.ndarray, target_word: str, windows, fill_value: float = 0.0):
    original_score = float(adapter.get_target_score(audio, target_word))
    score_drops = []
    isolated_scores = []
    for window in windows:
        masked = audio.copy()
        masked[window.start_sample:window.end_sample] = fill_value
        masked_score = float(adapter.get_target_score(masked, target_word))
        score_drops.append(original_score - masked_score)

        isolated = np.zeros_like(audio)
        isolated[window.start_sample:window.end_sample] = audio[window.start_sample:window.end_sample]
        isolated_scores.append(float(adapter.get_target_score(isolated, target_word)))
    return np.asarray(score_drops, dtype=np.float32), np.asarray(isolated_scores, dtype=np.float32)
