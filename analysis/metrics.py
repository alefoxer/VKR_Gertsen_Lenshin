from __future__ import annotations

import numpy as np


def compute_attack_metrics(original_audio: np.ndarray, adv_audio: np.ndarray, original_score: float, final_score: float, success: bool):
    delta = (adv_audio - original_audio).astype(np.float32)
    power_signal = float(np.mean(original_audio ** 2) + 1e-8)
    power_noise = float(np.mean(delta ** 2) + 1e-8)
    snr_db = 10.0 * np.log10(power_signal / power_noise) if power_noise > 0 else float("inf")
    return {
        "score_gain": float(final_score - original_score),
        "delta_l2": float(np.linalg.norm(delta)),
        "delta_linf": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "snr_db": float(snr_db),
        "success": bool(success),
        "changed_fraction": float(np.mean(np.abs(delta) > 1e-5)),
    }
