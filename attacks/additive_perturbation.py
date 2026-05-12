from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackOutcome
from attacks.utils import clamp_delta, normalize_map
from utils.config import AttackConfig


def run_additive_perturbation_attack(adapter, base_audio: np.ndarray, target_word: str, config: AttackConfig) -> AttackOutcome:
    x = torch.from_numpy(base_audio).float()
    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config.learning_rate)
    original_score = float(adapter.get_target_score(base_audio, target_word))

    best_score = original_score
    best_delta = np.zeros_like(base_audio)
    for _ in range(config.num_steps):
        optimizer.zero_grad()
        clipped_delta = clamp_delta(delta, config.max_delta)
        adv = torch.clamp(x + clipped_delta, -1.0, 1.0)
        score = adapter.target_score_tensor(adv, target_word)
        l2_penalty = torch.mean(clipped_delta ** 2)
        loss = -(score - config.l2_weight * l2_penalty)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta[:] = clamp_delta(delta, config.max_delta)
            adv_np = torch.clamp(x + delta, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
            score_now = float(adapter.get_target_score(adv_np, target_word))
            if score_now > best_score:
                best_score = score_now
                best_delta = (adv_np - base_audio).astype(np.float32)
            if score_now >= config.goal_score:
                break

    adv_audio = np.clip(base_audio + best_delta, -1.0, 1.0).astype(np.float32)
    return AttackOutcome(
        method="additive_perturbation",
        target_word=target_word,
        input_mode=config.input_mode,
        original_score=original_score,
        final_score=float(best_score),
        adversarial_audio=adv_audio,
        delta=best_delta,
        change_map=normalize_map(best_delta),
        success=best_score >= config.goal_score or best_score >= original_score + config.success_margin,
        metadata={},
    )
