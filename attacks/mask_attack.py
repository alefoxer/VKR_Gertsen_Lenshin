from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackOutcome
from attacks.utils import clamp_delta, normalize_map
from utils.config import AttackConfig


def run_mask_attack(adapter, base_audio: np.ndarray, target_word: str, config: AttackConfig) -> AttackOutcome:
    x = torch.from_numpy(base_audio).float()
    delta = torch.zeros_like(x, requires_grad=True)
    mask_logits = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.Adam([delta, mask_logits], lr=config.learning_rate)
    original_score = float(adapter.get_target_score(base_audio, target_word))
    best_score = original_score
    best_adv = base_audio.copy()
    best_mask = None
    for _ in range(config.num_steps):
        optimizer.zero_grad()
        mask = torch.sigmoid(mask_logits)
        clipped_delta = clamp_delta(delta, config.max_delta)
        adv = torch.clamp(x + mask * clipped_delta, -1.0, 1.0)
        score = adapter.target_score_tensor(adv, target_word)
        l2_penalty = torch.mean((mask * clipped_delta) ** 2)
        sparsity = mask.mean()
        smoothness = torch.mean(torch.abs(mask[1:] - mask[:-1])) if mask.numel() > 1 else torch.tensor(0.0)
        loss = -(score - config.l2_weight * l2_penalty - config.mask_sparsity_weight * sparsity - config.mask_smoothness_weight * smoothness)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta[:] = clamp_delta(delta, config.max_delta)
            adv_np = torch.clamp(x + torch.sigmoid(mask_logits) * delta, -1.0, 1.0).detach().cpu().numpy().astype(np.float32)
            score_now = float(adapter.get_target_score(adv_np, target_word))
            if score_now > best_score:
                best_score = score_now
                best_adv = adv_np.copy()
                best_mask = torch.sigmoid(mask_logits).detach().cpu().numpy().astype(np.float32)
            if score_now >= config.goal_score:
                break
    best_delta = (best_adv - base_audio).astype(np.float32)
    return AttackOutcome(
        method="mask_attack",
        target_word=target_word,
        input_mode=config.input_mode,
        original_score=original_score,
        final_score=float(best_score),
        adversarial_audio=best_adv,
        delta=best_delta,
        change_map=normalize_map(best_delta if best_mask is None else np.abs(best_delta) + 0.5 * best_mask),
        success=best_score >= config.goal_score or best_score >= original_score + config.success_margin,
        metadata={"mask_mean": float(best_mask.mean()) if best_mask is not None else None},
    )
