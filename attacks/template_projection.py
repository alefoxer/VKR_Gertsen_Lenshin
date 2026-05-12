from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackOutcome
from attacks.utils import normalize_map
from utils.config import AttackConfig, AudioConfig


def run_template_projection_attack(adapter, base_audio: np.ndarray, target_word: str, config: AttackConfig, audio_config: AudioConfig) -> AttackOutcome:
    original_score = float(adapter.get_target_score(base_audio, target_word))
    reference = adapter.get_reference_pattern(target_word, len(base_audio), audio_config.sample_rate)
    if reference is None:
        reference = np.zeros_like(base_audio, dtype=np.float32)
    reference = reference.astype(np.float32)
    if config.explicit_attack:
        reference = np.clip(config.prototype_emphasis * reference, -1.0, 1.0).astype(np.float32)

    best_audio = np.clip(reference, -1.0, 1.0).astype(np.float32)
    best_score = float(adapter.get_target_score(best_audio, target_word))
    best_weight = 1.0

    for mix_weight in np.linspace(config.reference_mix_min, 1.0, num=8, dtype=np.float32):
        candidate = np.clip((1.0 - mix_weight) * base_audio + mix_weight * reference, -1.0, 1.0).astype(np.float32)
        score = float(adapter.get_target_score(candidate, target_word))
        if score > best_score:
            best_audio = candidate
            best_score = score
            best_weight = float(mix_weight)
        if score >= config.goal_score:
            break

    if adapter.supports_gradients() and best_score < config.goal_score:
        x = torch.from_numpy(best_audio).float()
        x_adv = x.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_adv], lr=max(config.learning_rate * 0.5, 0.01))
        reference_tensor = torch.from_numpy(reference).float()
        for _ in range(max(12, config.num_steps // 3)):
            optimizer.zero_grad()
            score = adapter.target_score_tensor(torch.clamp(x_adv, -1.0, 1.0), target_word)
            reference_penalty = torch.mean((x_adv - reference_tensor) ** 2)
            tv_penalty = torch.mean(torch.abs(x_adv[1:] - x_adv[:-1])) if x_adv.numel() > 1 else torch.tensor(0.0)
            emphasis_bonus = torch.mean(torch.abs(x_adv - torch.from_numpy(base_audio).float())) if config.explicit_attack else torch.tensor(0.0)
            loss = -(score + 0.01 * emphasis_bonus - 0.02 * reference_penalty - config.tv_weight * tv_penalty)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                x_adv[:] = torch.clamp(x_adv, -1.0, 1.0)
                candidate = x_adv.detach().cpu().numpy().astype(np.float32)
                score_now = float(adapter.get_target_score(candidate, target_word))
                if score_now > best_score:
                    best_audio = candidate.copy()
                    best_score = score_now
                if score_now >= config.goal_score:
                    break

    adv = best_audio.astype(np.float32)
    final_score = float(best_score)
    delta = (adv - base_audio).astype(np.float32)
    return AttackOutcome(
        method="template_projection",
        target_word=target_word,
        input_mode=config.input_mode,
        original_score=original_score,
        final_score=final_score,
        adversarial_audio=adv,
        delta=delta,
        change_map=normalize_map(delta),
        success=final_score >= config.goal_score or final_score >= original_score + config.success_margin,
        metadata={
            "projection_weight": best_weight,
            "used_reference_pattern": reference is not None,
            "explicit_attack": config.explicit_attack,
            "prototype_emphasis": config.prototype_emphasis,
        },
    )
