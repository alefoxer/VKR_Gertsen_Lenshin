from __future__ import annotations

import numpy as np
import torch

from attacks.base import AttackOutcome
from attacks.utils import clamp_delta, normalize_map
from audio.segmentation import build_patch_positions
from utils.config import AttackConfig, AudioConfig


def run_patch_insertion_attack(adapter, base_audio: np.ndarray, target_word: str, config: AttackConfig, audio_config: AudioConfig) -> AttackOutcome:
    x = torch.from_numpy(base_audio).float()
    patch_len = max(1, int(audio_config.sample_rate * config.patch_duration_sec))
    positions = build_patch_positions(len(base_audio), audio_config.sample_rate, config.patch_duration_sec, config.patch_stride_sec)
    positions = positions[:12]
    original_score = float(adapter.get_target_score(base_audio, target_word))
    best_score = original_score
    best_adv = base_audio.copy()
    best_start = 0
    for start in positions:
        patch = torch.zeros(patch_len, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([patch], lr=config.learning_rate)
        for _ in range(config.num_steps):
            optimizer.zero_grad()
            clipped_patch = clamp_delta(patch, config.max_delta)
            adv = x.clone()
            end = min(len(base_audio), start + patch_len)
            usable = end - start
            adv[start:end] = torch.clamp(adv[start:end] + clipped_patch[:usable], -1.0, 1.0)
            score = adapter.target_score_tensor(adv, target_word)
            l2_penalty = torch.mean(clipped_patch[:usable] ** 2)
            loss = -(score - config.l2_weight * l2_penalty)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            clipped_patch = clamp_delta(patch, config.max_delta)
            adv = x.clone()
            end = min(len(base_audio), start + patch_len)
            usable = end - start
            adv[start:end] = torch.clamp(adv[start:end] + clipped_patch[:usable], -1.0, 1.0)
            adv_np = adv.cpu().numpy().astype(np.float32)
            score_now = float(adapter.get_target_score(adv_np, target_word))
            if score_now > best_score:
                best_score = score_now
                best_adv = adv_np
                best_start = start
            if score_now >= config.goal_score:
                break
    best_delta = (best_adv - base_audio).astype(np.float32)
    return AttackOutcome(
        method="patch_insertion",
        target_word=target_word,
        input_mode=config.input_mode,
        original_score=original_score,
        final_score=float(best_score),
        adversarial_audio=best_adv,
        delta=best_delta,
        change_map=normalize_map(best_delta),
        success=best_score >= config.goal_score or best_score >= original_score + config.success_margin,
        metadata={"best_start_sec": best_start / audio_config.sample_rate},
    )
