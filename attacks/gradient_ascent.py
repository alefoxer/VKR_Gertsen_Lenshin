from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from attacks.base import AttackOutcome
from attacks.utils import clamp_delta, normalize_map
from utils.config import AttackConfig


def _target_objective(adapter, audio_tensor: torch.Tensor, target_word: str, objective: str) -> torch.Tensor:
    objective = (objective or "probability").lower()
    if objective == "logit":
        return adapter.target_logit_tensor(audio_tensor, target_word)
    if objective in {"cross_entropy", "target_cross_entropy", "ce"}:
        if hasattr(adapter, "vocabulary") and hasattr(adapter, "_logits_tensor"):
            logits = adapter._logits_tensor(adapter._prepare_tensor(audio_tensor))  # type: ignore[attr-defined]
            target_idx = adapter.get_vocabulary().index(target_word)
            return -F.cross_entropy(logits.view(1, -1), torch.tensor([target_idx], device=logits.device))
        if hasattr(adapter, "vocabulary") and hasattr(adapter, "_forward_logits"):
            logits = adapter._forward_logits(audio_tensor)  # type: ignore[attr-defined]
            target_idx = adapter.get_vocabulary().index(target_word)
            return -F.cross_entropy(logits.view(1, -1), torch.tensor([target_idx], device=logits.device))
        return torch.log(torch.clamp(adapter.target_score_tensor(audio_tensor, target_word), min=1e-8))
    return adapter.target_score_tensor(audio_tensor, target_word)


def run_gradient_ascent_attack(adapter, base_audio: np.ndarray, target_word: str, config: AttackConfig) -> AttackOutcome:
    if config.seed is not None:
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
    x = torch.from_numpy(base_audio).float()
    x_adv = x.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=config.learning_rate)
    original_score = float(adapter.get_target_score(base_audio, target_word))

    best_score = original_score
    best_adv = base_audio.copy()
    score_history = [original_score]
    loss_history = []
    best_history = [best_score]
    early_reason = "completed"
    steps_run = 0
    for step in range(config.num_steps):
        optimizer.zero_grad()
        delta = x_adv - x
        score = _target_objective(adapter, torch.clamp(x_adv, -1.0, 1.0), target_word, config.objective)
        l2_penalty = torch.mean(delta ** 2)
        tv_penalty = torch.mean(torch.abs(delta[1:] - delta[:-1])) if delta.numel() > 1 else torch.tensor(0.0)
        loss = -(score - config.l2_weight * l2_penalty - config.tv_weight * tv_penalty)
        loss.backward()
        optimizer.step()
        steps_run = step + 1
        with torch.no_grad():
            delta = clamp_delta(x_adv - x, config.max_delta)
            x_adv[:] = torch.clamp(x + delta, -1.0, 1.0)
            score_now = float(adapter.get_target_score(x_adv.detach().cpu().numpy().astype(np.float32), target_word))
            score_history.append(score_now)
            loss_history.append(float(loss.detach().cpu().item()))
            if score_now > best_score:
                best_score = score_now
                best_adv = x_adv.detach().cpu().numpy().astype(np.float32).copy()
            best_history.append(float(best_score))
            if score_now >= config.goal_score:
                early_reason = "goal_score_reached"
                break

    delta_np = (best_adv - base_audio).astype(np.float32)
    history = {
        "objective": config.objective,
        "score_per_step": [float(v) for v in score_history],
        "loss_per_step": [float(v) for v in loss_history],
        "best_score_per_step": [float(v) for v in best_history],
        "steps_run": steps_run,
        "early_stopping_reason": early_reason,
    }
    return AttackOutcome(
        method="gradient_ascent",
        target_word=target_word,
        input_mode=config.input_mode,
        original_score=original_score,
        final_score=float(best_score),
        adversarial_audio=best_adv,
        delta=delta_np,
        change_map=normalize_map(delta_np),
        success=best_score >= config.goal_score or best_score >= original_score + config.success_margin,
        metadata={"objective": config.objective, "steps_run": steps_run, "early_stopping_reason": early_reason},
        history=history,
        early_stopping_reason=early_reason,
    )
