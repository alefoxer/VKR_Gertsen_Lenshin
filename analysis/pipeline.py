from __future__ import annotations

import numpy as np

from analysis.attribution import build_ranked_segments
from analysis.explanation import build_textual_explanation
from analysis.metrics import compute_attack_metrics
from analysis.method_explanations import get_method_explanation
from analysis.occlusion import compute_occlusion_metrics
from analysis.saliency import compute_saliency_map
from analysis.types import FullRunResult
from analysis.segment_selection import assign_overlap_groups, select_diverse_segments
from attacks.additive_perturbation import run_additive_perturbation_attack
from attacks.gradient_ascent import run_gradient_ascent_attack
from attacks.mask_attack import run_mask_attack
from attacks.patch_insertion import run_patch_insertion_attack
from attacks.template_projection import run_template_projection_attack
from attacks.utils import create_seed_audio
from audio.preprocessing import describe_preprocessing
from audio.io import save_audio
from audio.segmentation import build_time_windows, window_means
from utils.config import dataclass_to_json_dict


def run_attack(adapter, uploaded_audio, target_word, attack_config, audio_config):
    base_audio = create_seed_audio(attack_config, audio_config, uploaded_audio)
    if attack_config.method == "gradient_ascent":
        return run_gradient_ascent_attack(adapter, base_audio, target_word, attack_config), base_audio
    if attack_config.method == "additive_perturbation":
        return run_additive_perturbation_attack(adapter, base_audio, target_word, attack_config), base_audio
    if attack_config.method == "mask_attack":
        return run_mask_attack(adapter, base_audio, target_word, attack_config), base_audio
    if attack_config.method == "patch_insertion":
        return run_patch_insertion_attack(adapter, base_audio, target_word, attack_config, audio_config), base_audio
    if attack_config.method == "template_projection":
        return run_template_projection_attack(adapter, base_audio, target_word, attack_config, audio_config), base_audio
    raise ValueError(f"Неизвестный метод: {attack_config.method}")


def run_full_pipeline(adapter, uploaded_audio, target_word, audio_config, attack_config, analysis_config, run_dir):
    attack_outcome, base_audio = run_attack(adapter, uploaded_audio, target_word, attack_config, audio_config)
    adv_audio = attack_outcome.adversarial_audio
    delta = attack_outcome.delta

    windows = build_time_windows(len(adv_audio), audio_config)
    saliency_map = compute_saliency_map(adapter, adv_audio, target_word, analysis_config.saliency_smooth_kernel)
    occlusion_drops, isolated_scores = compute_occlusion_metrics(adapter, adv_audio, target_word, windows, analysis_config.occlusion_fill_value)
    original_occlusion_drops, original_isolated_scores = compute_occlusion_metrics(adapter, base_audio, target_word, windows, analysis_config.occlusion_fill_value)
    saliency_means = window_means(saliency_map, windows)
    change_means = window_means(attack_outcome.change_map, windows)
    ranked_segments = build_ranked_segments(windows, saliency_means, occlusion_drops, isolated_scores, change_means)
    ranked_segments = assign_overlap_groups(ranked_segments, analysis_config.segment_overlap_threshold)

    for seg in ranked_segments:
        window = windows[seg.window_index]
        isolated_adv = np.zeros_like(adv_audio)
        isolated_adv[window.start_sample:window.end_sample] = adv_audio[window.start_sample:window.end_sample]
        isolated_probs = adapter.predict_proba(isolated_adv)
        seg.predicted_label = max(isolated_probs, key=isolated_probs.get)
        seg.target_probability = float(isolated_probs.get(target_word, 0.0))
        seg.is_exact_target_match = seg.predicted_label == target_word and seg.target_probability >= analysis_config.exact_match_threshold
        seg.rank_type = "exact_match" if seg.is_exact_target_match else "similar_fragment"
        seg.segment_role = "exact" if seg.is_exact_target_match else "supporting"

    exact_segments = select_diverse_segments(
        [seg for seg in ranked_segments if seg.is_exact_target_match],
        analysis_config.top_n,
        analysis_config.segment_overlap_threshold,
    )
    similar_segments = select_diverse_segments(
        [seg for seg in ranked_segments if not seg.is_exact_target_match],
        analysis_config.top_n,
        analysis_config.segment_overlap_threshold,
        existing=exact_segments,
    )
    top_segments = exact_segments + similar_segments

    segments_dir = run_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    for idx, seg in enumerate(top_segments, start=1):
        start = int(seg.start_sec * audio_config.sample_rate)
        end = int(seg.end_sec * audio_config.sample_rate)
        original_segment = base_audio[start:end]
        optimized_segment = adv_audio[start:end]
        prefix = "exact" if seg.rank_type == "exact_match" else "similar"
        seg.original_audio_path = save_audio(segments_dir / f"{prefix}_segment_{idx}_before.wav", original_segment, audio_config.sample_rate)
        seg.optimized_audio_path = save_audio(segments_dir / f"{prefix}_segment_{idx}_after.wav", optimized_segment, audio_config.sample_rate)
        seg.audio_path = seg.optimized_audio_path
        window_idx = seg.window_index
        seg.original_segment_score = float(original_isolated_scores[window_idx])
        seg.optimized_segment_score = float(isolated_scores[window_idx])
        seg.segment_score_gain = float(seg.optimized_segment_score - seg.original_segment_score)
        seg.contribution_to_gain = float(occlusion_drops[window_idx] - original_occlusion_drops[window_idx])
        if seg.segment_role != "exact":
            seg.segment_role = "core" if idx <= max(1, analysis_config.top_n // 2) else "supporting"

    original_path = save_audio(run_dir / "original.wav", base_audio, audio_config.sample_rate)
    adv_path = save_audio(run_dir / "adversarial.wav", adv_audio, audio_config.sample_rate)
    maximized_path = save_audio(run_dir / "maximized.wav", adv_audio, audio_config.sample_rate)
    key_pattern_path = save_audio(run_dir / f"key_pattern_{target_word}.wav", adv_audio, audio_config.sample_rate)
    delta_path = save_audio(run_dir / "delta.wav", delta, audio_config.sample_rate)

    original_prediction = adapter.predict(base_audio)
    adversarial_prediction = adapter.predict(adv_audio)
    before = adapter.predict_proba(base_audio)
    after = adapter.predict_proba(adv_audio)
    metrics = compute_attack_metrics(base_audio, adv_audio, attack_outcome.original_score, attack_outcome.final_score, attack_outcome.success)
    is_synthesized = attack_config.input_mode in {"maximize_from_noise", "maximize_from_silence"}
    if attack_config.input_mode == "attack_uploaded_audio":
        class_image_interpretation = (
            "Это измененный вариант исходного аудио, который повышает score целевого класса. "
            "Он показывает, какие изменения входа дают рост выбранного слова."
        )
    else:
        class_image_interpretation = (
            "Это синтезированный входной сигнал с высоким score выбранного класса. "
            "Он не обязан звучать как естественная речь: его назначение - показать входной аудиообраз класса."
        )

    time_axis = np.arange(len(base_audio), dtype=np.float32) / float(audio_config.sample_rate)
    model_contract_note = None
    if adapter.model_name == "custom_torch_raw":
        model_contract_note = (
            "custom_torch_raw uses the built-in generic PyTorch adapter. It is intended for raw waveform "
            "classifiers with the contract 16 kHz waveform -> logits/probabilities over a class vocabulary. "
            "CTC, seq2seq, tokenizer-based ASR, and feature-based models require a dedicated adapter."
        )
    result = FullRunResult(
        model_name=adapter.model_name,
        target_word=target_word,
        attack_method=attack_outcome.method,
        input_mode=attack_outcome.input_mode,
        original_prediction=original_prediction,
        adversarial_prediction=adversarial_prediction,
        original_score=attack_outcome.original_score,
        final_score=attack_outcome.final_score,
        score_gain=attack_outcome.score_gain,
        success=attack_outcome.success,
        goal_reached=attack_outcome.final_score >= attack_config.goal_score,
        time_axis=time_axis,
        waveform=base_audio.astype(np.float32),
        adversarial_waveform=adv_audio.astype(np.float32),
        delta_waveform=delta.astype(np.float32),
        saliency_map=saliency_map.astype(np.float32),
        change_map=attack_outcome.change_map.astype(np.float32),
        segments=top_segments,
        exact_segments=exact_segments,
        similar_segments=similar_segments,
        probabilities_before=before,
        probabilities_after=after,
        metadata={
            **metrics,
            **attack_outcome.metadata,
            "goal_score": attack_config.goal_score,
            "run_config": {
                "audio": dataclass_to_json_dict(audio_config),
                "attack": dataclass_to_json_dict(attack_config),
                "analysis": dataclass_to_json_dict(analysis_config),
            },
            "preprocessing": describe_preprocessing(audio_config, input_mode=attack_config.input_mode),
            "model_contract_note": model_contract_note,
            "optimization_history": attack_outcome.history,
            "early_stopping_reason": attack_outcome.early_stopping_reason,
            "exact_match_threshold": analysis_config.exact_match_threshold,
            "exact_segments_count": len(exact_segments),
            "similar_segments_count": len(similar_segments),
            "class_image": {
                "target_word": target_word,
                "model_name": adapter.model_name,
                "input_mode": attack_config.input_mode,
                "attack_method": attack_config.method,
                "objective": attack_config.objective,
                "class_image_audio_path": maximized_path,
                "class_image_score": float(attack_outcome.final_score),
                "class_image_prediction": adversarial_prediction,
                "class_image_interpretation": class_image_interpretation,
                "is_synthesized_from_noise": attack_config.input_mode == "maximize_from_noise",
                "is_synthesized_from_silence": attack_config.input_mode == "maximize_from_silence",
                "is_uploaded_audio_attack": attack_config.input_mode == "attack_uploaded_audio",
                "is_synthesized": is_synthesized,
            },
            "saved_audio": {
                "original": original_path,
                "adversarial": adv_path,
                "maximized": maximized_path,
                "key_pattern": key_pattern_path,
                "delta": delta_path,
            },
        },
    )
    result.textual_explanation = build_textual_explanation(result)
    result.method_explanation = get_method_explanation(result.attack_method)
    return result
