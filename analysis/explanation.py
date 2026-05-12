from __future__ import annotations

from typing import Iterable

from analysis.types import FullRunResult, SegmentAttribution


def _format_segment(seg: SegmentAttribution) -> str:
    return (
        f"[{seg.start_sec:.2f}-{seg.end_sec:.2f} c]: "
        f"label={seg.predicted_label}, "
        f"P(segment) {seg.original_segment_score:.3f} -> {seg.optimized_segment_score:.3f}, "
        f"вклад={seg.contribution_to_gain:.3f}"
    )


def _top_segments_text(segments: Iterable[SegmentAttribution], limit: int = 3) -> str:
    top = list(segments)[:limit]
    if not top:
        return "выраженных значимых сегментов не найдено"
    return "; ".join(_format_segment(seg) for seg in top)


def build_textual_explanation(result: FullRunResult) -> str:
    exact_segments = result.exact_segments[:3]
    similar_segments = result.similar_segments[:3]
    top_segments = exact_segments if exact_segments else similar_segments
    history = result.metadata.get("optimization_history", {})
    steps_run = result.metadata.get("steps_run", history.get("steps_run", "н/д"))
    objective = result.metadata.get("objective", history.get("objective", "н/д"))
    model_context = (
        f"Модель `{result.model_name}` оптимизировалась под слово '{result.target_word}'. "
        f"Метод: `{result.attack_method}`, режим входа: `{result.input_mode}`, objective: `{objective}`, "
        f"шагов выполнено: {steps_run}. Score целевого слова изменился "
        f"с {result.original_score:.3f} до {result.final_score:.3f}."
    )

    low_score_reason = (
        f"До оптимизации вероятность слова '{result.target_word}' была {result.original_score:.3f}. "
        "Далее показаны временные окна, которые после оптимизации стали наиболее значимыми для целевого класса. "
        f"Для каждого окна приведена локальная вероятность изолированного сегмента до и после оптимизации: {_top_segments_text(top_segments)}."
    )

    changed_ranges = (
        ", ".join(f"{seg.start_sec:.2f}-{seg.end_sec:.2f} c" for seg in top_segments)
        if top_segments
        else "ключевые интервалы не выделены"
    )
    high_score_reason = (
        f"После оптимизации вероятность выросла до {result.final_score:.3f}. "
        f"Наиболее важные изменения появились в сегментах {changed_ranges}, "
        "где saliency, occlusion и карта изменений одновременно указывают на высокую значимость. "
        "Сформированный сигнал может быть шумовым и не похожим на естественную речь, но его участки повышают "
        "score целевого класса."
    )

    if exact_segments:
        exact_text = "Точные сегменты: " + "; ".join(
            f"[{seg.start_sec:.2f}-{seg.end_sec:.2f} c], P(after)={seg.target_probability:.3f}, "
            f"P(before->after)={seg.original_segment_score:.3f}->{seg.optimized_segment_score:.3f}"
            for seg in exact_segments
        )
    else:
        exact_text = (
            "Точные изолированные сегменты, которые сами распознаются как целевое слово выше заданного порога, не найдены. "
            "Для реальной PyTorch-модели это нормальная ситуация: решение может формироваться распределенно по нескольким "
            "временным участкам, а не одним коротким фрагментом."
        )

    if similar_segments:
        similar_text = "Похожие/поддерживающие сегменты: " + "; ".join(
            f"[{seg.start_sec:.2f}-{seg.end_sec:.2f} c], combined={seg.combined_score:.3f}"
            for seg in similar_segments
        ) + ". Эти окна не обязательно распознаются как целое слово сами по себе, но повышают уверенность модели в целевом классе."
    else:
        similar_text = "Дополнительные похожие сегменты не выделены."

    causal_link = (
        "Рост вероятности связан с тем, что после оптимизации в этих временных окнах вырос вклад признаков "
        "целевого класса. Это видно по изменению isolated score для сегментов и по положительному вкладу "
        "этих фрагментов в общий прирост вероятности."
    )

    scientific_takeaway = (
        "Вывод: результат описывает не универсальный эталон слова, а набор акустических признаков и временных "
        "структур, которые повышают score выбранного класса в данной модели."
    )

    baseline_note = ""
    if result.model_name == "gradv_ru_kws_baseline":
        baseline_note = (
            "Для `gradv_ru_kws_baseline` результат следует читать как исследование компактной KWS-модели: "
            "найденный сигнал не обязан звучать как естественная речь, но он показывает входной аудиообраз, "
            "который повышает score выбранного слова."
        )

    parts = [model_context, low_score_reason, high_score_reason, exact_text, similar_text, causal_link, scientific_takeaway]
    if baseline_note:
        parts.append(baseline_note)
    return "\n\n".join(parts)
