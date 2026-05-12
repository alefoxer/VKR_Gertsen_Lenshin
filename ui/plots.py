from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from analysis.types import FullRunResult


def create_study_comparison_figure(runs_df):
    fig = plt.figure(figsize=(15, 8), facecolor="#fcfbf7")
    valid = runs_df[runs_df["error"].fillna("") == ""] if "error" in runs_df.columns else runs_df

    ax1 = fig.add_subplot(2, 1, 1)
    if not valid.empty:
        for input_mode, group in valid.groupby("input_mode", dropna=False):
            group = group.sort_values("seed")
            ax1.plot(
                group["seed"].astype(int),
                group["final_score"].astype(float),
                marker="o",
                linewidth=1.6,
                label=str(input_mode),
            )
        ax1.set_ylim(0.0, 1.05)
    else:
        ax1.text(0.5, 0.5, "No completed runs", ha="center", va="center")
    ax1.set_title("Final score по seed и начальному условию")
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Final target score")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(loc="lower right")

    ax2 = fig.add_subplot(2, 1, 2)
    if not valid.empty:
        grouped = (
            valid.groupby("input_mode", dropna=False)
            .agg(
                mean_gain=("score_gain", "mean"),
                min_gain=("score_gain", "min"),
                max_gain=("score_gain", "max"),
            )
            .sort_values("mean_gain", ascending=False)
        )
        x = np.arange(len(grouped))
        mean_gain = grouped["mean_gain"].astype(float).to_numpy()
        yerr = np.vstack(
            [
                mean_gain - grouped["min_gain"].astype(float).to_numpy(),
                grouped["max_gain"].astype(float).to_numpy() - mean_gain,
            ]
        )
        ax2.bar(x, mean_gain, yerr=yerr, color="#00897b", ecolor="#263238", capsize=5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(grouped.index.tolist(), rotation=20, ha="right")
    else:
        ax2.text(0.5, 0.5, "No completed runs", ha="center", va="center")
    ax2.set_title("Средний score gain по условиям с min/max диапазоном")
    ax2.set_ylabel("Mean score gain")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def create_model_comparison_figure(summary_df):
    fig = plt.figure(figsize=(13, 7), facecolor="#fcfbf7")
    valid = summary_df[summary_df["error"].fillna("") == ""] if "error" in summary_df.columns else summary_df

    ax1 = fig.add_subplot(2, 1, 1)
    if not valid.empty:
        labels = valid["model_name"].astype(str).tolist()
        x = np.arange(len(labels))
        ax1.bar(x - 0.18, valid["mean_final_score"].astype(float), width=0.36, label="mean final", color="#42a5f5")
        ax1.bar(x + 0.18, valid["best_final_score"].astype(float), width=0.36, label="best final", color="#7e57c2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=15, ha="right")
        ax1.set_ylim(0.0, 1.05)
        ax1.legend(loc="lower right")
    else:
        ax1.text(0.5, 0.5, "No completed models", ha="center", va="center")
    ax1.set_title("Best/mean final score по моделям")
    ax1.set_ylabel("Target score")
    ax1.grid(axis="y", alpha=0.2)

    ax2 = fig.add_subplot(2, 1, 2)
    if not valid.empty:
        labels = valid["model_name"].astype(str).tolist()
        ax2.bar(np.arange(len(labels)), valid["success_rate"].astype(float), color="#00897b")
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels, rotation=15, ha="right")
        ax2.set_ylim(0.0, 1.05)
    else:
        ax2.text(0.5, 0.5, "No completed models", ha="center", va="center")
    ax2.set_title("Success rate по моделям")
    ax2.set_ylabel("Success rate")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def create_class_image_figure(result: FullRunResult):
    sample_rate = int(result.metadata.get("run_config", {}).get("audio", {}).get("sample_rate", 16000))
    waveform = result.adversarial_waveform.astype(np.float32)
    fig = plt.figure(figsize=(13, 9), facecolor="#fcfbf7")

    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(result.time_axis, waveform, color="#00695c", linewidth=1.1)
    ax1.set_title(f"Входной образ класса «{result.target_word}»: waveform", fontsize=14)
    ax1.set_ylabel("Амплитуда")
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(3, 1, 2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)
        ax2.specgram(waveform + 1e-8, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    ax2.set_title("Spectrogram найденного образа класса", fontsize=14)
    ax2.set_ylabel("Частота, Гц")
    ax2.set_xlabel("Время, сек")

    ax3 = fig.add_subplot(3, 1, 3)
    labels = list(result.probabilities_after.keys())
    values = [result.probabilities_after[label] for label in labels]
    colors = ["#90a4ae" if label != result.target_word else "#2e7d32" for label in labels]
    ax3.bar(np.arange(len(labels)), values, color=colors)
    ax3.set_xticks(np.arange(len(labels)))
    ax3.set_xticklabels(labels, rotation=20)
    ax3.set_ylim(0.0, 1.05)
    ax3.set_ylabel("P(class)")
    ax3.set_title(
        f"Вероятности классов для найденного образа: P({result.target_word})={result.final_score:.4f}",
        fontsize=14,
    )
    ax3.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    return fig


def create_attack_figure(result: FullRunResult):
    sample_rate = int(result.metadata.get("run_config", {}).get("audio", {}).get("sample_rate", 16000))
    history = result.metadata.get("optimization_history", {})
    fig = plt.figure(figsize=(14, 16), facecolor="#fcfbf7")

    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(result.time_axis, result.waveform, color="#546e7a", linewidth=1.1, label="Исходный")
    ax1.plot(result.time_axis, result.adversarial_waveform, color="#00796b", linewidth=1.1, alpha=0.9, label="Образ класса")
    ax1.set_title("Waveform: исходный сигнал и найденный входной образ", fontsize=14)
    ax1.set_ylabel("Амплитуда")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(result.time_axis, result.delta_waveform, color="#ef6c00", linewidth=1.0)
    ax2.fill_between(result.time_axis, 0.0, result.delta_waveform, color="#ffcc80", alpha=0.45)
    ax2.set_title("Изменение сигнала: сформированный входной образ класса", fontsize=14)
    ax2.set_ylabel("Delta")
    ax2.grid(alpha=0.2)

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(result.time_axis, result.saliency_map, color="#3949ab", linewidth=1.3, label="Saliency")
    ax3.plot(result.time_axis, result.change_map, color="#d81b60", linewidth=1.2, label="Карта изменений")
    for idx, seg in enumerate(result.exact_segments, start=1):
        ax3.axvspan(seg.start_sec, seg.end_sec, color="#66bb6a", alpha=0.24)
        ax3.text(seg.start_sec, 1.01, f"e{idx}", fontsize=9, va="bottom")
    for idx, seg in enumerate(result.similar_segments, start=1):
        ax3.axvspan(seg.start_sec, seg.end_sec, color="#ffca28", alpha=0.18)
        ax3.text(seg.start_sec, 0.94, f"s{idx}", fontsize=8, va="bottom")
    ax3.set_title("Точные и похожие временные фрагменты, влияющие на score", fontsize=14)
    ax3.set_xlabel("Время, сек")
    ax3.set_ylim(-0.02, 1.08)
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right")

    ax4 = fig.add_subplot(5, 1, 4)
    spectrogram_signal = result.adversarial_waveform.astype(np.float32) + 1e-8
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in log10", category=RuntimeWarning)
        ax4.specgram(spectrogram_signal, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    ax4.set_title("Spectrogram: максимизирующий аудио-образ", fontsize=14)
    ax4.set_ylabel("Частота, Гц")
    ax4.set_xlabel("Время, сек")

    ax5 = fig.add_subplot(5, 1, 5)
    scores = history.get("score_per_step") or []
    losses = history.get("loss_per_step") or []
    if scores:
        ax5.plot(range(len(scores)), scores, color="#00897b", linewidth=1.6, label="P(target)")
    else:
        ax5.plot([0, 1], [result.original_score, result.final_score], color="#00897b", linewidth=1.6, marker="o", label="P(target)")
        ax5.text(
            0.5,
            min(1.0, max(result.original_score, result.final_score) + 0.08),
            "Для выбранного метода доступен итоговый прирост; пошаговая история пишется для Gradient ascent.",
            ha="center",
            fontsize=10,
        )
    if losses:
        ax5_twin = ax5.twinx()
        ax5_twin.plot(range(1, len(losses) + 1), losses, color="#8e24aa", linewidth=1.0, alpha=0.65, label="loss")
        ax5_twin.set_ylabel("Loss")
    ax5.set_title("История оптимизации", fontsize=14)
    ax5.set_xlabel("Шаг")
    ax5.set_ylabel("Вероятность цели")
    ax5.set_ylim(0.0, 1.05)
    ax5.grid(alpha=0.2)
    ax5.legend(loc="lower right")

    fig.tight_layout()
    return fig


def create_probability_figure(before: dict[str, float], after: dict[str, float], target_word: str):
    labels = list(before.keys())
    x = np.arange(len(labels))
    width = 0.36
    fig = plt.figure(figsize=(12, 5), facecolor="#fcfbf7")
    ax = fig.add_subplot(1, 1, 1)
    before_vals = [before[k] for k in labels]
    after_vals = [after[k] for k in labels]
    colors_before = ["#b0bec5" if label != target_word else "#ffcc80" for label in labels]
    colors_after = ["#90a4ae" if label != target_word else "#66bb6a" for label in labels]

    ax.bar(x - width / 2, before_vals, width=width, label="До", color=colors_before)
    ax.bar(x + width / 2, after_vals, width=width, label="После", color=colors_after)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Вероятность")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Вероятности классов. Целевое слово: «{target_word}»", fontsize=14)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()

    fig.tight_layout()
    return fig
