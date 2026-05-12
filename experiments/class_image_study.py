from __future__ import annotations

import copy
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from analysis.pipeline import run_full_pipeline
from utils.config import AnalysisConfig, AttackConfig, AudioConfig, dataclass_to_json_dict
from utils.export import make_run_dir, save_json, segments_to_dataframe


@dataclass
class ClassImageStudyResult:
    study_id: str
    study_dir: Path
    runs_df: pd.DataFrame
    condition_summary_df: pd.DataFrame
    summary: dict[str, Any]
    best_audio_path: str
    best_by_condition_paths: dict[str, str]
    prototype_mean_audio_path: str | None
    similarity_matrix_csv_path: str | None
    prototype_summary_json_path: str | None
    prototype_summary_plot_path: str | None
    spectral_stability_json_path: str | None
    spectral_stability_plot_path: str | None
    runs_csv_path: str
    condition_summary_csv_path: str
    summary_json_path: str
    report_path: str
    manifest_json_path: str


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _save_run_exports(result, run_dir: Path) -> tuple[str, str]:
    segments_csv = run_dir / "segments.csv"
    class_image_segments_csv = run_dir / "class_image_segments.csv"
    segments_df = segments_to_dataframe(result.segments)
    segments_df.to_csv(segments_csv, index=False)
    segments_df.to_csv(class_image_segments_csv, index=False)
    save_json(run_dir / "summary.json", result.summary_dict())
    return str(segments_csv), str(class_image_segments_csv)


def _run_row_from_result(result, run_dir: Path, seed: int) -> dict[str, Any]:
    metadata = result.metadata or {}
    class_image = metadata.get("class_image", {})
    _, class_image_segments_csv = _save_run_exports(result, run_dir)
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": result.model_name,
        "target_word": result.target_word,
        "input_mode": result.input_mode,
        "seed": seed,
        "method": result.attack_method,
        "objective": metadata.get("objective", class_image.get("objective", "")),
        "original_score": float(result.original_score),
        "final_score": float(result.final_score),
        "score_gain": float(result.score_gain),
        "success": bool(result.success),
        "goal_reached": bool(result.goal_reached),
        "steps_run": metadata.get("steps_run", metadata.get("optimization_history", {}).get("steps_run")),
        "early_stopping_reason": metadata.get("early_stopping_reason", ""),
        "snr_db": _safe_float(metadata.get("snr_db")),
        "l2": _safe_float(metadata.get("delta_l2")),
        "linf": _safe_float(metadata.get("delta_linf")),
        "exact_segments_count": int(metadata.get("exact_segments_count", len(result.exact_segments))),
        "similar_segments_count": int(metadata.get("similar_segments_count", len(result.similar_segments))),
        "class_image_audio_path": class_image.get("class_image_audio_path", metadata.get("saved_audio", {}).get("maximized", "")),
        "summary_path": str(run_dir / "summary.json"),
        "class_image_segments_csv": class_image_segments_csv,
        "error": "",
    }


def _error_row(
    run_dir: Path,
    model_name: str,
    target_word: str,
    input_mode: str,
    seed: int,
    attack_config: AttackConfig,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "target_word": target_word,
        "input_mode": input_mode,
        "seed": seed,
        "method": attack_config.method,
        "objective": attack_config.objective,
        "original_score": None,
        "final_score": None,
        "score_gain": None,
        "success": False,
        "goal_reached": False,
        "steps_run": None,
        "early_stopping_reason": "error",
        "snr_db": None,
        "l2": None,
        "linf": None,
        "exact_segments_count": 0,
        "similar_segments_count": 0,
        "class_image_audio_path": "",
        "summary_path": "",
        "class_image_segments_csv": "",
        "error": str(exc),
    }


def _aggregate(rows_df: pd.DataFrame, study_dir: Path, study_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    valid = rows_df[rows_df["error"].fillna("") == ""].copy()
    if valid.empty:
        return {
            "study_id": study_id,
            "study_dir": str(study_dir),
            "total_runs": int(len(rows_df)),
            "completed_runs": 0,
            "failed_runs": int(len(rows_df)),
            "success_rate": 0.0,
            "mean_final_score": None,
            "std_final_score": None,
            "best_final_score": None,
            "mean_score_gain": None,
            "mean_snr_db": None,
            "mean_l2": None,
            "mean_linf": None,
            "best_input_mode": None,
            "best_seed": None,
            "best_run_dir": None,
            "best_class_image_audio_path": None,
            "parameters": parameters,
            "interpretation": "Все запуски исследования завершились ошибкой; статистика образа класса не построена.",
        }

    best_idx = valid["final_score"].astype(float).idxmax()
    best_row = valid.loc[best_idx]
    return {
        "study_id": study_id,
        "study_dir": str(study_dir),
        "total_runs": int(len(rows_df)),
        "completed_runs": int(len(valid)),
        "failed_runs": int(len(rows_df) - len(valid)),
        "success_rate": float(valid["success"].mean()),
        "mean_final_score": _safe_float(valid["final_score"].mean()),
        "std_final_score": _safe_float(valid["final_score"].std(ddof=0)),
        "best_final_score": _safe_float(best_row["final_score"]),
        "mean_score_gain": _safe_float(valid["score_gain"].mean()),
        "mean_snr_db": _safe_float(valid["snr_db"].mean()),
        "mean_l2": _safe_float(valid["l2"].mean()),
        "mean_linf": _safe_float(valid["linf"].mean()),
        "best_input_mode": str(best_row["input_mode"]),
        "best_seed": int(best_row["seed"]),
        "best_run_dir": str(best_row["run_dir"]),
        "best_class_image_audio_path": str(best_row["class_image_audio_path"]),
        "parameters": parameters,
        "interpretation": (
            "Исследование показывает устойчивость и вариативность входного образа класса при разных "
            "начальных условиях и seed. Для gradv_ru_kws_baseline результат описывает компактную KWS-модель, "
            "а не промышленную ASR-систему."
        ),
    }


def _build_condition_summary(valid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "input_mode",
                "runs",
                "success_rate",
                "mean_final_score",
                "std_final_score",
                "min_final_score",
                "max_final_score",
                "mean_score_gain",
                "std_score_gain",
                "best_seed",
                "best_run_dir",
                "best_class_image_audio_path",
            ]
        )
    for input_mode, group in valid.groupby("input_mode", dropna=False):
        best_idx = group["final_score"].astype(float).idxmax()
        best_row = group.loc[best_idx]
        rows.append(
            {
                "input_mode": str(input_mode),
                "runs": int(len(group)),
                "success_rate": float(group["success"].mean()),
                "mean_final_score": _safe_float(group["final_score"].mean()),
                "std_final_score": _safe_float(group["final_score"].std(ddof=0)),
                "min_final_score": _safe_float(group["final_score"].min()),
                "max_final_score": _safe_float(group["final_score"].max()),
                "mean_score_gain": _safe_float(group["score_gain"].mean()),
                "std_score_gain": _safe_float(group["score_gain"].std(ddof=0)),
                "best_seed": int(best_row["seed"]),
                "best_run_dir": str(best_row["run_dir"]),
                "best_class_image_audio_path": str(best_row["class_image_audio_path"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_final_score", "mean_final_score"], ascending=False).reset_index(drop=True)


def _copy_best_by_condition(condition_summary_df: pd.DataFrame, study_dir: Path) -> dict[str, str]:
    output_dir = study_dir / "best_by_condition"
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for row in condition_summary_df.itertuples(index=False):
        source = Path(str(row.best_class_image_audio_path))
        if not source.exists():
            continue
        safe_mode = str(row.input_mode).replace("/", "_").replace("\\", "_").replace(" ", "_")
        destination = output_dir / f"{safe_mode}_best.wav"
        shutil.copy2(source, destination)
        copied[str(row.input_mode)] = str(destination)
    return copied


def _mode_label(input_mode: str | None) -> str:
    labels = {
        "maximize_from_noise": "старт из шума",
        "maximize_from_silence": "старт из тишины",
        "attack_uploaded_audio": "атака загруженного аудио",
    }
    return labels.get(str(input_mode), str(input_mode))


def _build_study_conclusion(summary: dict[str, Any], condition_summary_df: pd.DataFrame) -> str:
    if not summary.get("completed_runs"):
        return "Вывод не построен: нет успешно завершенных запусков исследования."

    best_mode = summary.get("best_input_mode")
    best_score = summary.get("best_final_score")
    success_rate = float(summary.get("success_rate") or 0.0)
    std_final = float(summary.get("std_final_score") or 0.0)
    target_word = summary.get("parameters", {}).get("target_word", "целевой класс")
    model_name = summary.get("parameters", {}).get("model_name", "модель")

    if success_rate >= 0.8 and std_final <= 0.05:
        stability = "образ класса находится устойчиво: большинство запусков дают близкий высокий score."
    elif std_final >= 0.12:
        stability = "результат заметно зависит от начальных условий и seed: найденные образы вариативны."
    else:
        stability = "образ класса находится, но есть умеренная вариативность между запусками."

    comparison = ""
    if not condition_summary_df.empty and {"maximize_from_noise", "maximize_from_silence"}.issubset(set(condition_summary_df["input_mode"])):
        by_mode = condition_summary_df.set_index("input_mode")
        noise_score = float(by_mode.loc["maximize_from_noise", "mean_final_score"])
        silence_score = float(by_mode.loc["maximize_from_silence", "mean_final_score"])
        if noise_score > silence_score + 0.03:
            comparison = " Старт из шума дал score выше старта из тишины: шум дает оптимизации больше исходных вариаций."
        elif silence_score > noise_score + 0.03:
            comparison = " Старт из тишины дал score выше старта из шума: признаки класса формируются даже из почти пустого сигнала."
        else:
            comparison = " Старт из шума и старт из тишины дают близкие результаты, значит найденный образ не привязан к одному начальному условию."

    baseline_note = ""
    if model_name == "gradv_ru_kws_baseline":
        baseline_note = " Для gradv_ru_kws_baseline это описание поведения компактной KWS-модели, а не универсальный акустический эталон русской речи."

    return (
        f"Для модели `{model_name}` и класса `{target_word}` лучший входной образ получен в режиме "
        f"`{best_mode}` ({_mode_label(best_mode)}) с final score {best_score:.4f}. "
        f"По серии запусков {stability}{comparison}{baseline_note}"
    )


def _write_report(path: Path, summary: dict[str, Any], condition_summary_df: pd.DataFrame) -> str:
    lines = [
        "# Class Image Study",
        "",
        f"- study_id: `{summary.get('study_id')}`",
        f"- total_runs: {summary.get('total_runs')}",
        f"- completed_runs: {summary.get('completed_runs')}",
        f"- success_rate: {summary.get('success_rate')}",
        f"- mean_final_score: {summary.get('mean_final_score')}",
        f"- best_final_score: {summary.get('best_final_score')}",
        f"- best_input_mode: `{summary.get('best_input_mode')}`",
        f"- best_seed: `{summary.get('best_seed')}`",
        f"- best_run_dir: `{summary.get('best_run_dir')}`",
        f"- best_class_image_audio_path: `{summary.get('best_class_image_audio_path')}`",
        "",
        "## Interpretation",
        "",
        str(summary.get("study_conclusion", summary.get("interpretation", ""))),
        "",
        "## Condition Summary",
        "",
    ]
    if condition_summary_df.empty:
        lines.append("No completed condition-level runs.")
    else:
        lines.append("```text")
        lines.append(condition_summary_df.to_string(index=False))
        lines.append("```")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _write_manifest(path: Path, summary: dict[str, Any]) -> str:
    payload = {
        "study_id": summary.get("study_id"),
        "study_dir": summary.get("study_dir"),
        "exports": summary.get("exports", {}),
        "best_by_condition": summary.get("best_by_condition", {}),
        "best_class_image_audio_path": summary.get("best_class_image_audio_path"),
    }
    save_json(path, payload)
    return str(path)


def _load_class_images(valid_df: pd.DataFrame) -> list[np.ndarray]:
    waves: list[np.ndarray] = []
    for raw_path in valid_df["class_image_audio_path"].dropna().tolist():
        path = Path(str(raw_path))
        if not path.exists():
            continue
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size:
            waves.append(audio.astype(np.float32))
    if not waves:
        return []
    min_len = min(len(wave) for wave in waves)
    return [wave[:min_len] for wave in waves if len(wave) >= min_len]


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _frame_signal(wave: np.ndarray, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    if len(wave) < n_fft:
        wave = np.pad(wave, (0, n_fft - len(wave)))
    frame_count = 1 + (len(wave) - n_fft) // hop
    frames = np.empty((frame_count, n_fft), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop
        frames[idx] = wave[start : start + n_fft]
    return frames


def _magnitude_spectrogram(wave: np.ndarray, sample_rate: int, n_fft: int = 512, hop: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = _frame_signal(wave.astype(np.float32), n_fft=n_fft, hop=hop)
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1)).T.astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).astype(np.float32)
    times = (np.arange(spectrum.shape[1], dtype=np.float32) * hop / float(sample_rate)).astype(np.float32)
    return spectrum, freqs, times


def _spectral_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.log1p(a.astype(np.float64)).ravel()
    b = np.log1p(b.astype(np.float64)).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _spectral_band_energy(mean_spectrogram: np.ndarray, freqs: np.ndarray) -> list[dict[str, float]]:
    if freqs.size == 0:
        return []
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, int(freqs[-1]) + 1)]
    total = float(mean_spectrogram.sum() + 1e-12)
    rows: list[dict[str, float]] = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            continue
        energy = float(mean_spectrogram[mask].sum())
        rows.append({"low_hz": float(low), "high_hz": float(high), "energy_share": float(energy / total)})
    return rows


def _build_spectral_stability(matrix: np.ndarray, sample_rate: int, prototype_dir: Path) -> dict[str, Any]:
    specs: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    times: np.ndarray | None = None
    for wave in matrix:
        spec, freqs, times = _magnitude_spectrogram(wave, sample_rate)
        specs.append(spec)
    spec_matrix = np.stack(specs, axis=0).astype(np.float32)
    mean_spec = spec_matrix.mean(axis=0)
    std_spec = spec_matrix.std(axis=0)
    coeff_var = std_spec / (mean_spec + 1e-6)
    stability_map = 1.0 / (1.0 + coeff_var)

    spectral_similarity = np.eye(len(specs), dtype=np.float32)
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            value = _spectral_cosine_similarity(specs[i], specs[j])
            spectral_similarity[i, j] = value
            spectral_similarity[j, i] = value
    upper = spectral_similarity[np.triu_indices(len(specs), k=1)]
    mean_spectral_similarity = float(upper.mean()) if upper.size else None
    std_spectral_similarity = float(upper.std()) if upper.size else None
    stable_mask = (stability_map >= 0.70) & (mean_spec >= np.percentile(mean_spec, 60))
    stable_ratio = float(stable_mask.mean())
    freqs_arr = freqs if freqs is not None else np.array([], dtype=np.float32)
    times_arr = times if times is not None else np.array([], dtype=np.float32)
    band_energy = _spectral_band_energy(mean_spec, freqs_arr)
    dominant = max(band_energy, key=lambda item: item["energy_share"]) if band_energy else None

    mean_csv = prototype_dir / "spectrogram_mean.csv"
    std_csv = prototype_dir / "spectrogram_std.csv"
    stability_csv = prototype_dir / "spectrogram_stability_map.csv"
    similarity_csv = prototype_dir / "spectral_similarity_matrix.csv"
    pd.DataFrame(mean_spec).to_csv(mean_csv, index=False)
    pd.DataFrame(std_spec).to_csv(std_csv, index=False)
    pd.DataFrame(stability_map).to_csv(stability_csv, index=False)
    pd.DataFrame(spectral_similarity).to_csv(similarity_csv, index=False)

    plot_path = prototype_dir / "spectrogram_stability.png"
    fig = plt.figure(figsize=(13, 10), facecolor="#fcfbf7")
    extent = [0.0, float(times_arr[-1]) if len(times_arr) else 0.0, 0.0, float(freqs_arr[-1]) if len(freqs_arr) else 0.0]
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(20.0 * np.log10(mean_spec + 1e-8), origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax1.set_title("Mean magnitude spectrogram")
    ax1.set_xlabel("Time, sec")
    ax1.set_ylabel("Frequency, Hz")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(stability_map, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
    ax2.set_title("Spectral stability map")
    ax2.set_xlabel("Time, sec")
    ax2.set_ylabel("Frequency, Hz")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)

    payload = {
        "available": True,
        "n_fft": 512,
        "hop_length": 128,
        "mean_spectral_similarity": mean_spectral_similarity,
        "std_spectral_similarity": std_spectral_similarity,
        "stable_time_frequency_ratio": stable_ratio,
        "dominant_frequency_band_hz": dominant,
        "band_energy_distribution": band_energy,
        "spectrogram_mean_csv_path": str(mean_csv),
        "spectrogram_std_csv_path": str(std_csv),
        "spectrogram_stability_map_csv_path": str(stability_csv),
        "spectral_similarity_matrix_csv_path": str(similarity_csv),
        "spectrogram_stability_plot_path": str(plot_path),
        "interpretation": (
            "Spectral stability shows which time-frequency zones are repeated across successful class images. "
            "High spectral similarity means that runs converge to related frequency patterns, even when the "
            "waveform phase or exact sample values differ."
        ),
    }
    json_path = prototype_dir / "spectrogram_stability.json"
    save_json(json_path, payload)
    payload["spectrogram_stability_json_path"] = str(json_path)
    return {
        "summary": payload,
        "json_path": str(json_path),
        "plot_path": str(plot_path),
        "mean_spectrogram": mean_spec,
        "stability_map": stability_map,
        "freqs": freqs_arr,
        "times": times_arr,
    }


def _save_prototype_plot(
    path: Path,
    mean_wave: np.ndarray,
    std_wave: np.ndarray,
    sample_rate: int,
    spectral: dict[str, Any] | None = None,
) -> str:
    time_axis = np.arange(len(mean_wave), dtype=np.float32) / float(sample_rate)
    rows = 3 if spectral else 2
    fig = plt.figure(figsize=(13, 4 * rows), facecolor="#fcfbf7")
    ax1 = fig.add_subplot(rows, 1, 1)
    ax1.plot(time_axis, mean_wave, color="#00695c", linewidth=1.2, label="mean waveform")
    ax1.fill_between(time_axis, mean_wave - std_wave, mean_wave + std_wave, color="#80cbc4", alpha=0.35, label="+/- std")
    ax1.set_title("Prototype waveform: mean class image")
    ax1.set_xlabel("Time, sec")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.2)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(rows, 1, 2)
    if spectral:
        mean_db = 20.0 * np.log10(np.asarray(spectral["mean_spectrogram"]) + 1e-8)
        freqs = np.asarray(spectral["freqs"])
        times = np.asarray(spectral["times"])
        extent = [0.0, float(times[-1]) if len(times) else 0.0, 0.0, float(freqs[-1]) if len(freqs) else 0.0]
        ax2.imshow(mean_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
    else:
        ax2.specgram(mean_wave + 1e-8, NFFT=512, Fs=sample_rate, noverlap=384, cmap="magma")
    ax2.set_title("Mean class image spectrogram")
    ax2.set_xlabel("Time, sec")
    ax2.set_ylabel("Frequency, Hz")

    if spectral:
        ax3 = fig.add_subplot(rows, 1, 3)
        freqs = np.asarray(spectral["freqs"])
        times = np.asarray(spectral["times"])
        extent = [0.0, float(times[-1]) if len(times) else 0.0, 0.0, float(freqs[-1]) if len(freqs) else 0.0]
        ax3.imshow(np.asarray(spectral["stability_map"]), origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
        ax3.set_title("Spectral stability map: stable time-frequency zones")
        ax3.set_xlabel("Time, sec")
        ax3.set_ylabel("Frequency, Hz")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)

def _build_prototype_analysis(valid_df: pd.DataFrame, study_dir: Path, sample_rate: int) -> dict[str, Any]:
    prototype_dir = study_dir / "prototype"
    prototype_dir.mkdir(parents=True, exist_ok=True)
    waves = _load_class_images(valid_df)
    if len(waves) < 2:
        summary = {
            "available": False,
            "reason": "Для prototype-анализа нужно минимум два успешно найденных образа класса.",
            "num_images": len(waves),
        }
        summary_path = prototype_dir / "prototype_summary.json"
        save_json(summary_path, summary)
        return {
            "summary": summary,
            "mean_audio_path": None,
            "similarity_matrix_csv_path": None,
            "summary_json_path": str(summary_path),
            "plot_path": None,
            "spectral_json_path": None,
            "spectral_plot_path": None,
        }

    matrix = np.stack(waves, axis=0).astype(np.float32)
    mean_wave = matrix.mean(axis=0)
    std_wave = matrix.std(axis=0)
    mean_abs_wave = np.abs(matrix).mean(axis=0)
    similarity = np.eye(len(waves), dtype=np.float32)
    for i in range(len(waves)):
        for j in range(i + 1, len(waves)):
            value = _safe_corr(matrix[i], matrix[j])
            similarity[i, j] = value
            similarity[j, i] = value
    upper = similarity[np.triu_indices(len(waves), k=1)]
    mean_similarity = float(upper.mean()) if upper.size else None
    std_similarity = float(upper.std()) if upper.size else None

    mean_audio_path = prototype_dir / "prototype_mean.wav"
    sf.write(mean_audio_path, np.clip(mean_wave, -1.0, 1.0), sample_rate)
    stats_csv = prototype_dir / "prototype_std.csv"
    pd.DataFrame(
        {
            "sample_index": np.arange(len(mean_wave)),
            "mean_waveform": mean_wave,
            "std_waveform": std_wave,
            "mean_absolute_waveform": mean_abs_wave,
        }
    ).to_csv(stats_csv, index=False)
    similarity_csv = prototype_dir / "similarity_matrix.csv"
    pd.DataFrame(similarity).to_csv(similarity_csv, index=False)
    spectral = _build_spectral_stability(matrix, sample_rate, prototype_dir)
    plot_path = prototype_dir / "prototype_summary.png"
    _save_prototype_plot(plot_path, mean_wave, std_wave, sample_rate, spectral=spectral)
    summary = {
        "available": True,
        "num_images": len(waves),
        "waveform_length": int(matrix.shape[1]),
        "mean_pairwise_similarity": mean_similarity,
        "std_pairwise_similarity": std_similarity,
        "prototype_mean_audio_path": str(mean_audio_path),
        "prototype_std_csv_path": str(stats_csv),
        "similarity_matrix_csv_path": str(similarity_csv),
        "prototype_summary_plot_path": str(plot_path),
        "spectral_stability": spectral["summary"],
        "interpretation": (
            "Высокая корреляция означает, что разные запуски приходят к похожему входному образу класса. "
            "Низкая корреляция означает, что модель допускает несколько разных сигналов, активирующих один класс."
        ),
    }
    summary_path = prototype_dir / "prototype_summary.json"
    save_json(summary_path, summary)
    return {
        "summary": summary,
        "mean_audio_path": str(mean_audio_path),
        "similarity_matrix_csv_path": str(similarity_csv),
        "summary_json_path": str(summary_path),
        "plot_path": str(plot_path),
        "spectral_json_path": spectral["json_path"],
        "spectral_plot_path": spectral["plot_path"],
    }


def run_class_image_study(
    *,
    adapter,
    target_word: str,
    input_modes: list[str],
    repeats: int,
    attack_config_template: AttackConfig,
    analysis_config: AnalysisConfig,
    audio_config: AudioConfig,
    output_root: Path,
    uploaded_audio: np.ndarray | None = None,
    seed_start: int | None = 1000,
) -> ClassImageStudyResult:
    if repeats < 2:
        raise ValueError("Study mode requires at least 2 repeats to produce real statistics.")
    if not input_modes:
        raise ValueError("At least one input mode is required for a class image study.")

    study_id = uuid.uuid4().hex[:12]
    study_dir = output_root / study_id
    runs_root = study_dir / "runs"
    study_dir.mkdir(parents=True, exist_ok=False)
    runs_root.mkdir(parents=True, exist_ok=True)

    seed_rng = np.random.default_rng()
    rows: list[dict[str, Any]] = []
    for mode_index, input_mode in enumerate(input_modes):
        for repeat_index in range(repeats):
            seed = int(seed_rng.integers(0, 2_147_483_000)) if seed_start is None else int(seed_start + mode_index * repeats + repeat_index)
            attack_config = copy.deepcopy(attack_config_template)
            attack_config.input_mode = input_mode
            attack_config.seed = seed
            run_dir = make_run_dir(runs_root)
            try:
                if input_mode == "attack_uploaded_audio" and uploaded_audio is None:
                    raise ValueError("attack_uploaded_audio was selected, but no uploaded audio was provided.")
                result = run_full_pipeline(
                    adapter=adapter,
                    uploaded_audio=uploaded_audio if input_mode == "attack_uploaded_audio" else None,
                    target_word=target_word,
                    audio_config=audio_config,
                    attack_config=attack_config,
                    analysis_config=analysis_config,
                    run_dir=run_dir,
                )
                rows.append(_run_row_from_result(result, run_dir, seed))
            except Exception as exc:  # keep the study useful even if one condition fails
                rows.append(_error_row(run_dir, getattr(adapter, "model_name", "unknown"), target_word, input_mode, seed, attack_config, exc))

    runs_df = pd.DataFrame(rows)
    valid_df = runs_df[runs_df["error"].fillna("") == ""].copy()
    condition_summary_df = _build_condition_summary(valid_df)
    parameters = {
        "model_name": getattr(adapter, "model_name", "unknown"),
        "target_word": target_word,
        "input_modes": input_modes,
        "repeats": repeats,
        "seed_start": seed_start,
        "audio": dataclass_to_json_dict(audio_config),
        "attack_template": dataclass_to_json_dict(attack_config_template),
        "analysis": dataclass_to_json_dict(analysis_config),
    }
    summary = _aggregate(runs_df, study_dir, study_id, parameters)
    best_by_condition_paths = _copy_best_by_condition(condition_summary_df, study_dir)
    if best_by_condition_paths:
        condition_summary_df["copied_best_audio_path"] = condition_summary_df["input_mode"].map(best_by_condition_paths)
    summary["condition_summary"] = condition_summary_df.to_dict(orient="records")
    summary["best_by_condition"] = best_by_condition_paths
    summary["study_conclusion"] = _build_study_conclusion(summary, condition_summary_df)
    prototype = _build_prototype_analysis(valid_df, study_dir, audio_config.sample_rate)
    summary["prototype_analysis"] = prototype["summary"]
    summary["mean_pairwise_similarity"] = prototype["summary"].get("mean_pairwise_similarity")
    summary["prototype_mean_audio_path"] = prototype["mean_audio_path"]
    summary["similarity_matrix_csv_path"] = prototype["similarity_matrix_csv_path"]
    summary["spectrogram_stability_json_path"] = prototype["spectral_json_path"]
    summary["spectrogram_stability_plot_path"] = prototype["spectral_plot_path"]

    best_audio_path = ""
    source_best_audio = summary.get("best_class_image_audio_path")
    if source_best_audio:
        source = Path(str(source_best_audio))
        if source.exists():
            best_audio = study_dir / "best_class_image.wav"
            shutil.copy2(source, best_audio)
            best_audio_path = str(best_audio)
            summary["best_class_image_audio_path"] = best_audio_path

    if not best_audio_path:
        summary["best_class_image_audio_path"] = None

    runs_csv_path = study_dir / "study_runs.csv"
    condition_summary_csv_path = study_dir / "condition_summary.csv"
    summary_json_path = study_dir / "study_summary.json"
    report_path = study_dir / "study_report.md"
    manifest_json_path = study_dir / "study_manifest.json"
    summary["exports"] = {
        "study_runs_csv": str(runs_csv_path),
        "condition_summary_csv": str(condition_summary_csv_path),
        "study_summary_json": str(summary_json_path),
        "study_report_md": str(report_path),
        "study_manifest_json": str(manifest_json_path),
        "best_class_image_wav": summary.get("best_class_image_audio_path"),
        "best_by_condition_dir": str(study_dir / "best_by_condition"),
        "prototype_mean_wav": prototype["mean_audio_path"],
        "prototype_summary_json": prototype["summary_json_path"],
        "prototype_summary_png": prototype["plot_path"],
        "similarity_matrix_csv": prototype["similarity_matrix_csv_path"],
        "spectrogram_stability_json": prototype["spectral_json_path"],
        "spectrogram_stability_png": prototype["spectral_plot_path"],
    }
    runs_df.to_csv(runs_csv_path, index=False)
    condition_summary_df.to_csv(condition_summary_csv_path, index=False)
    save_json(summary_json_path, summary)
    _write_report(report_path, summary, condition_summary_df)
    _write_manifest(manifest_json_path, summary)

    completed_runs = int(summary.get("completed_runs") or 0)
    if completed_runs == 0:
        raise RuntimeError(f"Class image study failed: all {len(runs_df)} runs ended with errors. See {runs_csv_path}")

    return ClassImageStudyResult(
        study_id=study_id,
        study_dir=study_dir,
        runs_df=runs_df,
        condition_summary_df=condition_summary_df,
        summary=summary,
        best_audio_path=best_audio_path,
        best_by_condition_paths=best_by_condition_paths,
        prototype_mean_audio_path=prototype["mean_audio_path"],
        similarity_matrix_csv_path=prototype["similarity_matrix_csv_path"],
        prototype_summary_json_path=prototype["summary_json_path"],
        prototype_summary_plot_path=prototype["plot_path"],
        spectral_stability_json_path=prototype["spectral_json_path"],
        spectral_stability_plot_path=prototype["spectral_plot_path"],
        runs_csv_path=str(runs_csv_path),
        condition_summary_csv_path=str(condition_summary_csv_path),
        summary_json_path=str(summary_json_path),
        report_path=str(report_path),
        manifest_json_path=str(manifest_json_path),
    )
