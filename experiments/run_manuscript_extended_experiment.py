from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.generic_torch_adapter import GenericTorchAdapter
from adapters.mock_russian_kws_adapter import MockRussianKWSAdapterA
from analysis.pipeline import run_full_pipeline
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig, dataclass_to_json_dict
from utils.export import save_json, segments_to_dataframe


BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_baseline.pt"
BASELINE_VOCAB_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_vocab.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "manuscript_extended_experiment"
CURRENT_WEIGHTS = {"saliency": 0.35, "occlusion": 0.25, "isolated": 0.15, "change": 0.25}
WEIGHT_PROFILES = {
    "current": CURRENT_WEIGHTS,
    "saliency-heavy": {"saliency": 0.50, "occlusion": 0.20, "isolated": 0.10, "change": 0.20},
    "occlusion-heavy": {"saliency": 0.20, "occlusion": 0.50, "isolated": 0.10, "change": 0.20},
    "isolated-heavy": {"saliency": 0.20, "occlusion": 0.20, "isolated": 0.40, "change": 0.20},
    "balanced": {"saliency": 0.25, "occlusion": 0.25, "isolated": 0.25, "change": 0.25},
}
READABLE_RU_TARGETS = ["да", "нет", "стоп", "вперёд", "назад", "привет", "включи", "выключи"]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: pd.Series | list[Any]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return _safe_float(numeric.mean()) if not numeric.empty else None


def _std(values: pd.Series | list[Any]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return _safe_float(numeric.std(ddof=0)) if not numeric.empty else None


def _best_value(values: pd.Series | list[Any]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return _safe_float(numeric.max()) if not numeric.empty else None


def _mojibake_alias(text: str) -> str:
    try:
        return text.encode("utf-8").decode("cp1251")
    except UnicodeError:
        return text


def _target_aliases(vocabulary: list[str]) -> dict[str, str]:
    aliases: dict[str, str] = {word: word for word in vocabulary}
    for readable in READABLE_RU_TARGETS:
        mojibake = _mojibake_alias(readable)
        if mojibake in vocabulary:
            aliases[readable] = mojibake
    return aliases


def _display_word(word: str, aliases: dict[str, str]) -> str:
    for readable, actual in aliases.items():
        if actual == word and readable in READABLE_RU_TARGETS:
            return readable
    return word


def _resolve_targets(raw_targets: list[str] | None, vocabulary: list[str]) -> list[str]:
    if not raw_targets:
        return list(vocabulary)
    aliases = _target_aliases(vocabulary)
    resolved: list[str] = []
    missing: list[str] = []
    for raw in raw_targets:
        target = aliases.get(raw, raw)
        if target not in vocabulary:
            missing.append(raw)
        elif target not in resolved:
            resolved.append(target)
    if missing:
        raise ValueError(
            "Unsupported target(s): "
            + ", ".join(repr(item) for item in missing)
            + ". Available vocabulary: "
            + ", ".join(repr(item) for item in vocabulary)
        )
    return resolved


def _load_adapter(model_path: Path) -> GenericTorchAdapter:
    adapter = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline" if model_path == BASELINE_MODEL_PATH else model_path.stem,
    )
    adapter.load_model(model_path)
    return adapter


def _build_attack_config(args: argparse.Namespace, input_mode: str, seed: int) -> AttackConfig:
    return AttackConfig(
        method=args.method,
        input_mode=input_mode,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        max_delta=args.max_delta,
        l2_weight=args.l2_weight,
        tv_weight=args.tv_weight,
        goal_score=args.goal_score,
        objective=args.objective,
        seed=seed,
        explicit_attack=True,
        prototype_emphasis=1.4,
        reference_mix_min=0.8,
    )


def _save_run_exports(result: Any, run_dir: Path) -> dict[str, str]:
    segments_csv = run_dir / "segments.csv"
    class_image_segments_csv = run_dir / "class_image_segments.csv"
    exact_segments_csv = run_dir / "exact_segments.csv"
    similar_segments_csv = run_dir / "similar_segments.csv"
    segments_to_dataframe(result.segments).to_csv(segments_csv, index=False)
    segments_to_dataframe(result.segments).to_csv(class_image_segments_csv, index=False)
    segments_to_dataframe(result.exact_segments).to_csv(exact_segments_csv, index=False)
    segments_to_dataframe(result.similar_segments).to_csv(similar_segments_csv, index=False)
    summary_json = save_json(run_dir / "summary.json", result.summary_dict())
    return {
        "summary_path": str(summary_json),
        "segments_csv": str(segments_csv),
        "class_image_segments_csv": str(class_image_segments_csv),
        "exact_segments_csv": str(exact_segments_csv),
        "similar_segments_csv": str(similar_segments_csv),
    }


def _run_one(
    *,
    adapter: Any,
    target_word: str,
    input_mode: str,
    seed: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    attack_config = _build_attack_config(args, input_mode, seed)
    started = time.perf_counter()
    try:
        result = run_full_pipeline(
            adapter=adapter,
            uploaded_audio=None,
            target_word=target_word,
            audio_config=DEFAULT_AUDIO_CONFIG,
            attack_config=attack_config,
            analysis_config=DEFAULT_ANALYSIS_CONFIG,
            run_dir=run_dir,
        )
        exports = _save_run_exports(result, run_dir)
        elapsed = time.perf_counter() - started
        metadata = result.metadata or {}
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": result.model_name,
            "target_word": target_word,
            "input_mode": input_mode,
            "seed": seed,
            "method": result.attack_method,
            "objective": args.objective,
            "original_score": float(result.original_score),
            "final_score": float(result.final_score),
            "score_gain": float(result.score_gain),
            "success": bool(result.success),
            "goal_reached": bool(result.goal_reached),
            "steps_run": metadata.get("steps_run", metadata.get("optimization_history", {}).get("steps_run")),
            "runtime_sec": elapsed,
            "exact_fragments": int(metadata.get("exact_segments_count", len(result.exact_segments))),
            "similar_fragments": int(metadata.get("similar_segments_count", len(result.similar_segments))),
            "class_image_audio_path": metadata.get("saved_audio", {}).get("maximized", ""),
            "error": "",
            **exports,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "model_name": getattr(adapter, "model_name", "unknown"),
            "target_word": target_word,
            "input_mode": input_mode,
            "seed": seed,
            "method": args.method,
            "objective": args.objective,
            "original_score": None,
            "final_score": None,
            "score_gain": None,
            "success": False,
            "goal_reached": False,
            "steps_run": None,
            "runtime_sec": elapsed,
            "exact_fragments": 0,
            "similar_fragments": 0,
            "class_image_audio_path": "",
            "summary_path": "",
            "segments_csv": "",
            "class_image_segments_csv": "",
            "exact_segments_csv": "",
            "similar_segments_csv": "",
            "error": str(exc),
        }


def _interval_key(row: pd.Series) -> tuple[float, float]:
    return (round(float(row["start_sec"]), 4), round(float(row["end_sec"]), 4))


def _load_segments(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(str(path))
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _rank_segments(segments: pd.DataFrame, method: str, weights: dict[str, float] | None = None) -> pd.DataFrame:
    if segments.empty:
        return segments.copy()
    ranked = segments.copy()
    if method == "integrated":
        ranked["_rank_score"] = pd.to_numeric(ranked.get("combined_score"), errors="coerce")
    elif method == "saliency":
        ranked["_rank_score"] = pd.to_numeric(ranked.get("saliency_mean"), errors="coerce")
    elif method == "occlusion":
        ranked["_rank_score"] = pd.to_numeric(ranked.get("occlusion_drop"), errors="coerce")
    elif method == "isolated":
        ranked["_rank_score"] = pd.to_numeric(ranked.get("isolated_score"), errors="coerce")
    elif method == "signal-change":
        ranked["_rank_score"] = pd.to_numeric(ranked.get("signal_change_mean", ranked.get("attack_change_mean")), errors="coerce")
    elif method == "weighted":
        profile = weights or CURRENT_WEIGHTS
        ranked["_rank_score"] = (
            profile["saliency"] * pd.to_numeric(ranked.get("saliency_mean"), errors="coerce").fillna(0.0)
            + profile["occlusion"] * pd.to_numeric(ranked.get("occlusion_drop"), errors="coerce").fillna(0.0)
            + profile["isolated"] * pd.to_numeric(ranked.get("isolated_score"), errors="coerce").fillna(0.0)
            + profile["change"]
            * pd.to_numeric(ranked.get("signal_change_mean", ranked.get("attack_change_mean")), errors="coerce").fillna(0.0)
        )
    else:
        raise ValueError(f"Unknown segment ranking method: {method}")
    return ranked.sort_values(["_rank_score", "target_probability", "start_sec"], ascending=[False, False, True]).reset_index(drop=True)


def _overlap_with_top3(top: pd.Series, reference_top3: pd.DataFrame) -> float | None:
    if reference_top3.empty:
        return None
    start = float(top["start_sec"])
    end = float(top["end_sec"])
    length = max(1e-8, end - start)
    best = 0.0
    for _, ref in reference_top3.iterrows():
        inter = max(0.0, min(end, float(ref["end_sec"])) - max(start, float(ref["start_sec"])))
        best = max(best, inter / length)
    return _safe_float(best)


def _rank_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    if a.empty or b.empty:
        return None
    a_rank = {_interval_key(row): idx + 1 for idx, row in a.iterrows()}
    b_rank = {_interval_key(row): idx + 1 for idx, row in b.iterrows()}
    keys = [key for key in a_rank if key in b_rank]
    if len(keys) < 2:
        return None
    av = pd.Series([a_rank[key] for key in keys], dtype="float64")
    bv = pd.Series([b_rank[key] for key in keys], dtype="float64")
    corr = av.corr(bv, method="spearman")
    return _safe_float(corr)


def _summarize_targets(runs_df: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in runs_df.groupby("target_word", dropna=False):
        valid = group[group["error"].fillna("") == ""].copy()
        best_row = valid.sort_values("final_score", ascending=False).iloc[0] if not valid.empty else None
        best_segments = _load_segments(None if best_row is None else best_row.get("segments_csv"))
        integrated = _rank_segments(best_segments, "integrated")
        top = integrated.iloc[0] if not integrated.empty else None
        rows.append(
            {
                "target_word": target,
                "target_word_display": _display_word(str(target), aliases),
                "number_of_runs": int(len(group)),
                "completed_runs": int(len(valid)),
                "failed_runs": int(len(group) - len(valid)),
                "success_rate": float(valid["success"].mean()) if not valid.empty else 0.0,
                "mean_original_score": _mean(valid["original_score"]) if not valid.empty else None,
                "mean_final_score": _mean(valid["final_score"]) if not valid.empty else None,
                "std_final_score": _std(valid["final_score"]) if not valid.empty else None,
                "best_final_score": _best_value(valid["final_score"]) if not valid.empty else None,
                "mean_score_gain": _mean(valid["score_gain"]) if not valid.empty else None,
                "best_seed": None if best_row is None else int(best_row["seed"]),
                "best_input_mode": None if best_row is None else str(best_row["input_mode"]),
                "number_of_exact_fragments": None if best_row is None else int(best_row["exact_fragments"]),
                "number_of_similar_fragments": None if best_row is None else int(best_row["similar_fragments"]),
                "top_ranked_segment_interval": None if top is None else f"{float(top['start_sec']):.4f}-{float(top['end_sec']):.4f}",
                "mean_combined_segment_score_for_top_segments": _mean(integrated.head(3)["combined_score"]) if not integrated.empty else None,
                "best_run_output_path": None if best_row is None else str(best_row["run_dir"]),
                "best_segments_csv": None if best_row is None else str(best_row["segments_csv"]),
            }
        )
    return pd.DataFrame(rows).sort_values("target_word").reset_index(drop=True)


def _summarize_input_modes(runs_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for input_mode, group in runs_df.groupby("input_mode", dropna=False):
        valid = group[group["error"].fillna("") == ""].copy()
        rows.append(
            {
                "input_mode": input_mode,
                "number_of_runs": int(len(group)),
                "success_rate": float(valid["success"].mean()) if not valid.empty else 0.0,
                "mean_final_score": _mean(valid["final_score"]) if not valid.empty else None,
                "std_final_score": _std(valid["final_score"]) if not valid.empty else None,
                "mean_score_gain": _mean(valid["score_gain"]) if not valid.empty else None,
                "best_final_score": _best_value(valid["final_score"]) if not valid.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values("input_mode").reset_index(drop=True)


def _segment_ablation(target_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [
        ("integrated", "integrated"),
        ("saliency-only", "saliency"),
        ("occlusion-only", "occlusion"),
        ("isolated-score-only", "isolated"),
        ("signal-change-only", "signal-change"),
    ]
    for row in target_summary_df.itertuples(index=False):
        segments = _load_segments(row.best_segments_csv)
        integrated = _rank_segments(segments, "integrated")
        integrated_top3 = integrated.head(3)
        for label, method in methods:
            ranked = _rank_segments(segments, method)
            if ranked.empty:
                rows.append(
                    {
                        "target_word": row.target_word,
                        "ranking_method": label,
                        "top_segment_start": None,
                        "top_segment_end": None,
                        "top_segment_score": None,
                        "overlap_with_integrated_top_3": None,
                        "exact_or_similar_label": None,
                    }
                )
                continue
            top = ranked.iloc[0]
            rows.append(
                {
                    "target_word": row.target_word,
                    "ranking_method": label,
                    "top_segment_start": _safe_float(top["start_sec"]),
                    "top_segment_end": _safe_float(top["end_sec"]),
                    "top_segment_score": _safe_float(top["_rank_score"]),
                    "overlap_with_integrated_top_3": _overlap_with_top3(top, integrated_top3),
                    "exact_or_similar_label": top.get("rank_type"),
                }
            )
    return pd.DataFrame(rows)


def _weight_sensitivity(target_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in target_summary_df.itertuples(index=False):
        segments = _load_segments(row.best_segments_csv)
        current = _rank_segments(segments, "weighted", CURRENT_WEIGHTS)
        current_top3 = current.head(3)
        for label, weights in WEIGHT_PROFILES.items():
            ranked = _rank_segments(segments, "weighted", weights)
            if ranked.empty:
                rows.append(
                    {
                        "target_word": row.target_word,
                        "weight_profile": label,
                        "top_segment_start": None,
                        "top_segment_end": None,
                        "top_segment_score": None,
                        "overlap_with_current_top_3": None,
                        "rank_correlation_with_current": None,
                    }
                )
                continue
            top = ranked.iloc[0]
            rows.append(
                {
                    "target_word": row.target_word,
                    "weight_profile": label,
                    "top_segment_start": _safe_float(top["start_sec"]),
                    "top_segment_end": _safe_float(top["end_sec"]),
                    "top_segment_score": _safe_float(top["_rank_score"]),
                    "overlap_with_current_top_3": _overlap_with_top3(top, current_top3),
                    "rank_correlation_with_current": _rank_correlation(current, ranked),
                }
            )
    return pd.DataFrame(rows)


def _copy_best_audio(target_summary_df: pd.DataFrame, assets_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    assets_dir.mkdir(parents=True, exist_ok=True)
    for row in target_summary_df.itertuples(index=False):
        run_dir = Path(str(row.best_run_output_path)) if row.best_run_output_path else None
        source = run_dir / "maximized.wav" if run_dir else None
        if source and source.exists():
            safe = str(row.target_word_display).replace("/", "_").replace("\\", "_").replace(" ", "_")
            destination = assets_dir / f"best_{safe}.wav"
            shutil.copy2(source, destination)
            copied[str(row.target_word)] = str(destination)
    return copied


def _write_empty_csv(path: Path, columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)


def _plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_figures(
    figures_dir: Path,
    target_summary_df: pd.DataFrame,
    input_mode_summary_df: pd.DataFrame,
    segment_ablation_df: pd.DataFrame,
    weight_sensitivity_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame | None,
) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    labels = target_summary_df["target_word_display"].astype(str).tolist()
    final_scores = pd.to_numeric(target_summary_df["mean_final_score"], errors="coerce").fillna(0.0).tolist()
    gains = pd.to_numeric(target_summary_df["mean_score_gain"], errors="coerce").fillna(0.0).tolist()
    path = figures_dir / "target_word_final_scores.png"
    _plot_bar(path, labels, final_scores, "Mean final score by target word", "Mean final score")
    paths["target_word_final_scores"] = str(path)
    path = figures_dir / "target_word_score_gains.png"
    _plot_bar(path, labels, gains, "Mean score gain by target word", "Mean score gain")
    paths["target_word_score_gains"] = str(path)
    path = figures_dir / "input_mode_success_rates.png"
    _plot_bar(
        path,
        input_mode_summary_df["input_mode"].astype(str).tolist(),
        pd.to_numeric(input_mode_summary_df["success_rate"], errors="coerce").fillna(0.0).tolist(),
        "Success rate by input mode",
        "Success rate",
    )
    paths["input_mode_success_rates"] = str(path)

    path = figures_dir / "segment_ablation_top_windows.png"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not segment_ablation_df.empty:
        pivot = segment_ablation_df.pivot_table(
            index="target_word", columns="ranking_method", values="top_segment_start", aggfunc="first"
        )
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Top segment start by ranking method")
    ax.set_ylabel("Start time, sec")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths["segment_ablation_top_windows"] = str(path)

    path = figures_dir / "weight_sensitivity_overlap.png"
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if not weight_sensitivity_df.empty:
        pivot = weight_sensitivity_df.pivot_table(
            index="target_word", columns="weight_profile", values="overlap_with_current_top_3", aggfunc="mean"
        )
        pivot.plot(kind="bar", ax=ax)
    ax.set_title("Overlap with current top-3 segments")
    ax.set_ylabel("Overlap ratio")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    paths["weight_sensitivity_overlap"] = str(path)

    if model_comparison_df is not None and not model_comparison_df.empty:
        path = figures_dir / "model_comparison.png"
        labels = model_comparison_df["model_name"].astype(str) + " / " + model_comparison_df["target_word"].astype(str)
        _plot_bar(
            path,
            labels.tolist(),
            pd.to_numeric(model_comparison_df["mean_final_score"], errors="coerce").fillna(0.0).tolist(),
            "Model comparison mean final scores",
            "Mean final score",
        )
        paths["model_comparison"] = str(path)
    return paths


def _write_report(
    path: Path,
    *,
    summary: dict[str, Any],
    target_summary_df: pd.DataFrame,
    input_mode_summary_df: pd.DataFrame,
    segment_ablation_df: pd.DataFrame,
    weight_sensitivity_df: pd.DataFrame,
    model_comparison_df: pd.DataFrame | None,
    figure_paths: dict[str, str],
    unavailable_metrics: list[str],
) -> str:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows."
        clean = df.copy()
        clean = clean.where(pd.notna(clean), "")
        columns = [str(col) for col in clean.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, data_row in clean.iterrows():
            values = [str(data_row[col]).replace("\n", " ") for col in clean.columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "# GRADV Extended Manuscript Experiment",
        "",
        "## Experiment Configuration",
        "",
        f"- experiment_id: `{summary['experiment_id']}`",
        f"- model: `{summary['configuration']['model']}`",
        f"- method: `{summary['configuration']['method']}`",
        f"- objective: `{summary['configuration']['objective']}`",
        f"- steps: `{summary['configuration']['steps']}`",
        f"- repeats: `{summary['configuration']['repeats']}`",
        f"- input_modes: `{summary['configuration']['input_modes']}`",
        f"- seed_start: `{summary['configuration']['seed_start']}`",
        f"- learning_rate: `{summary['configuration']['learning_rate']}`",
        f"- max_delta: `{summary['configuration']['max_delta']}`",
        f"- l2_weight: `{summary['configuration']['l2_weight']}`",
        f"- tv_weight: `{summary['configuration']['tv_weight']}`",
        f"- goal_score: `{summary['configuration']['goal_score']}`",
        "",
        "## Hardware And Runtime",
        "",
        f"- platform: `{summary['runtime']['platform']}`",
        f"- python: `{summary['runtime']['python']}`",
        f"- total_runtime_sec: `{summary['runtime']['total_runtime_sec']}`",
        "",
        "## Vocabulary And Targets",
        "",
        f"- vocabulary: `{summary['vocabulary']}`",
        f"- target_words: `{summary['target_words']}`",
        "",
        "## Target-Word Summary",
        "",
        markdown_table(target_summary_df.drop(columns=["best_segments_csv"], errors="ignore")),
        "",
        "## Input-Mode Summary",
        "",
        markdown_table(input_mode_summary_df),
        "",
        "## Segment-Ranking Ablation Summary",
        "",
        markdown_table(segment_ablation_df) if not segment_ablation_df.empty else "No segment metrics were available.",
        "",
        "## Weight-Sensitivity Summary",
        "",
        markdown_table(weight_sensitivity_df) if not weight_sensitivity_df.empty else "No segment metrics were available.",
        "",
    ]
    if model_comparison_df is not None and not model_comparison_df.empty:
        lines.extend(["## Model-Comparison Summary", "", markdown_table(model_comparison_df), ""])
    lines.extend(
        [
            "## Figures",
            "",
            *[f"- {name}: `{figure_path}`" for name, figure_path in figure_paths.items()],
            "",
            "## Unavailable Metrics",
            "",
        ]
    )
    if unavailable_metrics:
        lines.extend([f"- {item}" for item in unavailable_metrics])
    else:
        lines.append("- None detected.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is still a controlled manuscript-oriented experiment. Results are model-, vocabulary-, objective- and configuration-dependent. "
            "Arbitrary Russian/English words require a model and vocabulary trained or configured for those words.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _run_model_comparison(
    *,
    targets: list[str],
    input_modes: list[str],
    args: argparse.Namespace,
    experiment_dir: Path,
    selected_summary: pd.DataFrame,
) -> pd.DataFrame | None:
    try:
        mock = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
        mock.load_model()
    except Exception:
        return None
    mock_vocab = set(mock.get_vocabulary())
    comparable_targets = [target for target in targets if target in mock_vocab]
    if not comparable_targets:
        return None
    runs_root = experiment_dir / "model_comparison_runs"
    rows: list[dict[str, Any]] = []
    comparison_args = copy.copy(args)
    comparison_args.repeats = 1
    for target_index, target in enumerate(comparable_targets):
        for mode_index, input_mode in enumerate(input_modes):
            seed = int(args.seed_start + 1_000_000 + target_index * len(input_modes) + mode_index)
            run_dir = runs_root / f"mock_{target_index:02d}_{mode_index:02d}_{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(parents=True, exist_ok=True)
            rows.append(_run_one(adapter=mock, target_word=target, input_mode=input_mode, seed=seed, args=comparison_args, run_dir=run_dir))
    mock_runs = pd.DataFrame(rows)
    mock_summary = _summarize_targets(mock_runs, _target_aliases(mock.get_vocabulary()))
    selected = selected_summary[
        [
            "target_word",
            "number_of_runs",
            "completed_runs",
            "failed_runs",
            "success_rate",
            "mean_final_score",
            "std_final_score",
            "best_final_score",
            "mean_score_gain",
        ]
    ].copy()
    selected.insert(0, "model_name", "selected_model")
    mock_out = mock_summary[
        [
            "target_word",
            "number_of_runs",
            "completed_runs",
            "failed_runs",
            "success_rate",
            "mean_final_score",
            "std_final_score",
            "best_final_score",
            "mean_score_gain",
        ]
    ].copy()
    mock_out.insert(0, "model_name", mock.model_name)
    return pd.concat([selected, mock_out], ignore_index=True)


def run_extended_experiment(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.quick:
        args.steps = min(args.steps, 8)
        args.repeats = min(args.repeats, 1)
    model_path = Path(args.model)
    adapter = _load_adapter(model_path)
    vocabulary = adapter.get_vocabulary()
    aliases = _target_aliases(vocabulary)
    targets = _resolve_targets(args.targets, vocabulary)

    experiment_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    experiment_dir = Path(args.output_dir) / experiment_id
    runs_root = experiment_dir / "runs"
    figures_dir = experiment_dir / "figures"
    assets_dir = experiment_dir / "assets"
    runs_root.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets):
        for mode_index, input_mode in enumerate(args.input_modes):
            for repeat_index in range(args.repeats):
                seed = int(args.seed_start + target_index * len(args.input_modes) * args.repeats + mode_index * args.repeats + repeat_index)
                safe_target = _display_word(target, aliases).replace("/", "_").replace("\\", "_").replace(" ", "_")
                run_dir = runs_root / f"{safe_target}_{input_mode}_{seed}_{uuid.uuid4().hex[:8]}"
                run_dir.mkdir(parents=True, exist_ok=True)
                print(f"[RUN] target={_display_word(target, aliases)} actual={target} mode={input_mode} seed={seed}", flush=True)
                rows.append(_run_one(adapter=adapter, target_word=target, input_mode=input_mode, seed=seed, args=args, run_dir=run_dir))

    runs_df = pd.DataFrame(rows)
    target_summary_df = _summarize_targets(runs_df, aliases)
    input_mode_summary_df = _summarize_input_modes(runs_df)
    segment_ablation_df = _segment_ablation(target_summary_df)
    weight_sensitivity_df = _weight_sensitivity(target_summary_df)
    best_audio_paths = _copy_best_audio(target_summary_df, assets_dir)
    model_comparison_df = _run_model_comparison(
        targets=targets,
        input_modes=args.input_modes,
        args=args,
        experiment_dir=experiment_dir,
        selected_summary=target_summary_df,
    )

    figure_paths = _write_figures(
        figures_dir,
        target_summary_df,
        input_mode_summary_df,
        segment_ablation_df,
        weight_sensitivity_df,
        model_comparison_df,
    )
    total_runtime = time.perf_counter() - started
    valid = runs_df[runs_df["error"].fillna("") == ""]
    unavailable_metrics: list[str] = []
    if segment_ablation_df[["top_segment_start", "top_segment_score"]].isna().any().any():
        unavailable_metrics.append("Some segment ablation values are unavailable because a best run had no exported segment metrics.")
    if weight_sensitivity_df[["top_segment_start", "top_segment_score"]].isna().any().any():
        unavailable_metrics.append("Some weight-sensitivity values are unavailable because a best run had no exported segment metrics.")
    if model_comparison_df is None:
        unavailable_metrics.append("Model comparison was unavailable for the selected adapter/target vocabulary.")

    summary = {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "configuration": {
            "model": str(model_path),
            "method": args.method,
            "objective": args.objective,
            "steps": args.steps,
            "repeats": args.repeats,
            "input_modes": args.input_modes,
            "seed_start": args.seed_start,
            "learning_rate": args.learning_rate,
            "max_delta": args.max_delta,
            "l2_weight": args.l2_weight,
            "tv_weight": args.tv_weight,
            "goal_score": args.goal_score,
            "audio": dataclass_to_json_dict(DEFAULT_AUDIO_CONFIG),
            "analysis": dataclass_to_json_dict(DEFAULT_ANALYSIS_CONFIG),
        },
        "vocabulary": vocabulary,
        "target_words": targets,
        "target_words_display": [_display_word(target, aliases) for target in targets],
        "runs": {
            "total_runs": int(len(runs_df)),
            "completed_runs": int(len(valid)),
            "failed_runs": int(len(runs_df) - len(valid)),
            "success_rate": float(valid["success"].mean()) if not valid.empty else 0.0,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "total_runtime_sec": _safe_float(total_runtime),
        },
        "exports": {
            "extended_experiment_summary_json": str(experiment_dir / "extended_experiment_summary.json"),
            "extended_experiment_summary_csv": str(experiment_dir / "extended_experiment_summary.csv"),
            "target_word_summary_csv": str(experiment_dir / "target_word_summary.csv"),
            "input_mode_summary_csv": str(experiment_dir / "input_mode_summary.csv"),
            "segment_ablation_summary_csv": str(experiment_dir / "segment_ablation_summary.csv"),
            "weight_sensitivity_summary_csv": str(experiment_dir / "weight_sensitivity_summary.csv"),
            "model_comparison_summary_csv": str(experiment_dir / "model_comparison_summary.csv") if model_comparison_df is not None else None,
            "extended_experiment_report_md": str(experiment_dir / "extended_experiment_report.md"),
            "figures": figure_paths,
            "best_audio_by_target": best_audio_paths,
        },
        "target_word_summary": target_summary_df.drop(columns=["best_segments_csv"], errors="ignore").to_dict(orient="records"),
        "input_mode_summary": input_mode_summary_df.to_dict(orient="records"),
        "segment_ablation_summary": segment_ablation_df.to_dict(orient="records"),
        "weight_sensitivity_summary": weight_sensitivity_df.to_dict(orient="records"),
        "model_comparison_summary": None if model_comparison_df is None else model_comparison_df.to_dict(orient="records"),
        "unavailable_metrics": unavailable_metrics,
        "limitations": [
            "This is still a controlled manuscript-oriented experiment.",
            "Results are model-, vocabulary-, objective- and configuration-dependent.",
            "Arbitrary Russian/English words require a model and vocabulary trained or configured for those words.",
        ],
    }

    runs_df.to_csv(experiment_dir / "extended_experiment_summary.csv", index=False)
    target_summary_df.drop(columns=["best_segments_csv"], errors="ignore").to_csv(experiment_dir / "target_word_summary.csv", index=False)
    input_mode_summary_df.to_csv(experiment_dir / "input_mode_summary.csv", index=False)
    segment_ablation_df.to_csv(experiment_dir / "segment_ablation_summary.csv", index=False)
    weight_sensitivity_df.to_csv(experiment_dir / "weight_sensitivity_summary.csv", index=False)
    if model_comparison_df is not None:
        model_comparison_df.to_csv(experiment_dir / "model_comparison_summary.csv", index=False)
    save_json(experiment_dir / "extended_experiment_summary.json", summary)
    _write_report(
        experiment_dir / "extended_experiment_report.md",
        summary=summary,
        target_summary_df=target_summary_df,
        input_mode_summary_df=input_mode_summary_df,
        segment_ablation_df=segment_ablation_df,
        weight_sensitivity_df=weight_sensitivity_df,
        model_comparison_df=model_comparison_df,
        figure_paths=figure_paths,
        unavailable_metrics=unavailable_metrics,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the extended manuscript reproducibility experiment for GRADV.")
    parser.add_argument("--targets", nargs="+", default=None, help="Target words. Defaults to all words in the selected vocabulary.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--input-modes", nargs="+", default=["maximize_from_noise", "maximize_from_silence"])
    parser.add_argument("--seed-start", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=BASELINE_MODEL_PATH)
    parser.add_argument("--method", default="gradient_ascent")
    parser.add_argument("--objective", default="logit")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-delta", type=float, default=0.20)
    parser.add_argument("--l2-weight", type=float, default=0.002)
    parser.add_argument("--tv-weight", type=float, default=0.001)
    parser.add_argument("--goal-score", type=float, default=0.85)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--quick", action="store_true", help="Run a short validation version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_extended_experiment(args)
    exports = summary["exports"]
    print("[OK] manuscript extended experiment completed")
    print(f"OUTPUT_DIR={summary['experiment_dir']}")
    print(f"RUNS_COMPLETED={summary['runs']['completed_runs']}")
    print(f"RUNS_FAILED={summary['runs']['failed_runs']}")
    print(f"REPORT={exports['extended_experiment_report_md']}")
    print(f"SUMMARY_CSV={exports['extended_experiment_summary_csv']}")
    print("FIGURES=" + json.dumps(exports["figures"], ensure_ascii=False))
    if summary["unavailable_metrics"]:
        print("WARNINGS=" + json.dumps(summary["unavailable_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
