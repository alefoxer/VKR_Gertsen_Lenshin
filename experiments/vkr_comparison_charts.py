from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from adapters.generic_torch_adapter import GenericTorchAdapter
from analysis.pipeline import run_full_pipeline
from experiments.class_image_study import run_class_image_study
from ui.gradio_app import BASELINE_MODEL_PATH, BASELINE_VOCAB_PATH
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig
from utils.export import make_run_dir


METHODS = ["gradient_ascent", "mask_attack", "template_projection"]
INPUT_MODES = ["maximize_from_noise", "maximize_from_silence"]


def _load_baseline() -> GenericTorchAdapter:
    adapter = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    adapter.load_model(BASELINE_MODEL_PATH)
    return adapter


def _attack_config(method: str, input_mode: str, steps: int, seed: int) -> AttackConfig:
    return AttackConfig(
        method=method,
        input_mode=input_mode,
        num_steps=steps,
        learning_rate=0.03,
        max_delta=0.20,
        l2_weight=0.002,
        tv_weight=0.001,
        goal_score=0.85,
        objective="logit",
        seed=seed,
        explicit_attack=True,
        prototype_emphasis=1.4,
        reference_mix_min=0.8,
    )


def _bar_chart(
    rows: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    labels = rows[x_col].astype(str).tolist()
    values = rows[y_col].astype(float).tolist()
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#d97706", "#be123c"]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")
    fig.subplots_adjust(left=0.28, right=0.92, top=0.78, bottom=0.18)

    y_positions = list(range(len(labels)))
    colors = [palette[index % len(palette)] for index in range(len(labels))]
    bars = ax.barh(y_positions, values, color=colors, height=0.56, edgecolor="#0f172a", linewidth=0.6)
    ax.invert_yaxis()

    fig.text(0.08, 0.93, title, ha="left", va="top", fontsize=15, fontweight="bold", color="#0f172a")
    fig.text(
        0.08,
        0.875,
        "Данные получены из текущего запуска GRADV baseline",
        ha="left",
        va="top",
        fontsize=9.5,
        color="#475569",
    )

    ax.set_xlabel(ylabel, fontsize=11, labelpad=10, color="#1e293b")
    ax.set_ylabel(xlabel, fontsize=11, labelpad=14, color="#1e293b")
    ax.set_xlim(0.0, 1.08)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10.5, color="#0f172a")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.tick_params(axis="x", labelsize=10, colors="#334155")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#cbd5e1", alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")

    for bar, value in zip(bars, values):
        ax.text(
            min(float(value) + 0.018, 1.045),
            bar.get_y() + bar.get_height() / 2,
            f"{float(value):.4f}",
            ha="left",
            va="center",
            fontsize=9.5,
            color="#0f172a",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "#f1f5f9", "edgecolor": "#cbd5e1", "linewidth": 0.6},
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_method_comparison(adapter: GenericTorchAdapter, target_word: str, output_dir: Path, steps: int, seed_start: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, method in enumerate(METHODS):
        run_dir = make_run_dir(output_dir / "method_runs")
        config = _attack_config(method, "maximize_from_noise", steps, seed_start + index)
        try:
            result = run_full_pipeline(
                adapter=adapter,
                uploaded_audio=None,
                target_word=target_word,
                audio_config=DEFAULT_AUDIO_CONFIG,
                attack_config=config,
                analysis_config=DEFAULT_ANALYSIS_CONFIG,
                run_dir=run_dir,
            )
            rows.append(
                {
                    "method": method,
                    "final_probability": float(result.final_score),
                    "original_score": float(result.original_score),
                    "score_gain": float(result.score_gain),
                    "goal_reached": bool(result.goal_reached),
                    "run_dir": str(run_dir),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method": method,
                    "final_probability": float("nan"),
                    "original_score": float("nan"),
                    "score_gain": float("nan"),
                    "goal_reached": False,
                    "run_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "method_comparison.csv", index=False)
    return df


def run_condition_comparison(adapter: GenericTorchAdapter, target_word: str, output_dir: Path, steps: int, repeats: int, seed_start: int) -> pd.DataFrame:
    study = run_class_image_study(
        adapter=adapter,
        target_word=target_word,
        input_modes=INPUT_MODES,
        repeats=repeats,
        attack_config_template=_attack_config("gradient_ascent", "maximize_from_noise", steps, seed_start),
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=output_dir / "condition_study",
        seed_start=seed_start,
    )
    df = study.condition_summary_df.rename(columns={"input_mode": "initial_condition", "mean_final_score": "mean_final_score"})
    df.to_csv(output_dir / "condition_comparison.csv", index=False)
    return df


def build_charts(output_root: Path, steps: int, repeats: int, seed_start: int) -> Path:
    run_id = uuid.uuid4().hex[:12]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    adapter = _load_baseline()
    target_word = adapter.get_vocabulary()[0]

    method_df = run_method_comparison(adapter, target_word, output_dir, steps, seed_start)
    valid_methods = method_df[method_df["error"].fillna("") == ""].copy()
    _bar_chart(
        valid_methods,
        x_col="method",
        y_col="final_probability",
        title="Сравнение методов оптимизации по итоговой вероятности",
        xlabel="Метод оптимизации",
        ylabel="Средняя итоговая вероятность",
        output_path=output_dir / "method_final_probability.png",
    )

    condition_df = run_condition_comparison(adapter, target_word, output_dir, steps, repeats, seed_start + 100)
    _bar_chart(
        condition_df,
        x_col="initial_condition",
        y_col="mean_final_score",
        title="Сравнение начальных условий study-режима",
        xlabel="Начальное условие",
        ylabel="Средний итоговый score",
        output_path=output_dir / "study_initial_conditions.png",
    )

    manifest = {
        "run_id": run_id,
        "model": "gradv_ru_kws_baseline",
        "target_word": target_word,
        "steps": steps,
        "repeats": repeats,
        "seed_start": seed_start,
        "method_chart": str(output_dir / "method_final_probability.png"),
        "condition_chart": str(output_dir / "study_initial_conditions.png"),
        "method_csv": str(output_dir / "method_comparison.csv"),
        "condition_csv": str(output_dir / "condition_comparison.csv"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VKR comparison bar charts from GRADV pipeline runs.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "vkr_chart_screens")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=7300)
    args = parser.parse_args()
    output_dir = build_charts(args.output_root, args.steps, args.repeats, args.seed_start)
    print(f"Saved VKR charts to: {output_dir}")


if __name__ == "__main__":
    main()
