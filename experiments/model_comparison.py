from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.class_image_study import run_class_image_study
from utils.config import AnalysisConfig, AttackConfig, AudioConfig, dataclass_to_json_dict
from utils.export import save_json


@dataclass
class ModelComparisonResult:
    comparison_id: str
    comparison_dir: Path
    runs_df: pd.DataFrame
    summary_df: pd.DataFrame
    summary: dict[str, Any]
    conclusion: str
    runs_csv_path: str
    summary_csv_path: str
    summary_json_path: str
    report_path: str


def _comparison_conclusion(summary_df: pd.DataFrame, target_word: str) -> str:
    valid = summary_df[summary_df["error"].fillna("") == ""].copy()
    if valid.empty:
        return "Сравнение моделей не построено: все выбранные модели завершились ошибкой."
    best = valid.sort_values(["best_final_score", "mean_final_score"], ascending=False).iloc[0]
    hardest = valid.sort_values(["best_final_score", "mean_final_score"], ascending=True).iloc[0]
    return (
        f"Для класса `{target_word}` легче всего максимизировалась модель `{best['model_name']}` "
        f"(best final score {float(best['best_final_score']):.4f}, success rate {float(best['success_rate']):.2f}). "
        f"Наиболее трудной в этой серии оказалась `{hardest['model_name']}` "
        f"(best final score {float(hardest['best_final_score']):.4f}). "
        "Это сравнение описывает поведение конкретных подключенных моделей при одинаковых параметрах GRADV."
    )


def _write_report(path: Path, summary: dict[str, Any], summary_df: pd.DataFrame) -> str:
    lines = [
        "# Model Comparison Report",
        "",
        f"- comparison_id: `{summary.get('comparison_id')}`",
        f"- target_word: `{summary.get('target_word')}`",
        f"- total_models: {summary.get('total_models')}",
        "",
        "## Вывод",
        "",
        str(summary.get("conclusion", "")),
        "",
        "## Таблица сравнения",
        "",
        "```text",
        summary_df.to_string(index=False),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_model_comparison(
    *,
    model_adapters: list[tuple[str, Any]],
    target_word: str,
    input_modes: list[str],
    repeats: int,
    attack_config_template: AttackConfig,
    analysis_config: AnalysisConfig,
    audio_config: AudioConfig,
    output_root: Path,
    uploaded_audio=None,
    seed_start: int | None = 3000,
) -> ModelComparisonResult:
    if len(model_adapters) < 2:
        raise ValueError("Для сравнения нужно выбрать минимум две модели.")
    comparison_id = uuid.uuid4().hex[:12]
    comparison_dir = output_root / comparison_id
    studies_root = comparison_dir / "studies"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    studies_root.mkdir(parents=True, exist_ok=True)

    all_run_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for model_index, (model_name, adapter) in enumerate(model_adapters):
        try:
            study = run_class_image_study(
                adapter=adapter,
                target_word=target_word,
                input_modes=input_modes,
                repeats=repeats,
                attack_config_template=attack_config_template,
                analysis_config=analysis_config,
                audio_config=audio_config,
                output_root=studies_root / model_name,
                uploaded_audio=uploaded_audio,
                seed_start=None if seed_start is None else int(seed_start + model_index * repeats * max(1, len(input_modes))),
            )
            runs = study.runs_df.copy()
            runs["comparison_model_name"] = model_name
            runs["study_dir"] = str(study.study_dir)
            all_run_rows.append(runs)
            row = {
                "model_name": model_name,
                "target_word": target_word,
                "total_runs": study.summary.get("total_runs"),
                "completed_runs": study.summary.get("completed_runs"),
                "success_rate": study.summary.get("success_rate"),
                "mean_final_score": study.summary.get("mean_final_score"),
                "best_final_score": study.summary.get("best_final_score"),
                "mean_score_gain": study.summary.get("mean_score_gain"),
                "best_input_mode": study.summary.get("best_input_mode"),
                "best_seed": study.summary.get("best_seed"),
                "best_class_image_audio_path": study.summary.get("best_class_image_audio_path"),
                "study_dir": str(study.study_dir),
                "error": "",
            }
            best_path = study.summary.get("best_class_image_audio_path")
            if best_path and Path(str(best_path)).exists():
                copied = comparison_dir / f"{model_name}_best_class_image.wav"
                shutil.copy2(str(best_path), copied)
                row["copied_best_class_image_audio_path"] = str(copied)
            summary_rows.append(row)
        except Exception as exc:
            summary_rows.append(
                {
                    "model_name": model_name,
                    "target_word": target_word,
                    "total_runs": 0,
                    "completed_runs": 0,
                    "success_rate": 0.0,
                    "mean_final_score": None,
                    "best_final_score": None,
                    "mean_score_gain": None,
                    "best_input_mode": None,
                    "best_seed": None,
                    "best_class_image_audio_path": None,
                    "study_dir": "",
                    "error": str(exc),
                }
            )

    runs_df = pd.concat(all_run_rows, ignore_index=True) if all_run_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    conclusion = _comparison_conclusion(summary_df, target_word)
    summary = {
        "comparison_id": comparison_id,
        "comparison_dir": str(comparison_dir),
        "target_word": target_word,
        "total_models": len(model_adapters),
        "parameters": {
            "input_modes": input_modes,
            "repeats": repeats,
            "audio": dataclass_to_json_dict(audio_config),
            "attack_template": dataclass_to_json_dict(attack_config_template),
            "analysis": dataclass_to_json_dict(analysis_config),
        },
        "models": summary_df.to_dict(orient="records"),
        "conclusion": conclusion,
    }

    runs_csv = comparison_dir / "model_comparison_runs.csv"
    summary_csv = comparison_dir / "model_comparison_summary.csv"
    summary_json = comparison_dir / "model_comparison_summary.json"
    report = comparison_dir / "model_comparison_report.md"
    runs_df.to_csv(runs_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    save_json(summary_json, summary)
    _write_report(report, summary, summary_df)
    return ModelComparisonResult(
        comparison_id=comparison_id,
        comparison_dir=comparison_dir,
        runs_df=runs_df,
        summary_df=summary_df,
        summary=summary,
        conclusion=conclusion,
        runs_csv_path=str(runs_csv),
        summary_csv_path=str(summary_csv),
        summary_json_path=str(summary_json),
        report_path=str(report),
    )
