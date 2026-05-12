from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.export import save_json


@dataclass
class ResearchReportResult:
    report_id: str
    report_dir: Path
    report_md_path: str
    files: list[str]


def _copy_if_exists(src: str | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    path = Path(str(src))
    if not path.exists():
        return None
    destination = dst_dir / path.name
    shutil.copy2(path, destination)
    return str(destination)


def export_study_report(*, study_summary_path: str | Path, output_root: Path) -> ResearchReportResult:
    summary_path = Path(study_summary_path)
    if not summary_path.exists():
        raise ValueError(f"Study summary not found: {summary_path}")
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    report_id = uuid.uuid4().hex[:12]
    report_dir = output_root / report_id
    assets_dir = report_dir / "assets"
    report_dir.mkdir(parents=True, exist_ok=False)
    assets_dir.mkdir(parents=True, exist_ok=True)

    exports = summary.get("exports", {}) or {}
    copied: list[str] = []
    for key in [
        "study_runs_csv",
        "condition_summary_csv",
        "study_summary_json",
        "study_report_md",
        "best_class_image_wav",
        "prototype_mean_wav",
        "prototype_summary_json",
        "prototype_summary_png",
        "similarity_matrix_csv",
        "spectrogram_stability_json",
        "spectrogram_stability_png",
    ]:
        copied_path = _copy_if_exists(exports.get(key), assets_dir)
        if copied_path:
            copied.append(copied_path)
    for path in (summary.get("best_by_condition") or {}).values():
        copied_path = _copy_if_exists(path, assets_dir)
        if copied_path:
            copied.append(copied_path)

    params = summary.get("parameters", {})
    prototype = summary.get("prototype_analysis", {})
    report_md = report_dir / "report.md"
    lines = [
        "# Отчет исследования GRADV",
        "",
        "## Цель эксперимента",
        "",
        "Выявить входной аудио-образ, который конкретная модель распознавания речи связывает с заданным классом/словом, и оценить устойчивость найденного образа при разных начальных условиях.",
        "",
        "## Параметры",
        "",
        f"- Модель: `{params.get('model_name')}`",
        f"- Целевое слово: `{params.get('target_word')}`",
        f"- Начальные условия: `{params.get('input_modes')}`",
        f"- Повторов: `{params.get('repeats')}`",
        f"- Лучший режим: `{summary.get('best_input_mode')}`",
        f"- Лучший seed: `{summary.get('best_seed')}`",
        "",
        "## Результаты",
        "",
        f"- Success rate: `{summary.get('success_rate')}`",
        f"- Mean final score: `{summary.get('mean_final_score')}`",
        f"- Best final score: `{summary.get('best_final_score')}`",
        f"- Mean score gain: `{summary.get('mean_score_gain')}`",
        f"- Mean pairwise similarity: `{summary.get('mean_pairwise_similarity')}`",
        "",
        "## Вывод",
        "",
        str(summary.get("study_conclusion") or summary.get("interpretation") or ""),
        "",
        "## Интерпретация",
        "",
        "Найденный сигнал является входным образом класса для данной модели: он повышает score выбранного слова. Он не обязан звучать как естественная речь, потому что описывает входные признаки, влияющие на итоговую оценку класса.",
        "",
        "Статистика по повторам показывает устойчивость или вариативность найденного образа. Если similarity высокая, разные запуски дают похожие сигналы. Если similarity низкая, найденные входные образы класса заметно различаются.",
        "",
        "## Ограничения",
        "",
        "`gradv_ru_kws_baseline` является компактной демонстрационной KWS-моделью для проверки pipeline, а не промышленной ASR-системой.",
        "",
        "## Файлы отчета",
        "",
    ]
    for path in copied:
        lines.append(f"- `{path}`")
    report_md.write_text("\n".join(lines), encoding="utf-8")
    manifest_path = report_dir / "report_manifest.json"
    save_json(manifest_path, {"report_id": report_id, "source_study_summary": str(summary_path), "files": copied, "report_md": str(report_md)})
    files = [str(report_md), str(manifest_path)] + copied
    return ResearchReportResult(report_id=report_id, report_dir=report_dir, report_md_path=str(report_md), files=files)
