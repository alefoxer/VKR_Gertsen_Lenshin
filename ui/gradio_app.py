from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from adapters.generic_torch_adapter import GenericTorchAdapter, ModelLoadError, load_vocabulary
from adapters.mock_russian_commands_adapter import MockRussianKWSAdapterB
from adapters.mock_russian_kws_adapter import MockRussianKWSAdapterA
from analysis.method_explanations import METHOD_EXPLANATIONS
from analysis.pipeline import run_full_pipeline
from audio.io import load_audio
from experiments.class_image_study import run_class_image_study
from experiments.model_comparison import run_model_comparison
from experiments.report_export import export_study_report
from ui.plots import (
    create_attack_figure,
    create_class_image_figure,
    create_model_comparison_figure,
    create_probability_figure,
    create_study_comparison_figure,
)
from utils.config import (
    DEFAULT_ANALYSIS_CONFIG,
    DEFAULT_APP_CONFIG,
    DEFAULT_AUDIO_CONFIG,
    AnalysisConfig,
    AttackConfig,
)
from utils.export import make_run_dir, save_json, segments_to_dataframe
from utils.russian_targets import RUSSIAN_TARGETS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_baseline.pt"
BASELINE_VOCAB_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_vocab.txt"
BASELINE_SUMMARY_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_training_summary.json"

INPUT_MODE_OPTIONS = {
    "Атаковать загруженное аудио": "attack_uploaded_audio",
    "Сгенерировать из тишины": "maximize_from_silence",
    "Сгенерировать из шума": "maximize_from_noise",
}

METHOD_OPTIONS = {
    "Template / pattern optimization": "template_projection",
    "Gradient ascent": "gradient_ascent",
    "Additive perturbation": "additive_perturbation",
    "Mask optimization": "mask_attack",
    "Patch / segment insertion": "patch_insertion",
}

BASELINE_RECOMMENDED = {
    "input_mode_label": "Сгенерировать из шума",
    "attack_method_label": "Gradient ascent",
    "objective": "logit",
    "num_steps": 80,
    "learning_rate": 0.03,
    "max_delta": 0.20,
    "l2_weight": 0.002,
    "tv_weight": 0.001,
    "goal_score": 0.85,
}

VKR_BASELINE_QUALITY = {
    **BASELINE_RECOMMENDED,
    "num_steps": 80,
    "study_repeats": 3,
}

MOCK_RECOMMENDED = {
    "input_mode_label": "Сгенерировать из тишины",
    "attack_method_label": "Template / pattern optimization",
    "objective": "probability",
    "num_steps": 120,
    "learning_rate": 0.08,
    "max_delta": 0.25,
    "l2_weight": 0.005,
    "tv_weight": 0.001,
    "goal_score": 0.99,
}

CUSTOM_TORCH_RECOMMENDED = {
    "input_mode_label": "Сгенерировать из шума",
    "attack_method_label": "Gradient ascent",
    "objective": "logit",
    "num_steps": 60,
    "learning_rate": 0.03,
    "max_delta": 0.20,
    "l2_weight": 0.002,
    "tv_weight": 0.001,
    "goal_score": 0.85,
}


def _metric_card(title: str, value: str, accent: str) -> str:
    return (
        "<div style='padding:18px;border-radius:8px;"
        "background:var(--block-background-fill);"
        "color:var(--body-text-color);"
        "border:1px solid var(--border-color-primary);"
        f"border-left:6px solid {accent};"
        "box-shadow:0 4px 16px rgba(0,0,0,0.10);min-height:110px;'>"
        f"<div style='font-size:15px;opacity:0.85;margin-bottom:10px;'>{title}</div>"
        f"<div style='font-size:34px;font-weight:700;line-height:1.1;'>{value}</div>"
        "</div>"
    )


def _study_summary_cards(summary: dict) -> str:
    def fmt(value, digits: int = 4) -> str:
        if value is None:
            return "н/д"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    return (
        "<div style='display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;'>"
        f"{_metric_card('Запусков', fmt(summary.get('completed_runs'), 0), '#26a69a')}"
        f"{_metric_card('Success rate', fmt(summary.get('success_rate')), '#66bb6a')}"
        f"{_metric_card('Mean final', fmt(summary.get('mean_final_score')), '#42a5f5')}"
        f"{_metric_card('Best final', fmt(summary.get('best_final_score')), '#7e57c2')}"
        f"{_metric_card('Лучший режим', fmt(summary.get('best_input_mode')), '#ffa726')}"
        f"{_metric_card('Лучший seed', fmt(summary.get('best_seed'), 0), '#ec407a')}"
        "</div>"
    )


def _study_explanation(summary: dict) -> str:
    failed = int(summary.get("failed_runs") or 0)
    warning = f"\n\n**Предупреждение:** {failed} запуск(ов) завершились ошибкой; подробности есть в `study_runs.csv`." if failed else ""
    return (
        "## Исследование образа класса\n\n"
        "Это статистический режим: программа несколько раз ищет входной образ одного класса при разных "
        "начальных условиях и seed. Так фиксируется устойчивость результата, лучший score и зависимость "
        "от инициализации.\n\n"
        f"**Лучший запуск:** `{summary.get('best_input_mode')}` при seed `{summary.get('best_seed')}`.  \n"
        f"**Лучший score:** `{summary.get('best_final_score')}`.  \n"
        f"**Экспорт:** `{summary.get('study_dir')}`.\n\n"
        "Для `gradv_ru_kws_baseline` это исследование поведения компактной KWS-модели, а не промышленной ASR-системы."
        f"{warning}"
    )


def _study_conclusion_markdown(summary: dict) -> str:
    conclusion = summary.get("study_conclusion") or summary.get("interpretation") or "Вывод по исследованию пока не построен."
    best_by_condition = summary.get("best_by_condition") or {}
    if best_by_condition:
        best_lines = "\n".join(f"- `{mode}`: `{path}`" for mode, path in best_by_condition.items())
    else:
        best_lines = "- Лучшие образы по условиям не сохранены."
    return (
        "## Вывод по исследованию\n\n"
        f"{conclusion}\n\n"
        "**Лучшие образы по условиям:**\n"
        f"{best_lines}"
    )


def _parameter_warning(model_name: str, method_label: str, objective: str, steps: int) -> str:
    warnings: list[str] = []
    if model_name == "gradv_ru_kws_baseline":
        if method_label != "Gradient ascent":
            warnings.append("Для baseline в режиме ВКР рекомендуется `Gradient ascent`.")
        if objective != "logit":
            warnings.append("Для baseline рекомендуется objective `logit`, потому что он стабильнее оптимизируется.")
        if int(steps) < 40:
            warnings.append("Слишком мало шагов может дать поверхностный образ класса; для качества лучше 60-100.")
    if not warnings:
        return ""
    return "### Предупреждение по параметрам\n\n" + "\n".join(f"- {item}" for item in warnings)


def _class_image_passport_html(result) -> str:
    mode_title = "Синтезированный образ класса" if result.input_mode != "attack_uploaded_audio" else "Модифицированный входной сигнал"
    return (
        "<div style='border:1px solid var(--border-color-primary);border-radius:8px;padding:16px;"
        "background:var(--block-background-fill);color:var(--body-text-color);margin:10px 0 16px 0;"
        "box-shadow:0 4px 16px rgba(0,0,0,0.10);'>"
        "<div style='font-size:18px;font-weight:700;margin-bottom:10px;color:var(--body-text-color);'>Паспорт выявленного образа</div>"
        "<div style='display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;'>"
        f"<div><b>Тип результата</b><br>{mode_title}</div>"
        f"<div><b>Модель</b><br><code>{result.model_name}</code></div>"
        f"<div><b>Класс / слово</b><br><code>{result.target_word}</code></div>"
        f"<div><b>Score</b><br>{result.original_score:.4f} -> {result.final_score:.4f}</div>"
        f"<div><b>Прирост</b><br>{result.score_gain:+.4f}</div>"
        f"<div><b>Предсказание</b><br><code>{result.adversarial_prediction}</code></div>"
        "</div>"
        "</div>"
    )


def _method_help(label: str) -> str:
    return METHOD_EXPLANATIONS[METHOD_OPTIONS[label]]


def _baseline_status() -> str:
    if not BASELINE_MODEL_PATH.exists() or not BASELINE_VOCAB_PATH.exists():
        return (
            "**Baseline не найдена.** Обучите ее командой "
            "`python experiments\\train_gradv_ru_kws_baseline.py`, затем перезапустите приложение."
        )
    return (
        "**Выбрана `gradv_ru_kws_baseline`.** Это локальная компактная PyTorch KWS-модель. "
        "Для нее рекомендуются: `Gradient ascent`, objective `logit`, старт из шума, 80 шагов. "
        "`Template / pattern optimization` оставлен прежде всего для mock-моделей."
    )


def _mock_status() -> str:
    return (
        "**Выбрана mock-модель.** Для быстрого демонстрационного прогона подходят текущие параметры "
        "и `Template / pattern optimization`."
    )


def _custom_status() -> str:
    return (
        "**Выбрана пользовательская PyTorch-модель.** Укажите `.pt/.pth/TorchScript` и словарь классов. "
        "Текущий универсальный адаптер рассчитан на модели, которые принимают raw waveform 16 kHz "
        "и возвращают logits/probabilities по словарю классов. Для CTC, seq2seq, tokenizer-based ASR "
        "и feature-based моделей нужен отдельный проектный адаптер. Обычно стоит начать с `Gradient ascent` и objective `logit`."
    )


def _model_status(model_name: str) -> str:
    if model_name == "gradv_ru_kws_baseline":
        return _baseline_status()
    if model_name == "custom_torch_raw":
        return _custom_status()
    return _mock_status()


def _class_image_markdown(result, input_mode_label: str, attack_method_label: str, objective: str) -> str:
    steps_run = result.metadata.get("steps_run", result.metadata.get("optimization_history", {}).get("steps_run", "н/д"))
    class_image = result.metadata.get("class_image", {})
    if result.input_mode == "attack_uploaded_audio":
        found_text = (
            "Найден измененный вариант исходного аудио, который усиливает распознавание целевого слова. "
            "Это показывает, какие изменения входного сигнала повышают score заданного класса."
        )
    else:
        found_text = (
            "Найден синтезированный входной сигнал с высоким score выбранного слова. "
            "Он не обязан звучать как естественная речь: его назначение - показать входной аудиообраз заданного класса."
        )
    baseline_note = ""
    if result.model_name == "gradv_ru_kws_baseline":
        baseline_note = (
            "\n\nДля `gradv_ru_kws_baseline` это образ поведения компактной KWS-модели, а не универсальный "
            "акустический эталон слова. Поэтому такие образы полезно сравнивать между разными моделями, словами и начальными условиями."
        )
    return (
        "## Выявленный входной образ класса\n\n"
        f"**Что найдено:** {found_text}\n\n"
        f"**Модель:** `{result.model_name}`  \n"
        f"**Целевое слово:** `{result.target_word}`  \n"
        f"**Режим поиска:** {input_mode_label}  \n"
        f"**Метод:** {attack_method_label}, objective `{objective}`  \n"
        f"**Score целевого класса:** {result.original_score:.4f} -> {result.final_score:.4f} "
        f"({result.score_gain:+.4f})  \n"
        f"**Итоговое предсказание модели:** `{result.adversarial_prediction}`  \n"
        f"**Шагов оптимизации:** {steps_run}\n\n"
        "Графики ниже показывают форму найденного входа, спектрограмму, изменение вероятностей классов "
        "и историю оптимизации. Рост вероятности показывает, что найденный сигнал содержит участки, "
        f"которые дают вклад в класс `{result.target_word}`."
        f"{baseline_note}"
    )


def _exact_segments_notice(result) -> str:
    if result.exact_segments:
        return (
            f"Найдено самостоятельных фрагментов: {len(result.exact_segments)}. "
            "Это короткие окна, которые сами по себе распознаются как целевое слово выше строгого порога."
        )
    return (
        "Строгих фрагментов, которые сами по себе распознаются как целевое слово выше порога, не найдено. "
        "Это допустимо: образ класса может быть распределен по нескольким участкам сигнала. "
        "Основную детализацию смотрите в блоке значимых фрагментов найденного образа."
    )


class AppController:
    def __init__(self) -> None:
        self.audio_config = DEFAULT_AUDIO_CONFIG
        self.analysis_config = DEFAULT_ANALYSIS_CONFIG
        self.app_config = DEFAULT_APP_CONFIG
        self.last_study_summary_path: str | None = None
        self.registry = {
            "mock_ru_kws_a": MockRussianKWSAdapterA(sample_rate=self.audio_config.sample_rate),
            "mock_ru_kws_b": MockRussianKWSAdapterB(sample_rate=self.audio_config.sample_rate),
            "custom_torch_raw": GenericTorchAdapter(vocabulary=list(RUSSIAN_TARGETS)),
        }
        if BASELINE_MODEL_PATH.exists() and BASELINE_VOCAB_PATH.exists():
            self.registry["gradv_ru_kws_baseline"] = GenericTorchAdapter(
                vocabulary_path=BASELINE_VOCAB_PATH,
                model_name_override="gradv_ru_kws_baseline",
            )
        for name, adapter in self.registry.items():
            if name == "gradv_ru_kws_baseline":
                adapter.load_model(BASELINE_MODEL_PATH)
            elif name != "custom_torch_raw":
                adapter.load_model()

    def list_models(self):
        return list(self.registry.keys())

    def get_adapter(self, model_name: str):
        return self.registry[model_name]

    def _baseline_info(self, adapter) -> dict:
        info = adapter.get_model_info()
        training_summary = {}
        if BASELINE_SUMMARY_PATH.exists():
            try:
                payload = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
                training_summary = {
                    "best_val_accuracy": payload.get("best_val_accuracy"),
                    "final_val_accuracy": payload.get("final_val_accuracy"),
                    "model_parameters": payload.get("model_parameters"),
                    "train_examples": payload.get("train_examples"),
                    "val_examples": payload.get("val_examples"),
                }
            except json.JSONDecodeError:
                training_summary = {"warning": "training summary exists but is not valid JSON"}
        info.update(
            {
                "model_name": "gradv_ru_kws_baseline",
                "description": "Локальная компактная PyTorch-модель классификации русских команд для проверки GRADV pipeline.",
                "model_path": str(BASELINE_MODEL_PATH),
                "vocabulary_path": str(BASELINE_VOCAB_PATH),
                "training_summary_path": str(BASELINE_SUMMARY_PATH),
                "input_contract": "raw waveform 16 kHz, shape (batch, time) or (batch, 1, time)",
                "output_contract": "logits for Russian keyword classes",
                "recommended_parameters": BASELINE_RECOMMENDED,
                "training_summary": training_summary,
                "classes": adapter.get_vocabulary(),
            }
        )
        return info

    def refresh_model(self, model_name: str, model_path: str, vocabulary_path: str = "", device: str = "cpu"):
        if model_name == "custom_torch_raw":
            if not model_path:
                vocab = load_vocabulary(vocabulary_path or None)
                model_info = {
                    "model_name": "custom_torch_raw",
                    "status": "Укажите путь к TorchScript/.pt/.pth модели и нажмите загрузку.",
                    "vocabulary_size": len(vocab),
                    "device": device,
                    "input_contract": "raw waveform 16 kHz -> logits/probabilities for vocabulary classes",
                    "supported_checkpoints": "TorchScript, full torch.nn.Module, or checkpoint dict with key 'model'",
                    "not_supported_by_generic_adapter": "state_dict-only checkpoints without architecture, CTC/seq2seq/tokenizer ASR, feature-based models without a custom adapter",
                }
                return vocab, gr.Dropdown(choices=vocab, value=vocab[0]), json.dumps(model_info, ensure_ascii=False, indent=2)
            try:
                adapter = GenericTorchAdapter(vocabulary_path=vocabulary_path or None, device=device)
                adapter.load_model(model_path)
            except (ModelLoadError, ValueError, RuntimeError) as exc:
                raise gr.Error(f"Не удалось загрузить PyTorch-модель: {exc}") from exc
            self.registry[model_name] = adapter
        elif model_name == "gradv_ru_kws_baseline":
            adapter = GenericTorchAdapter(
                vocabulary_path=BASELINE_VOCAB_PATH,
                device=device,
                model_name_override="gradv_ru_kws_baseline",
            )
            adapter.load_model(BASELINE_MODEL_PATH)
            self.registry[model_name] = adapter
        else:
            adapter = self.get_adapter(model_name)
            adapter.load_model(model_path or None)
        vocab = adapter.get_vocabulary()
        info = self._baseline_info(adapter) if model_name == "gradv_ru_kws_baseline" else adapter.get_model_info()
        model_info = json.dumps(info, ensure_ascii=False, indent=2)
        return vocab, gr.Dropdown(choices=vocab, value=vocab[0]), model_info

    def _load_selected_adapter(self, model_name: str, model_path: str, vocabulary_path: str, device: str):
        if model_name == "custom_torch_raw":
            if not model_path:
                raise gr.Error("Для custom_torch_raw укажите путь к .pt/.pth или TorchScript модели.")
            adapter = GenericTorchAdapter(vocabulary_path=vocabulary_path or None, device=device)
            adapter.load_model(model_path)
            self.registry[model_name] = adapter
            return adapter
        if model_name == "gradv_ru_kws_baseline":
            adapter = GenericTorchAdapter(
                vocabulary_path=BASELINE_VOCAB_PATH,
                device=device,
                model_name_override="gradv_ru_kws_baseline",
            )
            adapter.load_model(BASELINE_MODEL_PATH)
            self.registry[model_name] = adapter
            return adapter
        adapter = self.get_adapter(model_name)
        adapter.load_model(model_path or None)
        return adapter

    def run_study(
        self,
        model_name: str,
        model_path: str,
        vocabulary_path: str,
        device: str,
        audio_path: str,
        target_word: str,
        study_input_mode_labels: list[str],
        repeats: int,
        auto_seed: bool,
        seed_start: int,
        attack_method_label: str,
        objective: str,
        num_steps: int,
        learning_rate: float,
        max_delta: float,
        l2_weight: float,
        tv_weight: float,
        patch_duration_sec: float,
        patch_stride_sec: float,
        goal_score: float,
        exact_match_threshold: float,
        top_n: int,
    ):
        adapter = self._load_selected_adapter(model_name, model_path, vocabulary_path, device)
        selected_labels = study_input_mode_labels or []
        input_modes = [INPUT_MODE_OPTIONS[label] for label in selected_labels if label in INPUT_MODE_OPTIONS]
        uploaded_audio = None
        if "attack_uploaded_audio" in input_modes:
            if audio_path:
                uploaded_audio, _ = load_audio(audio_path, self.audio_config)
            else:
                input_modes = [mode for mode in input_modes if mode != "attack_uploaded_audio"]
                gr.Warning("Режим атаки загруженного аудио пропущен: файл не выбран.")
        if not input_modes:
            raise gr.Error("Выберите хотя бы одно начальное условие исследования.")
        if int(repeats) < 2:
            raise gr.Error("Для study-режима нужно минимум 2 повтора, иначе это не статистика.")

        attack_config = AttackConfig(
            method=METHOD_OPTIONS[attack_method_label],
            input_mode=input_modes[0],
            num_steps=int(num_steps),
            learning_rate=float(learning_rate),
            max_delta=float(max_delta),
            l2_weight=float(l2_weight),
            tv_weight=float(tv_weight),
            patch_duration_sec=float(patch_duration_sec),
            patch_stride_sec=float(patch_stride_sec),
            goal_score=float(goal_score),
            objective=objective,
            explicit_attack=True,
            prototype_emphasis=1.4,
            reference_mix_min=0.8,
        )
        analysis_config = AnalysisConfig(
            top_n=int(top_n),
            saliency_smooth_kernel=self.analysis_config.saliency_smooth_kernel,
            occlusion_fill_value=self.analysis_config.occlusion_fill_value,
            exact_match_threshold=float(exact_match_threshold),
            segment_overlap_threshold=self.analysis_config.segment_overlap_threshold,
            merge_gap_sec=self.analysis_config.merge_gap_sec,
        )
        result = run_class_image_study(
            adapter=adapter,
            target_word=target_word,
            input_modes=input_modes,
            repeats=int(repeats),
            attack_config_template=attack_config,
            analysis_config=analysis_config,
            audio_config=self.audio_config,
            output_root=self.app_config.outputs_dir / "studies",
            uploaded_audio=uploaded_audio,
            seed_start=None if auto_seed else int(seed_start),
        )
        files = [
            result.runs_csv_path,
            result.summary_json_path,
            result.report_path,
        ]
        if result.best_audio_path:
            files.append(result.best_audio_path)
        files.extend(result.best_by_condition_paths.values())
        files.extend([result.condition_summary_csv_path, result.manifest_json_path])
        if result.prototype_mean_audio_path:
            files.append(result.prototype_mean_audio_path)
        if result.similarity_matrix_csv_path:
            files.append(result.similarity_matrix_csv_path)
        if result.prototype_summary_json_path:
            files.append(result.prototype_summary_json_path)
        if result.prototype_summary_plot_path:
            files.append(result.prototype_summary_plot_path)
        if result.spectral_stability_json_path:
            files.append(result.spectral_stability_json_path)
        if result.spectral_stability_plot_path:
            files.append(result.spectral_stability_plot_path)
        self.last_study_summary_path = result.summary_json_path
        return (
            _study_summary_cards(result.summary),
            _study_explanation(result.summary),
            _study_conclusion_markdown(result.summary),
            result.runs_df,
            result.condition_summary_df,
            create_study_comparison_figure(result.runs_df),
            result.best_audio_path or None,
            list(result.best_by_condition_paths.values()),
            result.prototype_mean_audio_path,
            result.summary.get("mean_pairwise_similarity"),
            result.prototype_summary_plot_path,
            result.spectral_stability_plot_path,
            result.runs_csv_path,
            result.condition_summary_csv_path,
            result.summary_json_path,
            result.report_path,
            result.manifest_json_path,
            files,
            json.dumps(result.summary, ensure_ascii=False, indent=2),
        )

    def run_model_comparison_ui(
        self,
        model_names: list[str],
        model_path: str,
        vocabulary_path: str,
        device: str,
        audio_path: str,
        target_word: str,
        study_input_mode_labels: list[str],
        repeats: int,
        auto_seed: bool,
        seed_start: int,
        attack_method_label: str,
        objective: str,
        num_steps: int,
        learning_rate: float,
        max_delta: float,
        l2_weight: float,
        tv_weight: float,
        patch_duration_sec: float,
        patch_stride_sec: float,
        goal_score: float,
        exact_match_threshold: float,
        top_n: int,
    ):
        selected = model_names or []
        if len(selected) < 2:
            raise gr.Error("Выберите минимум две модели для сравнения.")
        input_modes = [INPUT_MODE_OPTIONS[label] for label in (study_input_mode_labels or []) if label in INPUT_MODE_OPTIONS]
        uploaded_audio = None
        if "attack_uploaded_audio" in input_modes:
            if audio_path:
                uploaded_audio, _ = load_audio(audio_path, self.audio_config)
            else:
                input_modes = [mode for mode in input_modes if mode != "attack_uploaded_audio"]
        if not input_modes:
            raise gr.Error("Выберите хотя бы одно начальное условие сравнения.")
        if int(repeats) < 2:
            raise gr.Error("Для сравнения моделей нужно минимум 2 повтора на модель.")

        adapters = []
        for name in selected:
            if name == "custom_torch_raw" and not model_path:
                continue
            adapters.append((name, self._load_selected_adapter(name, model_path, vocabulary_path, device)))
        if len(adapters) < 2:
            raise gr.Error("Для сравнения осталось меньше двух доступных моделей. Custom torch включается только после загрузки модели.")

        attack_config = AttackConfig(
            method=METHOD_OPTIONS[attack_method_label],
            input_mode=input_modes[0],
            num_steps=int(num_steps),
            learning_rate=float(learning_rate),
            max_delta=float(max_delta),
            l2_weight=float(l2_weight),
            tv_weight=float(tv_weight),
            patch_duration_sec=float(patch_duration_sec),
            patch_stride_sec=float(patch_stride_sec),
            goal_score=float(goal_score),
            objective=objective,
            explicit_attack=True,
            prototype_emphasis=1.4,
            reference_mix_min=0.8,
        )
        analysis_config = AnalysisConfig(
            top_n=int(top_n),
            saliency_smooth_kernel=self.analysis_config.saliency_smooth_kernel,
            occlusion_fill_value=self.analysis_config.occlusion_fill_value,
            exact_match_threshold=float(exact_match_threshold),
            segment_overlap_threshold=self.analysis_config.segment_overlap_threshold,
            merge_gap_sec=self.analysis_config.merge_gap_sec,
        )
        result = run_model_comparison(
            model_adapters=adapters,
            target_word=target_word,
            input_modes=input_modes,
            repeats=int(repeats),
            attack_config_template=attack_config,
            analysis_config=analysis_config,
            audio_config=self.audio_config,
            output_root=self.app_config.outputs_dir / "model_comparisons",
            uploaded_audio=uploaded_audio,
            seed_start=None if auto_seed else int(seed_start),
        )
        return (
            "## Сравнение моделей\n\n" + result.conclusion + f"\n\n**Экспорт:** `{result.comparison_dir}`",
            result.summary_df,
            create_model_comparison_figure(result.summary_df),
            result.runs_csv_path,
            result.summary_csv_path,
            result.summary_json_path,
            result.report_path,
            [result.runs_csv_path, result.summary_csv_path, result.summary_json_path, result.report_path],
            json.dumps(result.summary, ensure_ascii=False, indent=2),
        )

    def export_last_study_report(self, explicit_summary_path: str):
        summary_path = explicit_summary_path or self.last_study_summary_path
        if not summary_path:
            raise gr.Error("Сначала запустите study-режим или укажите путь к study_summary.json.")
        report = export_study_report(study_summary_path=summary_path, output_root=self.app_config.outputs_dir / "reports")
        return str(report.report_md_path), report.files

    def run(
        self,
        model_name: str,
        model_path: str,
        vocabulary_path: str,
        device: str,
        audio_path: str,
        target_word: str,
        input_mode_label: str,
        attack_method_label: str,
        objective: str,
        num_steps: int,
        learning_rate: float,
        max_delta: float,
        l2_weight: float,
        tv_weight: float,
        patch_duration_sec: float,
        patch_stride_sec: float,
        goal_score: float,
        exact_match_threshold: float,
        top_n: int,
    ):
        if model_name == "custom_torch_raw":
            if not model_path:
                raise gr.Error("Для custom_torch_raw укажите путь к .pt/.pth или TorchScript модели.")
            adapter = GenericTorchAdapter(vocabulary_path=vocabulary_path or None, device=device)
            adapter.load_model(model_path)
            self.registry[model_name] = adapter
        elif model_name == "gradv_ru_kws_baseline":
            adapter = GenericTorchAdapter(
                vocabulary_path=BASELINE_VOCAB_PATH,
                device=device,
                model_name_override="gradv_ru_kws_baseline",
            )
            adapter.load_model(BASELINE_MODEL_PATH)
            self.registry[model_name] = adapter
        else:
            adapter = self.get_adapter(model_name)
            adapter.load_model(model_path or None)

        input_mode = INPUT_MODE_OPTIONS[input_mode_label]
        attack_method = METHOD_OPTIONS[attack_method_label]

        uploaded_audio = None
        if input_mode == "attack_uploaded_audio":
            if not audio_path:
                raise gr.Error("Для режима атаки на загруженное аудио сначала выберите WAV или MP3 файл.")
            uploaded_audio, _ = load_audio(audio_path, self.audio_config)

        attack_config = AttackConfig(
            method=attack_method,
            input_mode=input_mode,
            num_steps=int(num_steps),
            learning_rate=float(learning_rate),
            max_delta=float(max_delta),
            l2_weight=float(l2_weight),
            tv_weight=float(tv_weight),
            patch_duration_sec=float(patch_duration_sec),
            patch_stride_sec=float(patch_stride_sec),
            goal_score=float(goal_score),
            objective=objective,
            explicit_attack=True,
            prototype_emphasis=1.4,
            reference_mix_min=0.8,
        )
        analysis_config = AnalysisConfig(
            top_n=int(top_n),
            saliency_smooth_kernel=self.analysis_config.saliency_smooth_kernel,
            occlusion_fill_value=self.analysis_config.occlusion_fill_value,
            exact_match_threshold=float(exact_match_threshold),
            segment_overlap_threshold=self.analysis_config.segment_overlap_threshold,
            merge_gap_sec=self.analysis_config.merge_gap_sec,
        )
        run_dir = make_run_dir(self.app_config.runs_dir)

        result = run_full_pipeline(
            adapter=adapter,
            uploaded_audio=uploaded_audio,
            target_word=target_word,
            audio_config=self.audio_config,
            attack_config=attack_config,
            analysis_config=analysis_config,
            run_dir=run_dir,
        )

        exact_segments_df = segments_to_dataframe(result.exact_segments)
        similar_segments_df = segments_to_dataframe(result.similar_segments)
        all_segments_df = segments_to_dataframe(result.segments)
        exact_segments_csv = run_dir / "exact_segments.csv"
        similar_segments_csv = run_dir / "similar_segments.csv"
        all_segments_csv = run_dir / "segments.csv"
        class_image_segments_csv = run_dir / "class_image_segments.csv"
        exact_segments_df.to_csv(exact_segments_csv, index=False)
        similar_segments_df.to_csv(similar_segments_csv, index=False)
        all_segments_df.to_csv(all_segments_csv, index=False)
        all_segments_df.to_csv(class_image_segments_csv, index=False)

        figure = create_attack_figure(result)
        class_image_figure = create_class_image_figure(result)
        prob_figure = create_probability_figure(result.probabilities_before, result.probabilities_after, target_word)

        before_score = f"{result.original_score:.4f}"
        after_score = f"{result.final_score:.4f}"
        gain = f"{result.score_gain:+.4f}"
        goal_status = "Цель достигнута" if result.goal_reached else "Порог еще не достигнут"
        summary_cards = (
            "<div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;'>"
            f"{_metric_card('Целевое слово', target_word, '#e0f2f1')}"
            f"{_metric_card('Вероятность до', before_score, '#fff3e0')}"
            f"{_metric_card('Вероятность после', after_score, '#dcedc8')}"
            f"{_metric_card('Прирост', gain, '#ede7f6')}"
            "</div>"
        )
        overview_md = (
            f"## Результат поиска аудио-образа\n"
            f"**Модель:** `{model_name}`\n\n"
            f"**Режим:** {input_mode_label}\n\n"
            f"**Метод:** {attack_method_label}\n\n"
            f"**Objective:** `{objective}`\n\n"
            f"**Шагов выполнено:** {result.metadata.get('steps_run', result.metadata.get('optimization_history', {}).get('steps_run', 'н/д'))}\n\n"
            f"**Статус:** {goal_status}\n\n"
            f"**Точных сегментов:** {len(result.exact_segments)}\n\n"
            f"**Похожих сегментов:** {len(result.similar_segments)}\n\n"
            f"**Предсказание модели:** до `{result.original_prediction}`, после `{result.adversarial_prediction}`\n\n"
            f"**Вывод:** {'Найден входной аудиообраз, который достигает целевого порога.' if result.goal_reached else 'Найден входной аудиообраз, который повышает вероятность целевого слова, но не достигает заданного порога.'}"
        )
        if model_name == "gradv_ru_kws_baseline":
            overview_md += (
                "\n\n**Примечание для baseline:** это реальная компактная PyTorch KWS-модель, поэтому основной "
                "исследовательский сценарий для нее — `Gradient ascent` по `logit` из шума. Exact-сегменты могут "
                "отсутствовать: это значит, что отдельное короткое окно само по себе не прошло строгий порог, но "
                "similar-сегменты все равно показывают участки, которые поддерживают распознавание целевого слова."
            )

        exact_before_files = [seg.original_audio_path for seg in result.exact_segments if seg.original_audio_path]
        exact_after_files = [seg.optimized_audio_path for seg in result.exact_segments if seg.optimized_audio_path]
        class_image_before_files = [seg.original_audio_path for seg in result.segments if seg.original_audio_path]
        class_image_after_files = [seg.optimized_audio_path for seg in result.segments if seg.optimized_audio_path]
        similar_after_files = [seg.optimized_audio_path for seg in result.similar_segments if seg.optimized_audio_path]
        summary_path = save_json(run_dir / "summary.json", result.summary_dict())
        all_files = [str(summary_path), str(all_segments_csv), str(class_image_segments_csv), str(exact_segments_csv), str(similar_segments_csv)] + class_image_before_files + class_image_after_files + [
            result.metadata["saved_audio"]["original"],
            result.metadata["saved_audio"]["maximized"],
            result.metadata["saved_audio"]["key_pattern"],
            result.metadata["saved_audio"]["delta"],
        ]
        summary_json = json.dumps(result.summary_dict(), ensure_ascii=False, indent=2)
        exact_visible = bool(result.exact_segments)
        return (
            summary_cards,
            _class_image_passport_html(result),
            _class_image_markdown(result, input_mode_label, attack_method_label, objective),
            overview_md,
            result.textual_explanation,
            result.method_explanation,
            result.metadata["saved_audio"]["key_pattern"],
            class_image_figure,
            figure,
            prob_figure,
            all_segments_df,
            _exact_segments_notice(result),
            gr.update(value=exact_segments_df, visible=exact_visible),
            similar_segments_df,
            result.metadata["saved_audio"]["original"],
            result.metadata["saved_audio"]["maximized"],
            result.metadata["saved_audio"]["delta"],
            class_image_before_files,
            class_image_after_files,
            gr.update(value=exact_before_files, visible=exact_visible),
            gr.update(value=exact_after_files, visible=exact_visible),
            similar_after_files,
            str(class_image_segments_csv),
            gr.update(value=str(exact_segments_csv), visible=exact_visible),
            str(similar_segments_csv),
            str(summary_path),
            all_files,
            summary_json,
        )


def build_app() -> gr.Blocks:
    controller = AppController()
    default_model = controller.list_models()[0]
    default_vocab = controller.get_adapter(default_model).get_vocabulary()
    default_method_label = "Template / pattern optimization"

    with gr.Blocks(title="RU Speech Pattern Maximizer") as demo:
        gr.Markdown(
            "# Поиск входных аудиообразов целевого слова\n"
            "Инструмент формирует входной сигнал с высоким score выбранного класса и показывает вклад временных участков, "
            "вероятности классов, графики сигнала и файлы экспорта."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Модель и цель")
                model_name = gr.Dropdown(choices=controller.list_models(), value=default_model, label="Модель")
                model_status = gr.Markdown(_model_status(default_model))
                model_path = gr.Textbox(label="Путь к модели .pt/.pth/TorchScript", value="", visible=False)
                vocabulary_path = gr.Textbox(label="Путь к словарю классов .txt/.json", value="", visible=False)
                device = gr.Radio(choices=["cpu", "cuda"], value="cpu", label="Устройство", visible=False)
                refresh_button = gr.Button("Загрузить / обновить модель")
                target_word = gr.Dropdown(
                    choices=default_vocab,
                    value="да" if "да" in default_vocab else default_vocab[0],
                    label="Целевое слово",
                )
                vocab_box = gr.JSON(label="Словарь модели")
                with gr.Accordion("Техническая информация о модели", open=False):
                    model_info = gr.Code(label="Информация о модели", language="json")

            with gr.Column(scale=1):
                gr.Markdown("### Режим поиска сигнала")
                input_mode = gr.Radio(
                    choices=list(INPUT_MODE_OPTIONS.keys()),
                    value="Сгенерировать из тишины",
                    label="Сценарий",
                )
                audio_input = gr.Audio(type="filepath", label="Входное аудио WAV / MP3")
                attack_method = gr.Dropdown(
                    choices=list(METHOD_OPTIONS.keys()),
                    value=default_method_label,
                    label="Метод оптимизации",
                )
                objective = gr.Dropdown(
                    choices=["probability", "logit", "cross_entropy"],
                    value="probability",
                    label="Objective (используется в Gradient ascent)",
                    visible=False,
                )
                attack_method_help = gr.Markdown(_method_help(default_method_label))
                vkr_preset_button = gr.Button("Применить пресет ВКР baseline quality")
                gr.Markdown(
                    "Пресет выставляет рекомендуемый сценарий для демонстрации ВКР: baseline, "
                    "поиск из шума, Gradient ascent, objective logit и параметры, ориентированные на качество результата."
                )
                parameter_warning = gr.Markdown(_parameter_warning(default_model, default_method_label, "probability", 120))
                run_button = gr.Button("Найти максимизирующий сигнал", variant="primary")

        with gr.Row():
            with gr.Column():
                num_steps = gr.Slider(20, 200, value=120, step=1, label="Число шагов оптимизации")
                learning_rate = gr.Slider(0.001, 0.15, value=0.08, step=0.001, label="Learning rate")
                goal_score = gr.Slider(0.50, 0.999, value=0.99, step=0.001, label="Целевая вероятность")
            with gr.Column():
                max_delta = gr.Slider(0.01, 0.35, value=0.25, step=0.005, label="Максимальное изменение сигнала")
                l2_weight = gr.Slider(0.0, 0.2, value=0.005, step=0.001, label="L2-регуляризация")
                tv_weight = gr.Slider(0.0, 0.1, value=0.001, step=0.001, label="TV-регуляризация")
            with gr.Column():
                patch_duration_sec = gr.Slider(0.05, 0.8, value=0.35, step=0.01, label="Длительность сегмента, сек")
                patch_stride_sec = gr.Slider(0.01, 0.3, value=0.05, step=0.01, label="Шаг сегмента, сек")
                exact_match_threshold = gr.Slider(0.90, 0.999, value=0.95, step=0.01, label="Порог точного сегмента")
                top_n = gr.Slider(3, 10, value=5, step=1, label="Top-N сегментов на группу")

        gr.Markdown("## Выявленный входной образ класса")
        summary_cards = gr.HTML(label="Ключевые метрики")
        class_image_passport = gr.HTML(label="Паспорт образа класса")
        class_image_description = gr.Markdown(label="Что найдено")

        with gr.Row():
            key_pattern_audio = gr.Audio(type="filepath", label="Входной образ класса")
        class_image_plot = gr.Plot(label="Образ класса: waveform, spectrogram и вероятности классов")
        with gr.Row():
            attack_plot = gr.Plot(label="Waveform, spectrogram, saliency и история оптимизации")
            prob_plot = gr.Plot(label="Вероятности классов")

        with gr.Row():
            original_audio = gr.Audio(type="filepath", label="Исходный сигнал")
            adversarial_audio = gr.Audio(type="filepath", label="Максимизирующий входной образ")
            delta_audio = gr.Audio(type="filepath", label="Разница сигналов")

        overview = gr.Markdown(label="Итог запуска")
        explanation = gr.Markdown(label="Объяснение результата")
        method_explanation = gr.Markdown(label="Как работает выбранный метод")

        gr.Markdown("## Сегментная детализация найденного образа")
        class_image_segments_table = gr.Dataframe(label="Значимые фрагменты найденного образа")
        exact_segments_notice = gr.Markdown(label="Статус самостоятельных фрагментов")
        exact_segments_table = gr.Dataframe(label="Самостоятельно распознаваемые фрагменты")
        similar_segments_table = gr.Dataframe(label="Значимые фрагменты найденного образа: supporting/core")
        with gr.Row():
            class_image_before_segments_files = gr.Files(label="Значимые фрагменты: до оптимизации")
            class_image_after_segments_files = gr.Files(label="Значимые фрагменты: после оптимизации")
        with gr.Row():
            exact_before_segments_files = gr.Files(label="Самостоятельные фрагменты: до оптимизации")
            exact_after_segments_files = gr.Files(label="Самостоятельные фрагменты: после оптимизации")
            similar_after_segments_files = gr.Files(label="Supporting/core фрагменты: после оптимизации")
        class_image_segments_csv = gr.File(label="CSV значимых фрагментов образа")
        exact_segments_csv = gr.File(label="CSV самостоятельных фрагментов")
        similar_segments_csv = gr.File(label="CSV supporting/core фрагментов")
        summary_json_file = gr.File(label="JSON summary")
        all_files = gr.Files(label="Все файлы запуска")
        with gr.Accordion("JSON summary", open=False):
            raw_summary = gr.Code(label="Raw summary.json", language="json")

        gr.Markdown("## Исследование образа класса")
        gr.Markdown(
            "Этот режим запускает поиск образа несколько раз и показывает статистику устойчивости: "
            "какие начальные условия и seed чаще дают высокий score выбранного слова."
        )
        study_default_modes = [
            label for label, value in INPUT_MODE_OPTIONS.items() if value in {"maximize_from_noise", "maximize_from_silence"}
        ]
        with gr.Row():
            with gr.Column():
                study_input_modes = gr.CheckboxGroup(
                    choices=list(INPUT_MODE_OPTIONS.keys()),
                    value=study_default_modes,
                    label="Начальные условия исследования",
                )
                study_repeats = gr.Slider(2, 10, value=3, step=1, label="Повторов на каждое условие")
            with gr.Column():
                study_auto_seed = gr.Checkbox(value=False, label="Auto-seed")
                study_seed_start = gr.Number(value=1000, precision=0, label="Начальный seed")
                study_button = gr.Button("Запустить исследование образа класса", variant="secondary")
        study_summary_cards = gr.HTML(label="Сводка исследования")
        study_explanation = gr.Markdown(label="Пояснение исследования")
        gr.Markdown("### Вывод по исследованию")
        study_conclusion = gr.Markdown(label="Вывод по исследованию")
        study_runs_table = gr.Dataframe(label="Все запуски исследования")
        study_condition_table = gr.Dataframe(label="Агрегаты по начальным условиям")
        study_plot = gr.Plot(label="Сравнение запусков")
        study_best_audio = gr.Audio(type="filepath", label="Лучший найденный образ класса")
        study_best_by_condition_files = gr.Files(label="Лучшие образы по условиям")
        gr.Markdown("### Устойчивый прототип образа класса")
        gr.Markdown(
            "Средний образ считается по успешным запускам. Высокая pairwise similarity означает, "
            "что разные seed и условия дают похожие входные образы; низкая similarity означает вариативность."
        )
        with gr.Row():
            study_prototype_audio = gr.Audio(type="filepath", label="Средний образ класса")
            study_similarity_value = gr.Number(label="Mean pairwise similarity", precision=4)
        study_prototype_plot = gr.Image(type="filepath", label="Prototype summary")
        study_spectral_stability_plot = gr.Image(type="filepath", label="Spectral stability")
        with gr.Row():
            study_runs_csv = gr.File(label="study_runs.csv")
            study_condition_csv = gr.File(label="condition_summary.csv")
            study_summary_json = gr.File(label="study_summary.json")
            study_report = gr.File(label="study_report.md")
            study_manifest = gr.File(label="study_manifest.json")
        study_files = gr.Files(label="Файлы исследования")
        with gr.Accordion("Raw study_summary.json", open=False):
            study_raw_summary = gr.Code(label="Raw study summary", language="json")

        gr.Markdown("## Отчет исследования для ВКР")
        gr.Markdown(
            "Экспортирует человекочитаемый Markdown-отчет по последнему study-запуску или по указанному `study_summary.json`. "
            "Отчет удобно переносить в текст диплома как описание эксперимента, результатов и ограничений baseline-модели."
        )
        with gr.Row():
            report_summary_path = gr.Textbox(label="study_summary.json для отчета (можно оставить пустым после study-запуска)", value="")
            report_button = gr.Button("Сформировать отчет исследования")
        with gr.Row():
            report_md = gr.File(label="report.md")
            report_files = gr.Files(label="Файлы отчета")

        gr.Markdown("## Сравнение моделей")
        gr.Markdown(
            "Этот режим запускает одинаковое исследование для нескольких моделей и показывает, "
            "где выше итоговый score и success rate для выбранного слова. Custom torch участвует только если модель реально загружена."
        )
        comparison_default_models = [
            name for name in ["mock_ru_kws_a", "gradv_ru_kws_baseline"] if name in controller.list_models()
        ]
        if len(comparison_default_models) < 2:
            comparison_default_models = controller.list_models()[:2]
        comparison_models = gr.CheckboxGroup(
            choices=controller.list_models(),
            value=comparison_default_models,
            label="Модели для сравнения",
        )
        comparison_button = gr.Button("Запустить сравнение моделей", variant="secondary")
        comparison_conclusion = gr.Markdown(label="Вывод по сравнению моделей")
        comparison_summary_table = gr.Dataframe(label="Сводка по моделям")
        comparison_plot = gr.Plot(label="Best/mean score и success rate по моделям")
        with gr.Row():
            comparison_runs_csv = gr.File(label="model_comparison_runs.csv")
            comparison_summary_csv = gr.File(label="model_comparison_summary.csv")
            comparison_summary_json = gr.File(label="model_comparison_summary.json")
            comparison_report = gr.File(label="model_comparison_report.md")
        comparison_files = gr.Files(label="Файлы сравнения моделей")
        with gr.Accordion("Raw model_comparison_summary.json", open=False):
            comparison_raw_summary = gr.Code(label="Raw comparison summary", language="json")

        demo.load(
            lambda: (
                controller.get_adapter(default_model).get_vocabulary(),
                json.dumps(controller.get_adapter(default_model).get_model_info(), ensure_ascii=False, indent=2),
            ),
            outputs=[vocab_box, model_info],
        )

        def on_refresh(model_name_value: str, model_path_value: str, vocabulary_path_value: str, device_value: str):
            return controller.refresh_model(model_name_value, model_path_value, vocabulary_path_value, device_value)

        def on_model_change(model_name_value: str, model_path_value: str, vocabulary_path_value: str, device_value: str):
            vocab, target_update, model_info_value = controller.refresh_model(
                model_name_value,
                model_path_value,
                vocabulary_path_value,
                device_value,
            )
            is_custom = model_name_value == "custom_torch_raw"
            if model_name_value == "gradv_ru_kws_baseline":
                recommended = BASELINE_RECOMMENDED
            elif model_name_value == "custom_torch_raw":
                recommended = CUSTOM_TORCH_RECOMMENDED
            else:
                recommended = MOCK_RECOMMENDED
            recommended_method = recommended["attack_method_label"]
            return (
                vocab,
                target_update,
                model_info_value,
                _model_status(model_name_value),
                gr.update(visible=is_custom),
                gr.update(visible=is_custom),
                gr.update(visible=is_custom),
                gr.update(value=recommended["input_mode_label"]),
                gr.update(value=recommended_method),
                gr.update(value=recommended["objective"], visible=METHOD_OPTIONS[recommended_method] == "gradient_ascent"),
                gr.update(value=recommended["num_steps"]),
                gr.update(value=recommended["learning_rate"]),
                gr.update(value=recommended["max_delta"]),
                gr.update(value=recommended["l2_weight"]),
                gr.update(value=recommended["tv_weight"]),
                gr.update(value=recommended["goal_score"]),
                _method_help(recommended_method),
                _parameter_warning(model_name_value, recommended_method, recommended["objective"], recommended["num_steps"]),
            )

        def on_method_change(label: str):
            is_gradient = METHOD_OPTIONS[label] == "gradient_ascent"
            return _method_help(label), gr.update(visible=is_gradient)

        def on_warning_change(model_name_value: str, method_label: str, objective_value: str, steps_value: int):
            return _parameter_warning(model_name_value, method_label, objective_value, int(steps_value))

        def apply_vkr_preset():
            vocab, target_update, model_info_value = controller.refresh_model("gradv_ru_kws_baseline", "", "", "cpu")
            target_value = "да" if "да" in vocab else vocab[0]
            return (
                gr.update(value="gradv_ru_kws_baseline"),
                vocab,
                gr.update(choices=vocab, value=target_value),
                model_info_value,
                _model_status("gradv_ru_kws_baseline"),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value="cpu"),
                gr.update(value=VKR_BASELINE_QUALITY["input_mode_label"]),
                gr.update(value=VKR_BASELINE_QUALITY["attack_method_label"]),
                gr.update(value=VKR_BASELINE_QUALITY["objective"], visible=True),
                gr.update(value=VKR_BASELINE_QUALITY["num_steps"]),
                gr.update(value=VKR_BASELINE_QUALITY["learning_rate"]),
                gr.update(value=VKR_BASELINE_QUALITY["max_delta"]),
                gr.update(value=VKR_BASELINE_QUALITY["l2_weight"]),
                gr.update(value=VKR_BASELINE_QUALITY["tv_weight"]),
                gr.update(value=VKR_BASELINE_QUALITY["goal_score"]),
                gr.update(value=VKR_BASELINE_QUALITY["study_repeats"]),
                gr.update(value=study_default_modes),
                _method_help(VKR_BASELINE_QUALITY["attack_method_label"]),
                _parameter_warning(
                    "gradv_ru_kws_baseline",
                    VKR_BASELINE_QUALITY["attack_method_label"],
                    VKR_BASELINE_QUALITY["objective"],
                    VKR_BASELINE_QUALITY["num_steps"],
                ),
            )

        refresh_button.click(on_refresh, inputs=[model_name, model_path, vocabulary_path, device], outputs=[vocab_box, target_word, model_info])
        model_name.change(
            on_model_change,
            inputs=[model_name, model_path, vocabulary_path, device],
            outputs=[
                vocab_box,
                target_word,
                model_info,
                model_status,
                model_path,
                vocabulary_path,
                device,
                input_mode,
                attack_method,
                objective,
                num_steps,
                learning_rate,
                max_delta,
                l2_weight,
                tv_weight,
                goal_score,
                attack_method_help,
                parameter_warning,
            ],
        )
        attack_method.change(on_method_change, inputs=[attack_method], outputs=[attack_method_help, objective])
        for warning_input in [model_name, attack_method, objective, num_steps]:
            warning_input.change(
                on_warning_change,
                inputs=[model_name, attack_method, objective, num_steps],
                outputs=[parameter_warning],
            )
        vkr_preset_button.click(
            apply_vkr_preset,
            inputs=[],
            outputs=[
                model_name,
                vocab_box,
                target_word,
                model_info,
                model_status,
                model_path,
                vocabulary_path,
                device,
                input_mode,
                attack_method,
                objective,
                num_steps,
                learning_rate,
                max_delta,
                l2_weight,
                tv_weight,
                goal_score,
                study_repeats,
                study_input_modes,
                attack_method_help,
                parameter_warning,
            ],
        )

        run_button.click(
            controller.run,
            inputs=[
                model_name,
                model_path,
                vocabulary_path,
                device,
                audio_input,
                target_word,
                input_mode,
                attack_method,
                objective,
                num_steps,
                learning_rate,
                max_delta,
                l2_weight,
                tv_weight,
                patch_duration_sec,
                patch_stride_sec,
                goal_score,
                exact_match_threshold,
                top_n,
            ],
            outputs=[
                summary_cards,
                class_image_passport,
                class_image_description,
                overview,
                explanation,
                method_explanation,
                key_pattern_audio,
                class_image_plot,
                attack_plot,
                prob_plot,
                class_image_segments_table,
                exact_segments_notice,
                exact_segments_table,
                similar_segments_table,
                original_audio,
                adversarial_audio,
                delta_audio,
                class_image_before_segments_files,
                class_image_after_segments_files,
                exact_before_segments_files,
                exact_after_segments_files,
                similar_after_segments_files,
                class_image_segments_csv,
                exact_segments_csv,
                similar_segments_csv,
                summary_json_file,
                all_files,
                raw_summary,
            ],
        )
        study_button.click(
            controller.run_study,
            inputs=[
                model_name,
                model_path,
                vocabulary_path,
                device,
                audio_input,
                target_word,
                study_input_modes,
                study_repeats,
                study_auto_seed,
                study_seed_start,
                attack_method,
                objective,
                num_steps,
                learning_rate,
                max_delta,
                l2_weight,
                tv_weight,
                patch_duration_sec,
                patch_stride_sec,
                goal_score,
                exact_match_threshold,
                top_n,
            ],
            outputs=[
                study_summary_cards,
                study_explanation,
                study_conclusion,
                study_runs_table,
                study_condition_table,
                study_plot,
                study_best_audio,
                study_best_by_condition_files,
                study_prototype_audio,
                study_similarity_value,
                study_prototype_plot,
                study_spectral_stability_plot,
                study_runs_csv,
                study_condition_csv,
                study_summary_json,
                study_report,
                study_manifest,
                study_files,
                study_raw_summary,
            ],
        )
        report_button.click(
            controller.export_last_study_report,
            inputs=[report_summary_path],
            outputs=[report_md, report_files],
        )
        comparison_button.click(
            controller.run_model_comparison_ui,
            inputs=[
                comparison_models,
                model_path,
                vocabulary_path,
                device,
                audio_input,
                target_word,
                study_input_modes,
                study_repeats,
                study_auto_seed,
                study_seed_start,
                attack_method,
                objective,
                num_steps,
                learning_rate,
                max_delta,
                l2_weight,
                tv_weight,
                patch_duration_sec,
                patch_stride_sec,
                goal_score,
                exact_match_threshold,
                top_n,
            ],
            outputs=[
                comparison_conclusion,
                comparison_summary_table,
                comparison_plot,
                comparison_runs_csv,
                comparison_summary_csv,
                comparison_summary_json,
                comparison_report,
                comparison_files,
                comparison_raw_summary,
            ],
        )
    return demo
