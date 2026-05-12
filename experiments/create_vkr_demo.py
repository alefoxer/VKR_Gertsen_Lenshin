from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.generic_torch_adapter import GenericTorchAdapter
from adapters.mock_russian_kws_adapter import MockRussianKWSAdapterA
from analysis.pipeline import run_full_pipeline
from experiments.class_image_study import run_class_image_study
from experiments.model_comparison import run_model_comparison
from experiments.report_export import export_study_report
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig
from utils.export import save_json, segments_to_dataframe
from utils.russian_targets import RUSSIAN_TARGETS


BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_baseline.pt"
BASELINE_VOCAB_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_vocab.txt"


def _assert_baseline_exists() -> None:
    missing = [path for path in [BASELINE_MODEL_PATH, BASELINE_VOCAB_PATH] if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "gradv_ru_kws_baseline artifacts are required for the VKR demo.\n"
            f"Missing:\n{formatted}\n"
            "Run: python experiments\\train_gradv_ru_kws_baseline.py"
        )


def _copy_file(src: str | Path | None, dst_dir: Path) -> str | None:
    if not src:
        return None
    source = Path(src)
    if not source.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / source.name
    shutil.copy2(source, destination)
    return str(destination)


def _rel(path: str | Path | None, root: Path) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _rel_map(mapping: dict[str, Any], root: Path) -> dict[str, Any]:
    return {key: _rel(value, root) if isinstance(value, (str, Path)) else value for key, value in mapping.items()}


def _save_single_run_exports(result, run_dir: Path) -> dict[str, str]:
    exact_segments_csv = run_dir / "exact_segments.csv"
    similar_segments_csv = run_dir / "similar_segments.csv"
    segments_csv = run_dir / "segments.csv"
    class_image_segments_csv = run_dir / "class_image_segments.csv"
    segments_to_dataframe(result.exact_segments).to_csv(exact_segments_csv, index=False)
    segments_to_dataframe(result.similar_segments).to_csv(similar_segments_csv, index=False)
    segments_to_dataframe(result.segments).to_csv(segments_csv, index=False)
    segments_to_dataframe(result.segments).to_csv(class_image_segments_csv, index=False)
    summary_json = save_json(run_dir / "summary.json", result.summary_dict())
    return {
        "summary_json": str(summary_json),
        "segments_csv": str(segments_csv),
        "class_image_segments_csv": str(class_image_segments_csv),
        "exact_segments_csv": str(exact_segments_csv),
        "similar_segments_csv": str(similar_segments_csv),
    }


def _write_protocol(
    path: Path,
    *,
    demo_id: str,
    target_word: str,
    attack_config: AttackConfig,
    comparison_steps: int,
    single_summary: dict[str, Any],
    study_summary: dict[str, Any],
    comparison_summary: dict[str, Any],
    export_paths: dict[str, Any],
) -> None:
    prototype = study_summary.get("prototype_analysis") or {}
    spectral = prototype.get("spectral_stability") or {}
    lines = [
        "# GRADV VKR Demo Protocol",
        "",
        "## Purpose",
        "",
        "This portable demo export shows the main result of GRADV: a class input image, "
        "meaning an input signal that a concrete speech-recognition/KWS model strongly associates "
        "with the selected word. The demo also checks repeatability across seeds/conditions and compares models.",
        "",
        "## Base Scenario",
        "",
        f"- demo_id: `{demo_id}`",
        "- model: `gradv_ru_kws_baseline`",
        f"- target word: `{target_word}`",
        "- single input mode: `maximize_from_noise`",
        f"- method: `{attack_config.method}`",
        f"- objective: `{attack_config.objective}`",
        f"- single/study steps: `{attack_config.num_steps}`",
        f"- model comparison steps: `{comparison_steps}`",
        f"- learning_rate: `{attack_config.learning_rate}`",
        f"- max_delta: `{attack_config.max_delta}`",
        f"- l2_weight: `{attack_config.l2_weight}`",
        f"- tv_weight: `{attack_config.tv_weight}`",
        f"- goal_score: `{attack_config.goal_score}`",
        "",
        "The parameters prioritize demonstration quality over the shortest execution time.",
        "",
        "## Single Run",
        "",
        f"- original_score: `{single_summary.get('original_score')}`",
        f"- final_score: `{single_summary.get('final_score')}`",
        f"- score_gain: `{single_summary.get('score_gain')}`",
        f"- final_prediction: `{single_summary.get('adversarial_prediction')}`",
        f"- goal_reached: `{single_summary.get('goal_reached')}`",
        f"- class image WAV: `{export_paths.get('single_class_image_wav')}`",
        "",
        "This run demonstrates the found class image itself. The generated signal does not have to sound like natural speech: "
        "its role is to maximize the model response for the target class.",
        "",
        "## Study: Repeatability and Variability",
        "",
        f"- total_runs: `{study_summary.get('total_runs')}`",
        f"- success_rate: `{study_summary.get('success_rate')}`",
        f"- mean_final_score: `{study_summary.get('mean_final_score')}`",
        f"- best_final_score: `{study_summary.get('best_final_score')}`",
        f"- best_input_mode: `{study_summary.get('best_input_mode')}`",
        f"- best_seed: `{study_summary.get('best_seed')}`",
        f"- mean_pairwise_similarity: `{study_summary.get('mean_pairwise_similarity')}`",
        "",
        str(study_summary.get("study_conclusion") or study_summary.get("interpretation") or ""),
        "",
        "## Prototype and Spectral Stability",
        "",
        f"- prototype available: `{prototype.get('available')}`",
        f"- successful images: `{prototype.get('num_images')}`",
        f"- prototype mean WAV: `{export_paths.get('prototype_mean_wav')}`",
        f"- waveform similarity matrix: `{export_paths.get('similarity_matrix_csv')}`",
        f"- spectral stability PNG: `{export_paths.get('spectrogram_stability_png')}`",
        f"- mean spectral similarity: `{spectral.get('mean_spectral_similarity')}`",
        f"- stable time-frequency ratio: `{spectral.get('stable_time_frequency_ratio')}`",
        "",
        "High waveform/spectral similarity means that different runs converge to related class-image patterns. "
        "Lower similarity means that the model admits several different input realizations for the same class.",
        "",
        "## Model Comparison",
        "",
        str(comparison_summary.get("conclusion") or ""),
        "",
        "## What To Show During The Defense",
        "",
        "1. Apply the `VKR baseline quality` preset and explain the selected parameters.",
        "2. Show the single found class image: audio, waveform, spectrogram, probabilities, optimization history.",
        "3. Show the study table: success rate, mean/best score, best seed, best input mode.",
        "4. Show prototype analysis: mean class image, waveform similarity, spectral stability.",
        "5. Show model comparison: different models can produce different optimization behavior.",
        "6. Use the research report as text material for the diploma chapter.",
        "",
        "## Limitations To State Honestly",
        "",
        "`gradv_ru_kws_baseline` is a compact demonstration KWS model, not an industrial ASR system. "
        "The result describes this concrete model. CTC/seq2seq ASR and tokenizer/feature-based models require proper adapters.",
        "",
        "## Portable Files",
        "",
        "All paths below are relative to this demo folder and remain valid after copying the folder to another PC.",
        "",
        f"- demo manifest: `{export_paths.get('demo_manifest_json')}`",
        f"- single summary: `{export_paths.get('single_summary_json')}`",
        f"- single class image WAV: `{export_paths.get('single_class_image_wav')}`",
        f"- study summary: `{export_paths.get('study_summary_json')}`",
        f"- study runs CSV: `{export_paths.get('study_runs_csv')}`",
        f"- condition summary CSV: `{export_paths.get('condition_summary_csv')}`",
        f"- best study WAV: `{export_paths.get('study_best_class_image_wav')}`",
        f"- prototype mean WAV: `{export_paths.get('prototype_mean_wav')}`",
        f"- model comparison summary: `{export_paths.get('model_comparison_summary_json')}`",
        f"- diploma report: `{export_paths.get('diploma_report_md')}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

def create_vkr_demo(
    *,
    output_root: Path,
    target_word: str,
    steps: int,
    repeats: int,
    comparison_repeats: int,
    comparison_steps: int | None,
    seed_start: int,
) -> dict[str, Any]:
    _assert_baseline_exists()
    if repeats < 3:
        raise ValueError("VKR demo study requires at least 3 repeats.")
    comparison_steps = int(comparison_steps or steps)

    demo_id = uuid.uuid4().hex[:12]
    demo_dir = output_root / demo_id
    single_dir = demo_dir / "single_run"
    assets_dir = demo_dir / "assets"
    demo_dir.mkdir(parents=True, exist_ok=False)
    single_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    attack_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=steps,
        learning_rate=0.03,
        max_delta=0.20,
        l2_weight=0.002,
        tv_weight=0.001,
        goal_score=0.85,
        objective="logit",
        seed=seed_start,
        explicit_attack=True,
        prototype_emphasis=1.4,
        reference_mix_min=0.8,
    )

    baseline = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    baseline.load_model(BASELINE_MODEL_PATH)
    if target_word not in baseline.get_vocabulary():
        raise ValueError(f"Target word {target_word!r} is not in baseline vocabulary: {baseline.get_vocabulary()}")

    single_result = run_full_pipeline(
        adapter=baseline,
        uploaded_audio=None,
        target_word=target_word,
        audio_config=DEFAULT_AUDIO_CONFIG,
        attack_config=attack_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        run_dir=single_dir,
    )
    single_exports = _save_single_run_exports(single_result, single_dir)

    study = run_class_image_study(
        adapter=baseline,
        target_word=target_word,
        input_modes=["maximize_from_noise", "maximize_from_silence"],
        repeats=repeats,
        attack_config_template=attack_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=demo_dir / "study",
        seed_start=seed_start + 100,
    )

    mock = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
    mock.load_model()
    comparison_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=comparison_steps,
        learning_rate=0.03,
        max_delta=0.20,
        l2_weight=0.002,
        tv_weight=0.001,
        goal_score=0.85,
        objective="logit",
        seed=seed_start + 500,
        explicit_attack=True,
        prototype_emphasis=1.4,
        reference_mix_min=0.8,
    )
    comparison = run_model_comparison(
        model_adapters=[("mock_ru_kws_a", mock), ("gradv_ru_kws_baseline", baseline)],
        target_word=target_word,
        input_modes=["maximize_from_noise"],
        repeats=comparison_repeats,
        attack_config_template=comparison_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=demo_dir / "model_comparison",
        seed_start=seed_start + 500,
    )

    report = export_study_report(
        study_summary_path=study.summary_json_path,
        output_root=demo_dir / "report",
    )

    copied_assets = {
        "single_class_image_wav": _copy_file(single_result.metadata["saved_audio"]["maximized"], assets_dir),
        "study_best_class_image_wav": _copy_file(study.best_audio_path, assets_dir),
        "prototype_mean_wav": _copy_file(study.prototype_mean_audio_path, assets_dir),
        "prototype_summary_png": _copy_file(study.prototype_summary_plot_path, assets_dir),
        "spectrogram_stability_png": _copy_file(study.spectral_stability_plot_path, assets_dir),
        "spectrogram_stability_json": _copy_file(study.spectral_stability_json_path, assets_dir),
        "similarity_matrix_csv": _copy_file(study.similarity_matrix_csv_path, assets_dir),
        "comparison_report_md": _copy_file(comparison.report_path, assets_dir),
        "study_report_md": _copy_file(study.report_path, assets_dir),
        "diploma_report_md": _copy_file(report.report_md_path, assets_dir),
    }

    single_summary = single_result.summary_dict()
    study_summary = json.loads(Path(study.summary_json_path).read_text(encoding="utf-8"))
    comparison_summary = json.loads(Path(comparison.summary_json_path).read_text(encoding="utf-8"))
    manifest = {
        "demo_id": demo_id,
        "demo_dir": str(demo_dir),
        "target_word": target_word,
        "parameters": {
            "steps": steps,
            "repeats": repeats,
            "comparison_repeats": comparison_repeats,
            "comparison_steps": comparison_steps,
            "seed_start": seed_start,
        },
        "single_run": {
            "dir": str(single_dir),
            **single_exports,
            "class_image_audio_path": single_result.metadata["saved_audio"]["maximized"],
            "final_score": single_result.final_score,
            "score_gain": single_result.score_gain,
        },
        "study": {
            "dir": str(study.study_dir),
            "summary_json": study.summary_json_path,
            "runs_csv": study.runs_csv_path,
            "condition_summary_csv": study.condition_summary_csv_path,
            "best_audio": study.best_audio_path,
            "prototype_mean_audio": study.prototype_mean_audio_path,
            "prototype_summary_plot": study.prototype_summary_plot_path,
            "mean_pairwise_similarity": study_summary.get("mean_pairwise_similarity"),
        },
        "model_comparison": {
            "dir": str(comparison.comparison_dir),
            "summary_json": comparison.summary_json_path,
            "summary_csv": comparison.summary_csv_path,
            "runs_csv": comparison.runs_csv_path,
            "report": comparison.report_path,
        },
        "report": {
            "dir": str(report.report_dir),
            "report_md": str(report.report_md_path),
            "files": report.files,
        },
        "assets": copied_assets,
    }
    relative_paths = {
        "single_summary_json": _rel(single_exports["summary_json"], demo_dir),
        "single_segments_csv": _rel(single_exports["segments_csv"], demo_dir),
        "single_class_image_wav": _rel(copied_assets["single_class_image_wav"], demo_dir),
        "study_summary_json": _rel(study.summary_json_path, demo_dir),
        "study_runs_csv": _rel(study.runs_csv_path, demo_dir),
        "condition_summary_csv": _rel(study.condition_summary_csv_path, demo_dir),
        "study_best_class_image_wav": _rel(copied_assets["study_best_class_image_wav"], demo_dir),
        "prototype_mean_wav": _rel(copied_assets["prototype_mean_wav"], demo_dir),
        "prototype_summary_png": _rel(copied_assets["prototype_summary_png"], demo_dir),
        "spectrogram_stability_png": _rel(copied_assets["spectrogram_stability_png"], demo_dir),
        "spectrogram_stability_json": _rel(copied_assets["spectrogram_stability_json"], demo_dir),
        "similarity_matrix_csv": _rel(copied_assets["similarity_matrix_csv"], demo_dir),
        "model_comparison_summary_json": _rel(comparison.summary_json_path, demo_dir),
        "model_comparison_summary_csv": _rel(comparison.summary_csv_path, demo_dir),
        "model_comparison_runs_csv": _rel(comparison.runs_csv_path, demo_dir),
        "diploma_report_md": _rel(copied_assets["diploma_report_md"], demo_dir),
    }
    protocol_path = demo_dir / "demo_protocol.md"
    relative_paths["demo_protocol_md"] = _rel(protocol_path, demo_dir)
    manifest["relative_paths"] = relative_paths
    _write_protocol(
        protocol_path,
        demo_id=demo_id,
        target_word=target_word,
        attack_config=attack_config,
        comparison_steps=comparison_steps,
        single_summary=single_summary,
        study_summary=study_summary,
        comparison_summary=comparison_summary,
        export_paths={**relative_paths, "demo_manifest_json": "demo_manifest.json"},
    )
    manifest["demo_protocol_md"] = str(protocol_path)
    manifest_path = Path(save_json(demo_dir / "demo_manifest.json", manifest))
    manifest["demo_manifest_json"] = str(manifest_path)
    manifest["relative_paths"]["demo_manifest_json"] = _rel(manifest_path, demo_dir)
    save_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible GRADV VKR demo export.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "demo_vkr")
    parser.add_argument("--target", default=RUSSIAN_TARGETS[0])
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--comparison-repeats", type=int, default=2)
    parser.add_argument("--comparison-steps", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=2400)
    args = parser.parse_args()

    manifest = create_vkr_demo(
        output_root=args.output_root,
        target_word=args.target,
        steps=args.steps,
        repeats=args.repeats,
        comparison_repeats=args.comparison_repeats,
        comparison_steps=args.comparison_steps,
        seed_start=args.seed_start,
    )
    print("[OK] VKR demo export created")
    print(f"DEMO_DIR={manifest['demo_dir']}")
    print(f"DEMO_PROTOCOL={manifest['demo_protocol_md']}")
    print(f"DEMO_MANIFEST={manifest['demo_manifest_json']}")
    print(f"FINAL_SCORE={manifest['single_run']['final_score']:.4f}")
    print(f"STUDY_BEST={manifest['study']['best_audio']}")


if __name__ == "__main__":
    main()
