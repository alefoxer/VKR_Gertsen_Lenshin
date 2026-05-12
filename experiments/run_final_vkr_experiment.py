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
from experiments.create_vkr_demo import create_vkr_demo
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
            "Final VKR experiment requires gradv_ru_kws_baseline artifacts.\n"
            f"Missing:\n{formatted}\n"
            "Run: python experiments\\train_gradv_ru_kws_baseline.py"
        )


def _copy_if_exists(src: str | Path | None, dst_dir: Path, name: str | None = None) -> str | None:
    if not src:
        return None
    source = Path(src)
    if not source.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    destination = dst_dir / (name or source.name)
    shutil.copy2(source, destination)
    return str(destination)


def _save_single_exports(result, run_dir: Path) -> dict[str, str]:
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
        "summary_json": str(summary_json),
        "segments_csv": str(segments_csv),
        "class_image_segments_csv": str(class_image_segments_csv),
        "exact_segments_csv": str(exact_segments_csv),
        "similar_segments_csv": str(similar_segments_csv),
    }


def _build_attack_config(*, steps: int, seed: int) -> AttackConfig:
    return AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
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


def _write_final_report(path: Path, summary: dict[str, Any]) -> str:
    single = summary["single_run"]
    study = summary["study"]
    comparison = summary["model_comparison"]
    prototype = study.get("prototype_analysis", {})
    spectral = prototype.get("spectral_stability", {})
    lines = [
        "# Final VKR Experiment Report",
        "",
        "## Goal",
        "",
        "This experiment prepares a reproducible result set for the GRADV diploma work. "
        "It finds a class input image for a concrete speech/KWS model, checks repeatability across "
        "initial conditions and seeds, analyzes the stable prototype, and compares model behavior.",
        "",
        "## Model And Target",
        "",
        "- model: `gradv_ru_kws_baseline`",
        f"- target word: `{summary['target_word']}`",
        "- model type: compact demonstration KWS model, not an industrial ASR system",
        "- input contract: raw waveform 16 kHz -> logits/probabilities over vocabulary classes",
        "",
        "## Parameters",
        "",
        f"- single/study steps: `{summary['parameters']['steps']}`",
        f"- study repeats per condition: `{summary['parameters']['study_repeats']}`",
        f"- study input modes: `{summary['parameters']['study_input_modes']}`",
        f"- comparison repeats: `{summary['parameters']['comparison_repeats']}`",
        f"- comparison steps: `{summary['parameters']['comparison_steps']}`",
        f"- seed_start: `{summary['parameters']['seed_start']}`",
        "- method: `gradient_ascent`",
        "- objective: `logit`",
        "- learning_rate: `0.03`",
        "- max_delta: `0.20`",
        "- l2_weight: `0.002`",
        "- tv_weight: `0.001`",
        "- goal_score: `0.85`",
        "",
        "## Single Run Result",
        "",
        f"- original_score: `{single.get('original_score')}`",
        f"- final_score: `{single.get('final_score')}`",
        f"- score_gain: `{single.get('score_gain')}`",
        f"- final_prediction: `{single.get('final_prediction')}`",
        f"- class_image_audio: `{single.get('class_image_audio_path')}`",
        f"- summary_json: `{single.get('summary_json')}`",
        "",
        "The single run gives the main class image: a synthesized signal that maximizes the model response "
        "for the target word. It does not have to sound like natural speech, because it represents model-specific "
        "features rather than a universal acoustic standard.",
        "",
        "## Study Result",
        "",
        f"- total_runs: `{study.get('total_runs')}`",
        f"- success_rate: `{study.get('success_rate')}`",
        f"- mean_final_score: `{study.get('mean_final_score')}`",
        f"- best_final_score: `{study.get('best_final_score')}`",
        f"- best_input_mode: `{study.get('best_input_mode')}`",
        f"- best_seed: `{study.get('best_seed')}`",
        f"- best_class_image_audio: `{study.get('best_class_image_audio_path')}`",
        "",
        str(study.get("study_conclusion") or study.get("interpretation") or ""),
        "",
        "## Stable Prototype",
        "",
        f"- prototype available: `{prototype.get('available')}`",
        f"- successful images: `{prototype.get('num_images')}`",
        f"- mean_pairwise_similarity: `{study.get('mean_pairwise_similarity')}`",
        f"- prototype_mean_audio: `{study.get('prototype_mean_audio_path')}`",
        f"- spectral_stability_png: `{study.get('spectrogram_stability_plot_path')}`",
        f"- mean_spectral_similarity: `{spectral.get('mean_spectral_similarity')}`",
        f"- stable_time_frequency_ratio: `{spectral.get('stable_time_frequency_ratio')}`",
        "",
        "High waveform or spectral similarity means that different starts produce related class-image patterns. "
        "Low similarity means the model accepts several different realizations of the same class.",
        "",
        "## Model Comparison",
        "",
        f"- comparison_summary_json: `{comparison.get('summary_json')}`",
        f"- comparison_summary_csv: `{comparison.get('summary_csv')}`",
        f"- comparison_runs_csv: `{comparison.get('runs_csv')}`",
        "",
        str(comparison.get("conclusion") or ""),
        "",
        "## Limitations",
        "",
        "`gradv_ru_kws_baseline` is a compact demonstration keyword-spotting model. "
        "The result describes this model, not all Russian speech. PyTorch support is adapter-based: "
        "the generic adapter supports raw waveform classifiers with vocabulary-sized logits/probabilities. "
        "CTC, seq2seq, tokenizer-based ASR, and feature-based models require dedicated adapters.",
        "",
        "## Key Files",
        "",
    ]
    for key, value in summary.get("key_files", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_final_vkr_experiment(
    *,
    output_root: Path,
    target_word: str | None = None,
    steps: int = 100,
    study_repeats: int = 5,
    comparison_repeats: int = 3,
    comparison_steps: int = 100,
    seed_start: int = 5000,
    quick: bool = False,
) -> dict[str, Any]:
    _assert_baseline_exists()
    if quick:
        steps = 8
        study_repeats = 2
        comparison_repeats = 2
        comparison_steps = 8

    experiment_id = uuid.uuid4().hex[:12]
    experiment_dir = output_root / experiment_id
    single_dir = experiment_dir / "single_run"
    assets_dir = experiment_dir / "assets"
    experiment_dir.mkdir(parents=True, exist_ok=False)
    single_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    baseline = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    baseline.load_model(BASELINE_MODEL_PATH)
    vocab = baseline.get_vocabulary()
    target = target_word or ("да" if "да" in vocab else RUSSIAN_TARGETS[0])
    if target not in vocab:
        raise ValueError(f"Target word {target!r} is not in baseline vocabulary: {vocab}")

    attack = _build_attack_config(steps=steps, seed=seed_start)
    single_result = run_full_pipeline(
        adapter=baseline,
        uploaded_audio=None,
        target_word=target,
        audio_config=DEFAULT_AUDIO_CONFIG,
        attack_config=attack,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        run_dir=single_dir,
    )
    single_exports = _save_single_exports(single_result, single_dir)

    study = run_class_image_study(
        adapter=baseline,
        target_word=target,
        input_modes=["maximize_from_noise", "maximize_from_silence"],
        repeats=study_repeats,
        attack_config_template=attack,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=experiment_dir / "study",
        seed_start=seed_start + 100,
    )
    study_summary = json.loads(Path(study.summary_json_path).read_text(encoding="utf-8"))

    mock = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
    mock.load_model()
    comparison_attack = _build_attack_config(steps=comparison_steps, seed=seed_start + 500)
    comparison = run_model_comparison(
        model_adapters=[("mock_ru_kws_a", mock), ("gradv_ru_kws_baseline", baseline)],
        target_word=target,
        input_modes=["maximize_from_noise"],
        repeats=comparison_repeats,
        attack_config_template=comparison_attack,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=experiment_dir / "model_comparison",
        seed_start=seed_start + 500,
    )
    comparison_summary = json.loads(Path(comparison.summary_json_path).read_text(encoding="utf-8"))

    report = export_study_report(study_summary_path=study.summary_json_path, output_root=experiment_dir / "research_report")
    demo = create_vkr_demo(
        output_root=experiment_dir / "demo_vkr",
        target_word=target,
        steps=steps,
        repeats=max(3, study_repeats),
        comparison_repeats=comparison_repeats,
        comparison_steps=comparison_steps,
        seed_start=seed_start + 1000,
    )

    copied = {
        "best_class_image_wav": _copy_if_exists(study.best_audio_path, assets_dir, "best_class_image.wav"),
        "single_class_image_wav": _copy_if_exists(single_result.metadata["saved_audio"]["maximized"], assets_dir, "single_class_image.wav"),
        "prototype_mean_wav": _copy_if_exists(study.prototype_mean_audio_path, assets_dir, "prototype_mean.wav"),
        "spectrogram_stability_png": _copy_if_exists(study.spectral_stability_plot_path, assets_dir, "spectrogram_stability.png"),
        "model_comparison_summary_json": _copy_if_exists(comparison.summary_json_path, assets_dir, "model_comparison_summary.json"),
        "study_summary_json": _copy_if_exists(study.summary_json_path, assets_dir, "study_summary.json"),
        "study_runs_csv": _copy_if_exists(study.runs_csv_path, assets_dir, "study_runs.csv"),
        "condition_summary_csv": _copy_if_exists(study.condition_summary_csv_path, assets_dir, "condition_summary.csv"),
    }

    single_summary = single_result.summary_dict()
    summary: dict[str, Any] = {
        "experiment_id": experiment_id,
        "experiment_dir": str(experiment_dir),
        "quick": quick,
        "target_word": target,
        "parameters": {
            "steps": steps,
            "study_repeats": study_repeats,
            "study_input_modes": ["maximize_from_noise", "maximize_from_silence"],
            "comparison_repeats": comparison_repeats,
            "comparison_steps": comparison_steps,
            "seed_start": seed_start,
        },
        "single_run": {
            "dir": str(single_dir),
            **single_exports,
            "original_score": single_summary.get("original_score"),
            "final_score": single_summary.get("final_score"),
            "score_gain": single_summary.get("score_gain"),
            "final_prediction": single_summary.get("adversarial_prediction"),
            "class_image_audio_path": single_result.metadata["saved_audio"]["maximized"],
        },
        "study": {
            **study_summary,
            "summary_json": study.summary_json_path,
            "runs_csv": study.runs_csv_path,
            "condition_summary_csv": study.condition_summary_csv_path,
            "prototype_mean_audio_path": study.prototype_mean_audio_path,
            "spectrogram_stability_plot_path": study.spectral_stability_plot_path,
        },
        "model_comparison": {
            **comparison_summary,
            "summary_json": comparison.summary_json_path,
            "summary_csv": comparison.summary_csv_path,
            "runs_csv": comparison.runs_csv_path,
            "report": comparison.report_path,
            "conclusion": comparison.conclusion,
        },
        "research_report": {
            "report_md": report.report_md_path,
            "files": report.files,
        },
        "demo_export": {
            "demo_dir": demo["demo_dir"],
            "demo_manifest_json": demo["demo_manifest_json"],
            "demo_protocol_md": demo["demo_protocol_md"],
        },
        "key_files": copied,
        "limitations": [
            "gradv_ru_kws_baseline is a compact demonstration KWS model, not an industrial ASR.",
            "The generic PyTorch adapter supports raw waveform classifiers, not arbitrary ASR architectures without adapters.",
            "Class images describe the behavior of the selected model, not a universal acoustic standard for the word.",
        ],
    }
    report_path = experiment_dir / "final_experiment_report.md"
    summary["final_experiment_report_md"] = _write_final_report(report_path, summary)
    summary_path = Path(save_json(experiment_dir / "final_experiment_summary.json", summary))
    summary["final_experiment_summary_json"] = str(summary_path)
    save_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the final reproducible VKR experiment for GRADV.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "final_vkr_experiment")
    parser.add_argument("--target", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--study-repeats", type=int, default=5)
    parser.add_argument("--comparison-repeats", type=int, default=3)
    parser.add_argument("--comparison-steps", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--quick", action="store_true", help="Run a short verification version.")
    args = parser.parse_args()

    summary = run_final_vkr_experiment(
        output_root=args.output_root,
        target_word=args.target,
        steps=args.steps,
        study_repeats=args.study_repeats,
        comparison_repeats=args.comparison_repeats,
        comparison_steps=args.comparison_steps,
        seed_start=args.seed_start,
        quick=args.quick,
    )
    print("[OK] final VKR experiment completed")
    print(f"EXPERIMENT_DIR={summary['experiment_dir']}")
    print(f"SUMMARY={summary['final_experiment_summary_json']}")
    print(f"REPORT={summary['final_experiment_report_md']}")
    print(f"SINGLE_FINAL_SCORE={summary['single_run']['final_score']:.4f}")
    print(f"STUDY_BEST={summary['study'].get('best_final_score')}")


if __name__ == "__main__":
    main()
