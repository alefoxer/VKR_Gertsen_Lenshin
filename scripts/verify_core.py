from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

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
from experiments.run_final_vkr_experiment import run_final_vkr_experiment
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig
from utils.export import make_run_dir, save_json, segments_to_dataframe
from utils.russian_targets import RUSSIAN_TARGETS


BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_baseline.pt"
BASELINE_VOCAB_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_vocab.txt"


def _assert_file(path: str | Path | None, label: str = "file") -> Path:
    if not path:
        raise AssertionError(f"Expected {label}, got empty path")
    file_path = Path(path)
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty {label}: {file_path}")
    return file_path


def _save_basic_exports(result, run_dir: Path) -> None:
    segments_csv = run_dir / "segments.csv"
    class_image_segments_csv = run_dir / "class_image_segments.csv"
    segments_to_dataframe(result.segments).to_csv(segments_csv, index=False)
    segments_to_dataframe(result.segments).to_csv(class_image_segments_csv, index=False)
    save_json(run_dir / "summary.json", result.summary_dict())
    _assert_file(segments_csv, "segments.csv")
    _assert_file(class_image_segments_csv, "class_image_segments.csv")
    _assert_file(run_dir / "summary.json", "summary.json")
    for audio_path in result.metadata.get("saved_audio", {}).values():
        _assert_file(audio_path, "pipeline WAV")


def _baseline_adapter() -> GenericTorchAdapter:
    if not BASELINE_MODEL_PATH.exists() or not BASELINE_VOCAB_PATH.exists():
        raise FileNotFoundError(
            "Baseline artifacts are missing. Run: python experiments\\train_gradv_ru_kws_baseline.py"
        )
    adapter = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    adapter.load_model(BASELINE_MODEL_PATH)
    return adapter


def verify_core() -> None:
    output_root = PROJECT_ROOT / "outputs" / "verify_core"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    target = RUSSIAN_TARGETS[0]
    attack = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=8,
        learning_rate=0.03,
        max_delta=0.20,
        l2_weight=0.002,
        tv_weight=0.001,
        goal_score=0.85,
        objective="logit",
        seed=1100,
        explicit_attack=True,
        prototype_emphasis=1.4,
        reference_mix_min=0.8,
    )

    mock = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
    mock.load_model()
    mock_run_dir = make_run_dir(output_root / "mock_single")
    mock_result = run_full_pipeline(
        mock,
        None,
        target,
        DEFAULT_AUDIO_CONFIG,
        attack,
        DEFAULT_ANALYSIS_CONFIG,
        mock_run_dir,
    )
    _save_basic_exports(mock_result, mock_run_dir)
    print(f"[OK] mock single pipeline: {mock_run_dir}")

    baseline = _baseline_adapter()
    silent_audio = np.zeros(DEFAULT_AUDIO_CONFIG.sample_rate, dtype=np.float32)
    probs = baseline.predict_proba(silent_audio)
    grad = baseline.compute_gradient(silent_audio, target)
    if target not in probs:
        raise AssertionError("Baseline predict_proba did not return the target class.")
    if len(grad) != DEFAULT_AUDIO_CONFIG.sample_rate:
        raise AssertionError("Baseline gradient length mismatch.")
    print("[OK] baseline predict_proba and compute_gradient")

    baseline_run_dir = make_run_dir(output_root / "baseline_single")
    baseline_result = run_full_pipeline(
        baseline,
        None,
        target,
        DEFAULT_AUDIO_CONFIG,
        attack,
        DEFAULT_ANALYSIS_CONFIG,
        baseline_run_dir,
    )
    _save_basic_exports(baseline_result, baseline_run_dir)
    print(f"[OK] baseline single pipeline: {baseline_run_dir}")

    study = run_class_image_study(
        adapter=baseline,
        target_word=target,
        input_modes=["maximize_from_noise", "maximize_from_silence"],
        repeats=2,
        attack_config_template=attack,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=output_root / "studies",
        seed_start=1200,
    )
    for path, label in [
        (study.runs_csv_path, "study_runs.csv"),
        (study.condition_summary_csv_path, "condition_summary.csv"),
        (study.summary_json_path, "study_summary.json"),
        (study.report_path, "study_report.md"),
        (study.manifest_json_path, "study_manifest.json"),
        (study.best_audio_path, "best_class_image.wav"),
        (study.prototype_mean_audio_path, "prototype_mean.wav"),
        (study.prototype_summary_json_path, "prototype_summary.json"),
        (study.prototype_summary_plot_path, "prototype_summary.png"),
        (study.spectral_stability_json_path, "spectrogram_stability.json"),
        (study.spectral_stability_plot_path, "spectrogram_stability.png"),
    ]:
        _assert_file(path, label)
    study_summary = json.loads(Path(study.summary_json_path).read_text(encoding="utf-8"))
    prototype = study_summary.get("prototype_analysis", {})
    if not prototype.get("spectral_stability"):
        raise AssertionError("study_summary.json does not include spectral_stability.")
    print(f"[OK] study with prototype and spectral stability: {study.study_dir}")

    comparison = run_model_comparison(
        model_adapters=[("mock_ru_kws_a", mock), ("gradv_ru_kws_baseline", baseline)],
        target_word=target,
        input_modes=["maximize_from_noise"],
        repeats=2,
        attack_config_template=attack,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=output_root / "model_comparisons",
        seed_start=1300,
    )
    for path, label in [
        (comparison.runs_csv_path, "model_comparison_runs.csv"),
        (comparison.summary_csv_path, "model_comparison_summary.csv"),
        (comparison.summary_json_path, "model_comparison_summary.json"),
        (comparison.report_path, "model_comparison_report.md"),
    ]:
        _assert_file(path, label)
    if comparison.summary_df.empty:
        raise AssertionError("Model comparison summary is empty.")
    print(f"[OK] model comparison: {comparison.comparison_dir}")

    report = export_study_report(study_summary_path=study.summary_json_path, output_root=output_root / "reports")
    _assert_file(report.report_md_path, "report.md")
    if not report.files:
        raise AssertionError("Research report export did not collect support files.")
    print(f"[OK] research report export: {report.report_dir}")

    demo = create_vkr_demo(
        output_root=output_root / "demo_vkr",
        target_word=target,
        steps=8,
        repeats=3,
        comparison_repeats=2,
        comparison_steps=8,
        seed_start=1400,
    )
    manifest_path = _assert_file(demo["demo_manifest_json"], "demo_manifest.json")
    protocol_path = _assert_file(demo["demo_protocol_md"], "demo_protocol.md")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_paths = manifest.get("relative_paths", {})
    required_relative = [
        "demo_protocol_md",
        "single_summary_json",
        "single_class_image_wav",
        "study_summary_json",
        "study_runs_csv",
        "condition_summary_csv",
        "prototype_mean_wav",
        "spectrogram_stability_png",
        "model_comparison_summary_json",
        "diploma_report_md",
    ]
    for key in required_relative:
        value = relative_paths.get(key)
        if not value or Path(value).is_absolute():
            raise AssertionError(f"Demo manifest relative_paths[{key!r}] is missing or absolute: {value!r}")
        _assert_file(manifest_path.parent / value, f"demo relative path {key}")
    protocol_text = protocol_path.read_text(encoding="utf-8")
    if "model comparison steps: `8`" not in protocol_text:
        raise AssertionError("Demo protocol does not show honest model-comparison steps.")
    print(f"[OK] portable VKR demo export: {manifest_path.parent}")

    final = run_final_vkr_experiment(
        output_root=output_root / "final_vkr_experiment",
        target_word=target,
        quick=True,
        seed_start=1600,
    )
    final_summary_path = _assert_file(final["final_experiment_summary_json"], "final_experiment_summary.json")
    _assert_file(final["final_experiment_report_md"], "final_experiment_report.md")
    for key in [
        "study_summary_json",
        "study_runs_csv",
        "condition_summary_csv",
        "prototype_mean_wav",
        "spectrogram_stability_png",
        "model_comparison_summary_json",
        "best_class_image_wav",
    ]:
        _assert_file(final["key_files"].get(key), f"final experiment {key}")
    final_payload = json.loads(final_summary_path.read_text(encoding="utf-8"))
    if not final_payload.get("limitations"):
        raise AssertionError("Final VKR experiment summary does not include limitations.")
    print(f"[OK] final VKR quick experiment: {final['experiment_dir']}")

    print("[OK] verify_core completed")


def main() -> None:
    verify_core()


if __name__ == "__main__":
    main()
