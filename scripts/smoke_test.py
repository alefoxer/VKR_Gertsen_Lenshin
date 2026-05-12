from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.generic_torch_adapter import GenericTorchAdapter
from adapters.mock_russian_kws_adapter import MockRussianKWSAdapterA
from analysis.pipeline import run_full_pipeline
from experiments.class_image_study import run_class_image_study
from experiments.generate_demo_audio import generate_signal
from experiments.model_comparison import run_model_comparison
from experiments.report_export import export_study_report
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig
from utils.export import make_run_dir, save_json, segments_to_dataframe
from utils.russian_targets import RUSSIAN_TARGETS


BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_baseline.pt"
BASELINE_VOCAB_PATH = PROJECT_ROOT / "models" / "gradv_ru_kws_vocab.txt"


class TinyWaveformClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x[:, 0, :]
        chunks = torch.chunk(x, 8, dim=1)
        features = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)
        return self.linear(features)


def _assert_file(path: str | Path) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Expected non-empty file: {path}")


def _save_result_exports(result, run_dir: Path) -> None:
    segments_csv = run_dir / "segments.csv"
    class_image_segments_csv = run_dir / "class_image_segments.csv"
    segments_to_dataframe(result.segments).to_csv(segments_csv, index=False)
    segments_to_dataframe(result.segments).to_csv(class_image_segments_csv, index=False)
    save_json(run_dir / "summary.json", result.summary_dict())
    _assert_file(segments_csv)
    _assert_file(class_image_segments_csv)
    _assert_file(run_dir / "summary.json")
    for key, path in result.metadata["saved_audio"].items():
        _assert_file(path)


def run_mock_pipeline(output_root: Path) -> None:
    adapter = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
    adapter.load_model()
    attack_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_silence",
        num_steps=25,
        learning_rate=0.08,
        goal_score=0.90,
        objective="probability",
        seed=123,
    )
    run_dir = make_run_dir(output_root / "mock_runs")
    result = run_full_pipeline(
        adapter,
        None,
        "да",
        DEFAULT_AUDIO_CONFIG,
        attack_config,
        DEFAULT_ANALYSIS_CONFIG,
        run_dir,
    )
    if result.final_score <= result.original_score:
        raise AssertionError("Mock pipeline did not improve target score.")
    if not result.segments:
        raise AssertionError("Mock pipeline did not produce ranked segments.")
    _save_result_exports(result, run_dir)
    print(f"[OK] mock pipeline: {run_dir} final_score={result.final_score:.4f}")


def run_generic_torch_adapter(output_root: Path) -> None:
    assets_dir = Path(tempfile.mkdtemp(prefix="gradv_smoke_"))
    assets_dir.mkdir(parents=True, exist_ok=True)
    vocab = ["да", "нет", "стоп"]
    vocab_path = assets_dir / "tiny_vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")

    model = TinyWaveformClassifier(num_classes=len(vocab))
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias[:] = torch.tensor([0.2, -0.1, -0.2])
        model.linear.weight[0, 2:6] = 0.4
    model_path = assets_dir / "tiny_torch_model.pt"
    traced = torch.jit.trace(model.eval(), torch.zeros(1, DEFAULT_AUDIO_CONFIG.sample_rate))
    traced.save(str(model_path))

    adapter = GenericTorchAdapter(vocabulary_path=vocab_path, device="cpu")
    adapter.load_model(model_path)
    audio = torch.zeros(DEFAULT_AUDIO_CONFIG.sample_rate, dtype=torch.float32).numpy()
    probs = adapter.predict_proba(audio)
    grad = adapter.compute_gradient(audio, "да")
    if set(probs) != set(vocab):
        raise AssertionError("Generic adapter returned wrong vocabulary.")
    if grad.shape != audio.shape:
        raise AssertionError("Generic adapter gradient shape mismatch.")
    print(f"[OK] generic torch adapter: {model_path}")


def run_baseline_model_if_available(output_root: Path) -> None:
    if not BASELINE_MODEL_PATH.exists() or not BASELINE_VOCAB_PATH.exists():
        print(
            "[WARN] gradv_ru_kws_baseline artifacts were not found; "
            "run: python experiments\\train_gradv_ru_kws_baseline.py"
        )
        return

    adapter = GenericTorchAdapter(
        vocabulary_path=BASELINE_VOCAB_PATH,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    adapter.load_model(BASELINE_MODEL_PATH)
    target = RUSSIAN_TARGETS[0]
    audio = generate_signal(target, variant=0).astype("float32")
    probs = adapter.predict_proba(audio)
    grad = adapter.compute_gradient(audio, target)
    if set(probs) != set(RUSSIAN_TARGETS):
        raise AssertionError("Baseline adapter returned wrong vocabulary.")
    if grad.shape != audio.shape:
        raise AssertionError("Baseline adapter gradient shape mismatch.")

    attack_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=8,
        learning_rate=0.03,
        max_delta=0.20,
        goal_score=0.85,
        objective="logit",
        seed=456,
    )
    run_dir = make_run_dir(output_root / "baseline_runs")
    result = run_full_pipeline(
        adapter,
        None,
        target,
        DEFAULT_AUDIO_CONFIG,
        attack_config,
        DEFAULT_ANALYSIS_CONFIG,
        run_dir,
    )
    if not result.segments:
        raise AssertionError("Baseline pipeline did not produce ranked segments.")
    _save_result_exports(result, run_dir)
    print(
        f"[OK] gradv_ru_kws_baseline: {run_dir} "
        f"target_score={probs[target]:.4f} final_score={result.final_score:.4f}"
    )

    study_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=6,
        learning_rate=0.03,
        max_delta=0.20,
        goal_score=0.85,
        objective="logit",
        seed=700,
    )
    study = run_class_image_study(
        adapter=adapter,
        target_word=target,
        input_modes=["maximize_from_noise"],
        repeats=2,
        attack_config_template=study_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=output_root / "studies",
        seed_start=700,
    )
    _assert_file(study.runs_csv_path)
    _assert_file(study.condition_summary_csv_path)
    _assert_file(study.summary_json_path)
    _assert_file(study.report_path)
    _assert_file(study.manifest_json_path)
    _assert_file(study.best_audio_path)
    if not study.best_by_condition_paths:
        raise AssertionError("Study smoke test did not produce best-by-condition audio files.")
    for best_audio in study.best_by_condition_paths.values():
        _assert_file(best_audio)
    if study.runs_df.empty:
        raise AssertionError("Study smoke test did not produce run rows.")
    if study.condition_summary_df.empty:
        raise AssertionError("Study smoke test did not produce condition summary rows.")
    summary_payload = json.loads(Path(study.summary_json_path).read_text(encoding="utf-8"))
    if not summary_payload.get("study_conclusion"):
        raise AssertionError("Study summary does not contain study_conclusion.")
    if not summary_payload.get("condition_summary"):
        raise AssertionError("Study summary does not contain condition_summary.")
    prototype = summary_payload.get("prototype_analysis", {})
    if not prototype:
        raise AssertionError("Study summary does not contain prototype_analysis.")
    if prototype.get("available"):
        _assert_file(study.prototype_mean_audio_path)
        _assert_file(study.similarity_matrix_csv_path)
        _assert_file(study.prototype_summary_json_path)
        _assert_file(study.prototype_summary_plot_path)
        _assert_file(study.spectral_stability_json_path)
        _assert_file(study.spectral_stability_plot_path)
        if summary_payload.get("mean_pairwise_similarity") is None:
            raise AssertionError("Prototype analysis did not save mean_pairwise_similarity.")
        spectral = prototype.get("spectral_stability", {})
        if not spectral.get("available"):
            raise AssertionError("Prototype analysis did not save spectral stability.")
        if spectral.get("mean_spectral_similarity") is None:
            raise AssertionError("Spectral stability did not save mean_spectral_similarity.")
    for summary_path in study.runs_df["summary_path"].dropna().tolist():
        _assert_file(summary_path)
    print(f"[OK] class image study: {study.study_dir} best={study.summary['best_final_score']}")

    report = export_study_report(study_summary_path=study.summary_json_path, output_root=output_root / "reports")
    _assert_file(report.report_md_path)
    if not report.files:
        raise AssertionError("Report export did not collect files.")
    print(f"[OK] study report export: {report.report_dir}")

    mock_adapter = MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate)
    mock_adapter.load_model()
    comparison = run_model_comparison(
        model_adapters=[
            ("mock_ru_kws_a", mock_adapter),
            ("gradv_ru_kws_baseline", adapter),
        ],
        target_word=target,
        input_modes=["maximize_from_noise"],
        repeats=2,
        attack_config_template=study_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        audio_config=DEFAULT_AUDIO_CONFIG,
        output_root=output_root / "model_comparisons",
        seed_start=900,
    )
    _assert_file(comparison.runs_csv_path)
    _assert_file(comparison.summary_csv_path)
    _assert_file(comparison.summary_json_path)
    _assert_file(comparison.report_path)
    if comparison.summary_df.empty:
        raise AssertionError("Model comparison did not produce summary rows.")
    print(f"[OK] model comparison: {comparison.comparison_dir}")


def main() -> None:
    output_root = PROJECT_ROOT / "outputs" / "smoke_test"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_mock_pipeline(output_root)
    run_generic_torch_adapter(output_root)
    run_baseline_model_if_available(output_root)
    print("[OK] smoke test completed")


if __name__ == "__main__":
    main()
