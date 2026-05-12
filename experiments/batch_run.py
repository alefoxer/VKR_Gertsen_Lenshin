from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import argparse
from pathlib import Path

import pandas as pd

from adapters.mock_russian_commands_adapter import MockRussianKWSAdapterB
from adapters.mock_russian_kws_adapter import MockRussianKWSAdapterA
from analysis.pipeline import run_full_pipeline
from audio.io import load_audio
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AnalysisConfig, AttackConfig
from utils.export import make_run_dir, save_json, segments_to_dataframe


def get_registry():
    reg = {
        "mock_ru_kws_a": MockRussianKWSAdapterA(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate),
        "mock_ru_kws_b": MockRussianKWSAdapterB(sample_rate=DEFAULT_AUDIO_CONFIG.sample_rate),
    }
    for a in reg.values():
        a.load_model()
    return reg


def run_batch(input_dir: Path, output_dir: Path, targets: list[str], goal: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = get_registry()
    methods = ["template_projection", "gradient_ascent", "additive_perturbation", "mask_attack", "patch_insertion"]
    input_modes = ["attack_uploaded_audio", "maximize_from_silence", "maximize_from_noise"]
    rows = []
    for file_path in sorted(input_dir.iterdir()):
        if file_path.suffix.lower() not in {".wav",".mp3",".flac",".ogg"}:
            continue
        audio, _ = load_audio(file_path, DEFAULT_AUDIO_CONFIG)
        for model_name, adapter in registry.items():
            for target_word in targets:
                for input_mode in input_modes:
                    for method in methods:
                        attack_config = AttackConfig(method=method, input_mode=input_mode, goal_score=goal)
                        run_dir = make_run_dir(output_dir / "runs")
                        uploaded_audio = audio if input_mode == "attack_uploaded_audio" else None
                        result = run_full_pipeline(adapter, uploaded_audio, target_word, DEFAULT_AUDIO_CONFIG, attack_config, DEFAULT_ANALYSIS_CONFIG, run_dir)
                        seg_df = segments_to_dataframe(result.segments)
                        seg_df.to_csv(run_dir / "segments.csv", index=False)
                        save_json(run_dir / "summary.json", result.summary_dict())
                        rows.append({
                            "file": file_path.name,
                            "model": model_name,
                            "target_word": target_word,
                            "input_mode": input_mode,
                            "attack_method": method,
                            "optimization_method": method,
                            "original_score": result.original_score,
                            "baseline_score": result.original_score,
                            "final_score": result.final_score,
                            "maximized_score": result.final_score,
                            "score_gain": result.score_gain,
                            "success": result.success,
                            "goal_reached": result.goal_reached,
                            "run_dir": str(run_dir),
                        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "detailed_results.csv", index=False)
    summary = (
        df.groupby(["model","target_word","input_mode","attack_method"], dropna=False)
        .agg(
            runs=("success","size"),
            success_rate=("success","mean"),
            goal_rate=("goal_reached","mean"),
            max_final_score=("final_score","max"),
            mean_final_score=("final_score","mean"),
            mean_gain=("score_gain","mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_results.csv", index=False)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-поиск максимизирующих образов")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch_results"))
    parser.add_argument("--targets", nargs="+", default=["да"])
    parser.add_argument("--goal", type=float, default=0.99)
    args = parser.parse_args()
    d, s = run_batch(args.input_dir, args.output_dir, args.targets, args.goal)
    print(d.head())
    print(s.head())
