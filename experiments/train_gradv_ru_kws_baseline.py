from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapters.generic_torch_adapter import GenericTorchAdapter
from analysis.pipeline import run_full_pipeline
from experiments.generate_demo_audio import generate_signal
from utils.config import DEFAULT_ANALYSIS_CONFIG, DEFAULT_AUDIO_CONFIG, AttackConfig
from utils.export import make_run_dir, save_json, segments_to_dataframe
from utils.russian_targets import RUSSIAN_TARGETS


class GradvRuKwsBaseline(nn.Module):
    """Compact raw-waveform keyword classifier for the GRADV demo classes."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.AvgPool1d(kernel_size=8, stride=8),
            nn.Conv1d(1, 16, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=21, stride=2, padding=10),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x[:, :1, :]
        features = self.frontend(x).squeeze(-1)
        return self.classifier(features)


def _normalize_peak(waveform: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-8:
        waveform = waveform / peak
    return waveform.astype(np.float32)


def _augment_waveform(waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = waveform.astype(np.float32).copy()

    gain = float(rng.uniform(0.65, 1.20))
    x *= gain

    shift = int(rng.integers(-2400, 2401))
    if shift > 0:
        x = np.concatenate([np.zeros(shift, dtype=np.float32), x[:-shift]])
    elif shift < 0:
        shift_abs = abs(shift)
        x = np.concatenate([x[shift_abs:], np.zeros(shift_abs, dtype=np.float32)])

    noise_std = float(rng.uniform(0.0, 0.045))
    if noise_std > 0:
        x += noise_std * rng.standard_normal(x.shape[0]).astype(np.float32)

    if rng.random() < 0.45:
        kernel_size = int(rng.choice([3, 5, 7]))
        kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
        smooth = np.convolve(x, kernel, mode="same").astype(np.float32)
        x = (0.70 * x + 0.30 * smooth).astype(np.float32)

    if rng.random() < 0.35:
        x = np.tanh(float(rng.uniform(0.9, 1.8)) * x).astype(np.float32)

    silence_len = int(rng.integers(0, 1200))
    if silence_len > 0 and rng.random() < 0.5:
        x[:silence_len] = 0.0
    if silence_len > 0 and rng.random() < 0.5:
        x[-silence_len:] = 0.0

    return np.clip(_normalize_peak(x), -1.0, 1.0)


def _make_split(
    examples_per_class: int,
    seed: int,
    augment: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    x_items = []
    y_items = []
    for class_idx, word in enumerate(RUSSIAN_TARGETS):
        for item_idx in range(examples_per_class):
            base = generate_signal(word, variant=item_idx % 7)
            waveform = _augment_waveform(base, rng) if augment else _normalize_peak(base)
            x_items.append(waveform)
            y_items.append(class_idx)
    x = torch.from_numpy(np.stack(x_items).astype(np.float32))
    y = torch.tensor(y_items, dtype=torch.long)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    return x[order], y[order]


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * int(y.numel())
        correct += int((logits.argmax(dim=1) == y).sum().detach().cpu().item())
        count += int(y.numel())
    return {"loss": total_loss / max(count, 1), "accuracy": correct / max(count, 1)}


def _count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _save_torchscript(model: nn.Module, output_path: Path, sample_count: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    example = torch.zeros(1, sample_count, dtype=torch.float32)
    traced = torch.jit.trace(model_cpu, example)
    temp_dir = Path(tempfile.mkdtemp(prefix="gradv_baseline_model_"))
    try:
        temp_path = temp_dir / output_path.name
        traced.save(str(temp_path))
        shutil.copyfile(temp_path, output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _save_pipeline_exports(result, run_dir: Path) -> None:
    segments_csv = run_dir / "segments.csv"
    segments_to_dataframe(result.segments).to_csv(segments_csv, index=False)
    save_json(run_dir / "summary.json", result.summary_dict())


def _verify_artifacts(model_path: Path, vocab_path: Path, output_root: Path) -> Dict[str, object]:
    adapter = GenericTorchAdapter(
        vocabulary_path=vocab_path,
        device="cpu",
        model_name_override="gradv_ru_kws_baseline",
    )
    adapter.load_model(model_path)

    target = RUSSIAN_TARGETS[0]
    audio = generate_signal(target, variant=0).astype(np.float32)
    probs = adapter.predict_proba(audio)
    gradient = adapter.compute_gradient(audio, target)
    if set(probs.keys()) != set(RUSSIAN_TARGETS):
        raise RuntimeError("Baseline adapter returned an unexpected vocabulary.")
    if gradient.shape != audio.shape:
        raise RuntimeError(f"Gradient shape mismatch: expected {audio.shape}, got {gradient.shape}.")

    attack_config = AttackConfig(
        method="gradient_ascent",
        input_mode="maximize_from_noise",
        num_steps=12,
        learning_rate=0.03,
        max_delta=0.20,
        l2_weight=0.002,
        tv_weight=0.001,
        goal_score=0.85,
        objective="logit",
        seed=321,
    )
    run_dir = make_run_dir(output_root / "baseline_checks")
    result = run_full_pipeline(
        adapter=adapter,
        uploaded_audio=None,
        target_word=target,
        audio_config=DEFAULT_AUDIO_CONFIG,
        attack_config=attack_config,
        analysis_config=DEFAULT_ANALYSIS_CONFIG,
        run_dir=run_dir,
    )
    _save_pipeline_exports(result, run_dir)
    return {
        "adapter_prediction": adapter.predict(audio),
        "adapter_target_probability": float(probs[target]),
        "gradient_shape": list(gradient.shape),
        "pipeline_run_dir": str(run_dir),
        "pipeline_final_score": float(result.final_score),
        "pipeline_segments": int(len(result.segments)),
    }


def train(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    sample_count = int(DEFAULT_AUDIO_CONFIG.sample_rate * DEFAULT_AUDIO_CONFIG.default_duration_sec)

    train_x, train_y = _make_split(args.examples_per_class, args.seed, augment=True)
    val_x, val_y = _make_split(args.val_examples_per_class, args.seed + 10_000, augment=True)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size, shuffle=False)

    model = GradvRuKwsBaseline(num_classes=len(RUSSIAN_TARGETS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_state = None
    best_val_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, None, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch:02d} train_acc={train_metrics['accuracy']:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} val_loss={val_metrics['loss']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "gradv_ru_kws_baseline.pt"
    vocab_path = output_dir / "gradv_ru_kws_vocab.txt"
    summary_path = output_dir / "gradv_ru_kws_training_summary.json"

    vocab_path.write_text("\n".join(RUSSIAN_TARGETS) + "\n", encoding="utf-8")
    _save_torchscript(model, model_path, sample_count)

    verification = {}
    if not args.skip_pipeline_check:
        verification = _verify_artifacts(model_path, vocab_path, PROJECT_ROOT / "outputs")

    summary = {
        "name": "gradv_ru_kws_baseline",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "vocabulary": list(RUSSIAN_TARGETS),
        "sample_rate": DEFAULT_AUDIO_CONFIG.sample_rate,
        "duration_sec": DEFAULT_AUDIO_CONFIG.default_duration_sec,
        "train_examples": int(len(train_y)),
        "val_examples": int(len(val_y)),
        "examples_per_class": args.examples_per_class,
        "val_examples_per_class": args.val_examples_per_class,
        "model_class": "GradvRuKwsBaseline",
        "model_parameters": _count_parameters(model),
        "device_used_for_training": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "augmentation": {
            "gain_range": [0.65, 1.20],
            "time_shift_samples": [-2400, 2400],
            "noise_std_range": [0.0, 0.045],
            "random_smoothing": True,
            "random_tanh_distortion": True,
            "random_edge_silence_samples": [0, 1200],
        },
        "history": history,
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "best_val_accuracy": best_val_accuracy,
        "model_path": str(model_path),
        "vocabulary_path": str(vocab_path),
        "verification": verification,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] saved model: {model_path}")
    print(f"[OK] saved vocabulary: {vocab_path}")
    print(f"[OK] saved summary: {summary_path}")
    if best_val_accuracy < args.min_val_accuracy:
        raise RuntimeError(
            f"Validation accuracy is {best_val_accuracy:.3f}, below required {args.min_val_accuracy:.3f}. "
            "Increase --epochs or --examples-per-class."
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GRADV Russian KWS baseline model.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--examples-per-class", type=int, default=80)
    parser.add_argument("--val-examples-per-class", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--min-val-accuracy", type=float, default=0.85)
    parser.add_argument("--skip-pipeline-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
