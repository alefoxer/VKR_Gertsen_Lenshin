from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from adapters.base import BaseRussianSpeechAdapter
from utils.russian_targets import RUSSIAN_TARGETS


class ModelLoadError(RuntimeError):
    pass


def load_vocabulary(vocabulary_path: str | Path | None) -> List[str]:
    if not vocabulary_path:
        return list(RUSSIAN_TARGETS)
    path = Path(vocabulary_path)
    if not path.exists():
        raise ModelLoadError(f"Vocabulary file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "vocabulary" in payload:
                payload = payload["vocabulary"]
            elif "classes" in payload:
                payload = payload["classes"]
            else:
                payload = list(payload.keys())
        if not isinstance(payload, list):
            raise ModelLoadError("JSON vocabulary must be a list or contain 'vocabulary'/'classes'.")
        return [str(item) for item in payload]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class GenericTorchAdapter(BaseRussianSpeechAdapter):
    model_name = "custom_torch_raw"

    def __init__(
        self,
        vocabulary: List[str] | None = None,
        vocabulary_path: str | Path | None = None,
        device: str = "cpu",
        input_shape: str = "batch_time",
        model_name_override: str | None = None,
    ) -> None:
        if model_name_override:
            self.model_name = model_name_override
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.vocabulary = list(vocabulary) if vocabulary is not None else load_vocabulary(vocabulary_path)
        self.vocabulary_path = str(vocabulary_path) if vocabulary_path else ""
        self.input_shape = input_shape
        self.model: torch.nn.Module | torch.jit.ScriptModule | None = None
        self.loaded_path = ""

    def load_model(self, model_path: str | Path | None = None) -> None:
        if not model_path:
            raise ModelLoadError("Для custom_torch_raw укажите путь к .pt/.pth модели.")
        path = Path(model_path)
        if not path.exists():
            raise ModelLoadError(f"Model file not found: {path}")

        try:
            try:
                model = torch.jit.load(str(path), map_location=self.device)
            except Exception:
                with path.open("rb") as handle:
                    model = torch.jit.load(handle, map_location=self.device)
        except Exception as jit_error:
            try:
                checkpoint = torch.load(str(path), map_location=self.device, weights_only=False)
            except Exception as load_error:
                raise ModelLoadError(
                    "Не удалось загрузить файл как TorchScript или PyTorch checkpoint. "
                    f"TorchScript error: {jit_error}; checkpoint error: {load_error}"
                ) from load_error
            if isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
                model = checkpoint["model"]
            else:
                raise ModelLoadError(
                    "Unsupported checkpoint. Save a full torch.nn.Module, a TorchScript model, "
                    "or a dict with key 'model'. State_dict-only checkpoints need a project-specific adapter."
                )

        model.to(self.device)
        model.eval()
        self.model = model
        self.loaded_path = str(path)
        self._validate_output_size()

    def _prepare_tensor(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(audio, np.ndarray):
            x = torch.from_numpy(audio).float().to(self.device)
        else:
            x = audio.float().to(self.device)
        x = x.flatten()
        if self.input_shape == "batch_channel_time":
            return x.view(1, 1, -1)
        return x.view(1, -1)

    def _forward_logits(self, audio: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ModelLoadError("Model is not loaded.")
        x = self._prepare_tensor(audio)
        output = self.model(x)
        if isinstance(output, dict):
            output = output.get("logits", output.get("out"))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise ModelLoadError("Model forward must return logits tensor, tuple/list with logits, or dict['logits'].")
        logits = output.squeeze(0).flatten()
        if logits.numel() != len(self.vocabulary):
            raise ModelLoadError(
                f"Model returned {logits.numel()} logits, but vocabulary has {len(self.vocabulary)} classes."
            )
        return logits

    def _validate_output_size(self) -> None:
        probe = torch.zeros(16000, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            self._forward_logits(probe)

    def predict(self, audio: np.ndarray) -> str:
        probs = self.predict_proba(audio)
        return max(probs, key=probs.get)

    def predict_proba(self, audio: np.ndarray) -> Dict[str, float]:
        with torch.no_grad():
            logits = self._forward_logits(audio)
            probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        return {word: float(prob) for word, prob in zip(self.vocabulary, probs)}

    def get_vocabulary(self) -> List[str]:
        return list(self.vocabulary)

    def get_target_score(self, audio: np.ndarray, target_text: str) -> float:
        with torch.no_grad():
            score = self.target_score_tensor(torch.from_numpy(audio).float().to(self.device), target_text)
        return float(score.detach().cpu().item())

    def target_score_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = self._forward_logits(audio_tensor)
        idx = self.vocabulary.index(target_text)
        return torch.softmax(logits, dim=0)[idx]

    def target_logit_tensor(self, audio_tensor: torch.Tensor, target_text: str) -> torch.Tensor:
        logits = self._forward_logits(audio_tensor)
        idx = self.vocabulary.index(target_text)
        return logits[idx]

    def compute_gradient(self, audio: np.ndarray, target_text: str) -> np.ndarray:
        x = torch.from_numpy(audio).float().to(self.device).clone().detach().requires_grad_(True)
        score = self.target_score_tensor(x, target_text)
        score.backward()
        return x.grad.detach().cpu().numpy().astype(np.float32)

    def supports_targeted_attack(self) -> bool:
        return True

    def get_model_info(self) -> Dict[str, str]:
        info = super().get_model_info()
        info.update(
            {
                "loaded_path": self.loaded_path,
                "vocabulary_path": self.vocabulary_path or "default_ru_targets",
                "device": str(self.device),
                "input_contract": "raw waveform, shape (batch, time) or (batch, channel, time)",
                "output_contract": "logits for vocabulary classes",
            }
        )
        return info
