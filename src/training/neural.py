"""Reusable neural training, reproducible seeds and checkpoint inference."""

from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from src.models.adaptive import AdaptiveFusion
from src.models.reliability import ReliabilityFusion
from src.dataset.quality import validate_quality
from src.dataset.preprocessing import ParticipantPreprocessor


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def resolve_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu, or cuda")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return requested


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def tensors(frame: pd.DataFrame, device: str):
    audio = [c for c in frame if c.startswith("audio__")]
    facial = [c for c in frame if c.startswith("facial__")]
    if not audio or not facial:
        raise ValueError("Both modalities must retain at least one training feature")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite preprocessed descriptors")
    return (
        torch.tensor(frame[audio].to_numpy(), dtype=torch.float32, device=device),
        torch.tensor(frame[facial].to_numpy(), dtype=torch.float32, device=device),
    )


def train_network(
    frame: pd.DataFrame,
    target: pd.Series,
    parameters: dict,
    seed: int,
    device: str,
    quality: pd.DataFrame | None = None,
):
    if not frame.index.equals(target.index) or set(target.unique()) != {0, 1}:
        raise ValueError("Aligned binary training data with both classes required")
    if (
        parameters["epochs"] < 1
        or parameters["learning_rate"] <= 0
        or parameters["weight_decay"] < 0
    ):
        raise ValueError("Invalid optimizer settings")
    seed_everything(seed)
    audio, facial = tensors(frame, device)
    config = {
        "audio_dim": audio.shape[1],
        "facial_dim": facial.shape[1],
        "hidden_dim": parameters["hidden_dim"],
        "dropout": parameters["dropout"],
    }
    if "fusion_mode" in parameters:
        if quality is not None:
            raise ValueError("Fixed fusion ablation does not accept quality inputs")
        config["fusion_mode"] = parameters["fusion_mode"]
    if quality is not None:
        validate_quality(quality, frame.index)
        config["quality_mean"] = quality.mean().tolist()
    model = (
        ReliabilityFusion(**config) if quality is not None else AdaptiveFusion(**config)
    ).to(device)
    inputs = (audio, facial)
    if quality is not None:
        inputs += (
            torch.tensor(quality.to_numpy(), dtype=torch.float32, device=device),
        )
    labels = torch.tensor(target.to_numpy(), dtype=torch.float32, device=device)
    pos_weight = float((target == 0).sum() / (target == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
    )
    history = []
    model.train()
    for epoch in range(parameters["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*inputs)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, config, history, pos_weight


def predict(
    model: AdaptiveFusion,
    frame: pd.DataFrame,
    device: str,
    quality: pd.DataFrame | None = None,
):
    model.eval()
    with torch.no_grad():
        inputs = tensors(frame, device)
        if isinstance(model, ReliabilityFusion):
            if quality is None:
                raise ValueError(
                    "Quality-aware checkpoint requires aligned quality inputs"
                )
            validate_quality(quality, frame.index)
            inputs += (
                torch.tensor(quality.to_numpy(), dtype=torch.float32, device=device),
            )
        elif quality is not None:
            raise ValueError("Stage 5 model does not accept quality inputs")
        logits, weights = model(*inputs)
        return torch.sigmoid(logits).cpu().numpy(), weights.cpu().numpy()


def preprocessing_state(p: ParticipantPreprocessor) -> dict:
    return {
        "input_columns": list(p.feature_names_in_),
        "output_columns": p.columns_,
        "dropped": p.dropped_,
        "medians": p.medians_.to_dict(),
        "means": p.means_.to_dict(),
        "scales": p.scales_.to_dict(),
    }


def predict_checkpoint(
    path: Path,
    raw: pd.DataFrame,
    device: str = "cpu",
    quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    device = resolve_device(device)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    prep = ParticipantPreprocessor.from_state(saved["preprocessing"])
    model_type = (
        ReliabilityFusion if "quality_mean" in saved["model_config"] else AdaptiveFusion
    )
    model = model_type(**saved["model_config"]).to(device)
    model.load_state_dict(saved["state_dict"])
    probability, weights = predict(model, prep.transform(raw), device, quality)
    return pd.DataFrame(
        {
            "probability_class_1": probability,
            "audio_weight": weights[:, 0],
            "facial_weight": weights[:, 1],
        },
        index=raw.index,
    )
