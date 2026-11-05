"""Fit selected acoustic/facial features with the fixed neural recipe."""

from __future__ import annotations
import time
import pandas as pd
from src.dataset.preprocessing import ParticipantPreprocessor
from src.dataset.quality import QUALITY_COLUMNS, validate_quality
from src.training.validation import TARGET_POLICY
from src.training.neural import train_network, predict, preprocessing_state, synchronize
from src.evaluation.metrics import binary_metrics


def fit_selected(
    train,
    y,
    dev,
    dy,
    columns,
    parameters,
    seed,
    device,
    train_quality=None,
    dev_quality=None,
):
    from src.training.validation import validate_partition

    validate_partition(train, y)
    validate_partition(dev, dy)
    if len(train.index.intersection(dev.index)) or list(train) != list(dev):
        raise ValueError("Overlapping participants or schema mismatch")
    if (
        not columns
        or len(set(columns)) != len(columns)
        or not set(columns).issubset(train.columns)
    ):
        raise ValueError("Invalid selected descriptor names")
    if (train_quality is None) != (dev_quality is None):
        raise ValueError("Both quality tables required")
    if train_quality is not None:
        validate_quality(train_quality, train.index)
        validate_quality(dev_quality, dev.index)
    synchronize(device)
    started = time.perf_counter()
    prep = ParticipantPreprocessor().fit(train[columns])
    model, config, history, pos_weight = train_network(
        prep.transform(train[columns]), y, parameters, seed, device, train_quality
    )
    synchronize(device)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability, weights = predict(
        model, prep.transform(dev[columns]), device, dev_quality
    )
    synchronize(device)
    inference_seconds = time.perf_counter() - started
    checkpoint = {
        "model_config": config,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "preprocessing": preprocessing_state(prep),
        "train_participant_ids": list(map(int, train.index)),
        "selected_features": columns,
        "seed": seed,
        "target_policy": TARGET_POLICY,
        "threshold": 0.5,
    }
    if train_quality is not None:
        checkpoint["quality_columns"] = QUALITY_COLUMNS
    result = {
        "seed": seed,
        "quality_inputs": train_quality is not None,
        "original_feature_count": len(train.columns),
        "selected_feature_count": len(columns),
        "retained_feature_count": len(prep.columns_),
        "selected_features": [
            {"name": c, "modality": c.split("__", 1)[0]} for c in columns
        ],
        "retained_feature_names": prep.columns_,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameters": parameters,
        "train_positive_weight": pos_weight,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "training_loss_by_epoch": history,
        "dev_metrics": binary_metrics(dy, probability),
        "test_evaluated": False,
        "target_policy": TARGET_POLICY,
    }
    predictions = pd.DataFrame(
        {
            "PHQ_Binary": dy,
            "probability_class_1": probability,
            "prediction": (probability >= 0.5).astype(int),
            "audio_weight": weights[:, 0],
            "facial_weight": weights[:, 1],
        },
        index=dev.index,
    )
    return checkpoint, result, predictions
