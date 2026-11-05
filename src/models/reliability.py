"""Reusable helpers extracted from the recorded research experiments."""

from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from src.models.adaptive import AdaptiveFusion
from src.dataset.quality import QUALITY_COLUMNS


class ReliabilityFusion(AdaptiveFusion):
    def __init__(
        self,
        audio_dim: int,
        facial_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.2,
        quality_mean: list[float] | None = None,
    ):
        super().__init__(audio_dim, facial_dim, hidden_dim, dropout)
        means = torch.tensor(
            quality_mean if quality_mean is not None else [0.0] * 8, dtype=torch.float32
        )
        if (
            means.shape != (8,)
            or not torch.isfinite(means).all()
            or (means < 0).any()
            or (means > 1).any()
        ):
            raise ValueError("Eight finite training quality means in [0,1] required")
        self.register_buffer("quality_mean", means)
        # No additional RNG draws: matches Stage 5 initialization and dropout RNG.
        self.quality_weight = nn.Parameter(torch.zeros(8, len(QUALITY_COLUMNS)))

    def forward(self, audio: torch.Tensor, facial: torch.Tensor, quality: torch.Tensor):
        if audio.ndim != 2 or facial.ndim != 2 or audio.shape[0] != facial.shape[0]:
            raise ValueError("Expected matching participant batches")
        if (
            audio.shape[1] != self.audio_dim
            or facial.shape[1] != self.facial_dim
            or quality.shape != (audio.shape[0], 8)
        ):
            raise ValueError("Feature or quality shape mismatch")
        if (
            not torch.isfinite(quality).all()
            or (quality < 0).any()
            or (quality > 1).any()
        ):
            raise ValueError("Invalid quality values")
        available = quality[:, [0, 3]]
        if (
            not ((available == 0) | (available == 1)).all()
            or (available.sum(dim=1) == 0).any()
        ):
            raise ValueError("At least one modality must be available")
        # Unavailable measurements cannot affect either representations or gate.
        a = (
            self.audio_encoder(
                torch.where(available[:, :1].bool(), audio, torch.zeros_like(audio))
            )
            * available[:, :1]
        )
        f = (
            self.facial_encoder(
                torch.where(available[:, 1:].bool(), facial, torch.zeros_like(facial))
            )
            * available[:, 1:]
        )
        hidden = self.gate[0](torch.cat([a, f], dim=1)) + F.linear(
            quality - self.quality_mean, self.quality_weight
        )
        logits = self.gate[2](self.gate[1](hidden))
        weights = torch.softmax(
            logits.masked_fill(available == 0, float("-inf")), dim=1
        )
        fused = weights[:, :1] * a + weights[:, 1:] * f
        return self.head(fused).squeeze(-1), weights
