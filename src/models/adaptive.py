"""Representation-only adaptive fusion, without explicit quality inputs."""
from __future__ import annotations
import torch
from torch import nn


class AdaptiveFusion(nn.Module):
    def __init__(self, audio_dim: int, facial_dim: int, hidden_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        if min(audio_dim, facial_dim, hidden_dim) < 1 or not 0 <= dropout < 1:
            raise ValueError('Positive feature dimensions and dropout in [0,1) required')
        self.audio_dim, self.facial_dim = audio_dim, facial_dim
        self.audio_encoder = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.facial_encoder = nn.Sequential(nn.Linear(facial_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, 8), nn.ReLU(), nn.Linear(8, 2))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, audio: torch.Tensor, facial: torch.Tensor):
        if audio.ndim != 2 or facial.ndim != 2 or audio.shape[0] != facial.shape[0]:
            raise ValueError('Expected matching participant batches')
        if audio.shape[1] != self.audio_dim or facial.shape[1] != self.facial_dim:
            raise ValueError('Modality feature dimensions differ from model')
        a, f = self.audio_encoder(audio), self.facial_encoder(facial)
        weights = torch.softmax(self.gate(torch.cat([a, f], dim=1)), dim=1)
        fused = weights[:, :1]*a + weights[:, 1:]*f
        return self.head(fused).squeeze(-1), weights
