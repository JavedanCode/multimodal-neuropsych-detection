"""Small masked temporal model with explicit quality-weighted modality gating."""
import numpy as np
import torch
from torch import nn

MODALITIES = ('text', 'audio', 'face')


class SequenceScaler:
    def fit(self, arrays, masks):
        self.parameters = []
        for i, x in enumerate(arrays):
            valid = x[masks[:, i]]
            if not len(valid):
                raise ValueError('Empty fitting modality')
            mean, sd = valid.mean(0), valid.std(0)
            self.parameters.append((mean, np.where(sd > 1e-6, sd, 1)))
        return self

    def transform(self, arrays, masks):
        return [torch.tensor(np.where(masks[:, i, :, None], (x-mean)/sd, 0), dtype=torch.float32)
                for i, (x, (mean, sd)) in enumerate(zip(arrays, self.parameters))]


class TemporalFusion(nn.Module):
    def __init__(self, dimensions, temporal=True, quality=False):
        super().__init__()
        self.quality = quality
        kernel = 3 if temporal else 1
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, 16), nn.ReLU()) for d in dimensions])
        self.convs = nn.ModuleList([nn.Conv1d(16, 16, kernel, padding=kernel//2) for _ in dimensions])
        self.gate = nn.Linear(16 * len(dimensions), len(dimensions))
        self.head = nn.Linear(16, 1)

    def forward(self, arrays, masks, quality, blind=False):
        vectors = []
        for i, (x, encoder, conv) in enumerate(zip(arrays, self.encoders, self.convs)):
            mask = masks[:, i, :, None]
            projected = encoder(x) * mask
            hidden = torch.relu(conv(projected.transpose(1, 2)).transpose(1, 2)) * mask
            vectors.append(hidden.sum(1) / mask.sum(1).clamp(min=1))
        available = masks.any(2)
        if not available.any(1).all():
            raise ValueError('All modalities missing')
        logits = self.gate(torch.cat(vectors, 1)).masked_fill(~available, -1e9)
        weights = torch.softmax(logits, 1)
        if self.quality and not blind:
            q = (quality * masks).sum(2) / masks.sum(2).clamp(min=1)
            weights = weights * q.clamp(min=1e-6) * available
            weights = weights / weights.sum(1, keepdim=True).clamp(min=1e-9)
        fused = (torch.stack(vectors, 1) * weights[:, :, None]).sum(1)
        return self.head(fused).squeeze(1), weights


def perturb(arrays, masks, quality, scenario, seed):
    """Deterministic feature stress; quality for injected noise is oracle information."""
    rng = np.random.default_rng(seed)
    xs = [x.clone() for x in arrays]
    m, q = masks.clone(), quality.clone()
    if scenario == 'audio_noise':
        xs[1] += torch.tensor(rng.normal(0, 2, xs[1].shape), dtype=xs[1].dtype) * m[:, 1, :, None]
        q[:, 1] /= 5
    elif scenario == 'face_loss':
        keep = torch.tensor(rng.random(m[:, 2].shape) >= .75)
        m[:, 2] &= keep
        q[:, 2] *= keep
        # Retained-bin fraction is an observable availability loss, not label information.
        q[:, 2] *= .25
    elif scenario.startswith('missing_'):
        idx = MODALITIES.index(scenario[8:])
        m[:, idx] = False
        q[:, idx] = 0
    elif scenario != 'clean':
        raise ValueError('Unknown stress condition')
    for i in range(len(xs)):
        xs[i] *= m[:, i, :, None]
    return xs, m, q


def augment(arrays, masks, quality, seed):
    rng = np.random.default_rng(seed)
    xs = [x.clone() for x in arrays]
    m, q = masks.clone(), quality.clone()
    drop = rng.random((len(m), 3)) < .15
    available = masks.any(2).numpy()
    for row in range(len(m)):
        if not (available[row] & ~drop[row]).any():
            drop[row, np.flatnonzero(available[row])[0]] = False
    for i in range(3):
        m[:, i] &= ~torch.tensor(drop[:, i, None])
        noisy = torch.tensor(rng.random(len(m)) < .3)
        xs[i] += torch.tensor(rng.normal(0, 2, xs[i].shape), dtype=xs[i].dtype) * noisy[:, None, None]
        xs[i] *= m[:, i, :, None]
        q[:, i] *= m[:, i]
        q[noisy, i] /= 5
    return xs, m, q
