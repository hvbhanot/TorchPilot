"""TabularNet: supports numeric + categorical (onehot or embedded) inputs, MLP or residual MLP."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchpilot.schemas import ArchSpec


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}


class _ResidualBlock(nn.Module):
    """Pre-activation residual block. Width-preserving; projection added if dims differ."""

    def __init__(self, in_dim: int, out_dim: int, activation: type[nn.Module], dropout: float, batch_norm: bool):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(in_dim) if batch_norm else nn.Identity()
        self.act1 = activation()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.act2 = activation()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        h = self.fc1(self.act1(self.norm1(x)))
        h = self.fc2(self.drop(self.act2(self.norm2(h))))
        return h + residual


def _build_backbone(spec: ArchSpec, in_dim: int) -> tuple[nn.Module, int]:
    act_cls = _ACTIVATIONS[spec.activation]
    layers: list[nn.Module] = []
    prev = in_dim
    if spec.architecture_type == "residual_mlp":
        for h in spec.hidden_sizes:
            layers.append(_ResidualBlock(prev, h, act_cls, spec.dropout, spec.batch_norm))
            prev = h
    else:
        for h in spec.hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if spec.batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_cls())
            if spec.dropout > 0:
                layers.append(nn.Dropout(spec.dropout))
            prev = h
    return nn.Sequential(*layers), prev


class TabularNet(nn.Module):
    """Tabular model. Forward signature: (x_num, x_cat) -> logits/preds."""

    def __init__(
        self,
        spec: ArchSpec,
        n_numeric: int,
        cat_cardinalities: list[int],
        n_outputs: int,
    ):
        super().__init__()
        self.spec = spec
        self.cat_cardinalities = cat_cardinalities
        self.use_embeddings = spec.categorical_encoding == "embedding" and len(cat_cardinalities) > 0

        if self.use_embeddings:
            self.embeddings = nn.ModuleList(
                [nn.Embedding(c, spec.embedding_dim) for c in cat_cardinalities]
            )
            cat_feature_dim = spec.embedding_dim * len(cat_cardinalities)
        else:
            self.embeddings = None
            cat_feature_dim = sum(cat_cardinalities)  # one-hot total width

        in_dim = n_numeric + cat_feature_dim
        self.backbone, last = _build_backbone(spec, in_dim)
        self.head = nn.Linear(last, n_outputs)

    def _encode_cats(self, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.shape[1] == 0:
            return x_cat.new_zeros((x_cat.shape[0], 0), dtype=torch.float32)
        if self.use_embeddings:
            pieces = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
            return torch.cat(pieces, dim=1)
        # One-hot encode each column with its own cardinality, then concat.
        pieces = [
            F.one_hot(x_cat[:, j], num_classes=card).float()
            for j, card in enumerate(self.cat_cardinalities)
        ]
        return torch.cat(pieces, dim=1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        cat_feat = self._encode_cats(x_cat)
        x = torch.cat([x_num, cat_feat], dim=1) if x_num.shape[1] > 0 else cat_feat
        return self.head(self.backbone(x))


def build_optimizer(spec: ArchSpec, model: nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    if spec.optimizer == "adam":
        return torch.optim.Adam(params, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=spec.learning_rate, weight_decay=spec.weight_decay)
    return torch.optim.SGD(
        params, lr=spec.learning_rate, weight_decay=spec.weight_decay, momentum=0.9
    )


def build_scheduler(spec: ArchSpec, optim: torch.optim.Optimizer):
    if spec.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(spec.epochs, 1))
    if spec.lr_scheduler == "step":
        step = max(spec.epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optim, step_size=step, gamma=0.5)
    return None
