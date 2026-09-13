from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MarketTransformer(nn.Module):
    def __init__(self, n_features: int, dim: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.project = nn.Linear(n_features, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=192,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pos = PositionalEncoding(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.pos(self.project(x)))
        return self.head(hidden[:, -1]).squeeze(-1)


def fit_classical(x: np.ndarray, y: np.ndarray):
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced")
    boosting = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
    )
    logistic.fit(x, y)
    boosting.fit(x, y)
    return logistic, boosting


def _batches(n: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def fit_transformer(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 128,
    patience: int = 5,
    seed: int = 7,
) -> MarketTransformer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = select_device()
    model = MarketTransformer(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    x_train = torch.as_tensor(x, dtype=torch.float32)
    y_train = torch.as_tensor(y, dtype=torch.float32)
    x_valid = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    y_valid = torch.as_tensor(y_val, dtype=torch.float32, device=device)

    best_state = None
    best_val = float("inf")
    stale = 0
    order = np.arange(len(x_train))

    for _ in range(epochs):
        model.train()
        np.random.shuffle(order)
        for start, end in _batches(len(order), batch_size):
            idx = order[start:end]
            batch_x = x_train[idx].to(device)
            batch_y = y_train[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_valid), y_valid).cpu())
        if val_loss + 1e-4 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model.to("cpu")


def transformer_proba(model: MarketTransformer, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    tensor = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for start, end in _batches(len(tensor), batch_size):
            batch = tensor[start:end].to(device)
            out.append(torch.sigmoid(model(batch)).cpu().numpy())
    if not out:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out)
