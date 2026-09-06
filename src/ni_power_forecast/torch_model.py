from __future__ import annotations

import numpy as np


class TorchMLPRegressor:
    """Small sklearn-like PyTorch MLP for optional experimentation."""

    def __init__(
        self, hidden: int = 64, epochs: int = 120, lr: float = 1e-3, random_seed: int = 42
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.random_seed = random_seed
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError('Install PyTorch support with: pip install -e ".[torch]"') from exc

        torch.manual_seed(self.random_seed)
        x = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.x_mean = x.mean(axis=0, keepdims=True)
        self.x_std = x.std(axis=0, keepdims=True) + 1e-6
        self.y_mean = target.mean(axis=0, keepdims=True)
        self.y_std = target.std(axis=0, keepdims=True) + 1e-6

        x = (x - self.x_mean) / self.x_std
        target = (target - self.y_mean) / self.y_std

        xt = torch.tensor(x)
        yt = torch.tensor(target)
        self.model = nn.Sequential(
            nn.Linear(x.shape[1], self.hidden),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.SmoothL1Loss()
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.model(xt)
            loss = loss_fn(pred, yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Model has not been fitted")
        import torch

        x = np.asarray(X, dtype=np.float32)
        x = (x - self.x_mean) / self.x_std
        self.model.eval()
        with torch.no_grad():
            pred = self.model(torch.tensor(x)).numpy()
        pred = pred * self.y_std + self.y_mean
        return pred.reshape(-1)
