# -*- coding: utf-8 -*-
"""
BiLSTMCompensador: red recurrente bidireccional que toma una ventana
temporal de features (IMU + odometría corregida + marcas + posición baseline)
y produce una corrección residual de posición (Δp_E, Δp_N) en el último
instante de la ventana.

Arquitectura:
    Input (W, n_features)
        ↓
    Bi-LSTM (hidden) × num_layers (con dropout entre capas)
        ↓
    Toma el output de la ÚLTIMA muestra (out[:, -1, :])
        ↓
    Linear (2*hidden → hidden) + ReLU + Dropout
        ↓
    Linear (hidden → 2)  ← (Δp_E, Δp_N)
"""
from __future__ import annotations
import torch
import torch.nn as nn


class BiLSTMCompensador(nn.Module):
    def __init__(self,
                 input_dim: int = 10,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 output_dim: int = 2,
                 dropout: float = 0.3):
        """
        Parameters
        ----------
        input_dim : número de features de entrada por muestra
        hidden_dim : tamaño del estado oculto del Bi-LSTM
        num_layers : número de capas apiladas del Bi-LSTM
        output_dim : 2 = (Δp_E, Δp_N)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Head: 2*hidden (bidireccional) → hidden → output
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, input_dim)
        out, _ = self.lstm(x)              # (batch, window, 2*hidden)
        last = out[:, -1, :]                # (batch, 2*hidden) — última muestra
        delta_p = self.head(last)           # (batch, output_dim)
        return delta_p

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Sanity check
    model = BiLSTMCompensador(input_dim=10, hidden_dim=64, num_layers=2)
    print(f"Modelo: {model}")
    print(f"Parámetros entrenables: {model.count_params():,}")

    # Forward dummy
    batch_size, window, input_dim = 8, 200, 10
    x = torch.randn(batch_size, window, input_dim)
    y = model(x)
    print(f"\nInput shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (batch_size, 2), f"Esperaba (8, 2), obtuve {y.shape}"
    print("✓ Forward OK")
