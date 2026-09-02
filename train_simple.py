# -*- coding: utf-8 -*-
"""
train_simple.py — Entrenamiento de prueba con UNA SOLA RUTA.

Objetivo: validar end-to-end que el pipeline Python funciona, sin entrar
todavía en el leave-one-out de las rutas. Hace:
  1. Carga el CSV de Piedecuesta
  2. Divide en train/val/test secuencial (60/20/20)
  3. Entrena la Bi-LSTM
  4. Reporta RMSE de baseline vs RMSE de modelo en el conjunto de test

"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import RutaDataset
from model import BiLSTMCompensador


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv',
                   default='../export_python/ruta_ParqueTematico_Piedecuesta_012.csv',
                   help='ruta al CSV de la ruta de prueba')
    p.add_argument('--window', type=int, default=200, help='tamaño de ventana')
    p.add_argument('--stride', type=int, default=20, help='paso entre ventanas')
    p.add_argument('--downsample', type=int, default=20,
                   help='factor de downsample (2000Hz/factor)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   help='cpu, cuda, mps (Apple Silicon) o auto')
    return p.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_arg)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE 2D considerando (E, N) como vector."""
    err = pred - target
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    # --- 1) Carga del dataset ---
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: no encuentro el CSV: {csv_path}\n"
                 f"Genéra primero con MATLAB (Algoritmo_2_strapdown_basico.m).")

    full_ds = RutaDataset(csv_path, window=args.window, stride=args.stride,
                          downsample_factor=args.downsample, normalize=True)
    print(f"\nDataset total: {len(full_ds)} ventanas")

    # --- 2) Split secuencial 60/20/20 (no aleatorio para preservar continuidad) ---
    N = len(full_ds)
    n_train = int(0.60 * N)
    n_val = int(0.20 * N)
    n_test = N - n_train - n_val
    idx_train = list(range(0, n_train))
    idx_val   = list(range(n_train, n_train + n_val))
    idx_test  = list(range(n_train + n_val, N))

    train_ds = Subset(full_ds, idx_train)
    val_ds   = Subset(full_ds, idx_val)
    test_ds  = Subset(full_ds, idx_test)
    print(f"Split: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # --- 3) Modelo + optim + loss ---
    n_features = len(full_ds.feature_cols)
    model = BiLSTMCompensador(input_dim=n_features,
                              hidden_dim=args.hidden,
                              num_layers=args.layers,
                              output_dim=2,
                              dropout=args.dropout).to(device)
    print(f"\nModelo: {model.count_params():,} parámetros")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # --- 4) Loop de entrenamiento ---
    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    print(f"\nEntrenando {args.epochs} épocas...")
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_losses.append(criterion(pred, y).item())

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        print(f"  Ep {ep+1:3d}/{args.epochs} | train MSE={tl:.4f} | "
              f"val MSE={vl:.4f} | {time.time()-t0:.1f}s")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- 5) Evaluación en test ---
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            all_pred.append(pred)
            all_target.append(y.numpy())

    pred = np.vstack(all_pred)
    target = np.vstack(all_target)
    rmse_baseline = rmse(np.zeros_like(target), target)
    rmse_model = rmse(pred, target)

    print(f"\n=== RESULTADOS (set de test) ===")
    print(f"RMSE baseline (sin Bi-LSTM):     {rmse_baseline:.3f} m")
    print(f"RMSE modelo   (con Bi-LSTM):     {rmse_model:.3f} m")
    if rmse_baseline > 0:
        mejora = (rmse_baseline - rmse_model) / rmse_baseline * 100
        print(f"Mejora porcentual:               {mejora:+.1f}%")
        print(f"Umbral del criterio SMART:       >= 30%")
        print(f"Resultado:                       "
              f"{'✓ APROBADO' if mejora >= 30 else '✗ NO ALCANZA'}")

    # Guardar modelo + historial
    out_dir = Path('../results/models')
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': best_state,
        'args': vars(args),
        'history': history,
        'final_rmse_baseline': rmse_baseline,
        'final_rmse_model': rmse_model,
    }, out_dir / 'bilstm_piedecuesta_simple.pt')
    print(f"\nModelo guardado en: {out_dir / 'bilstm_piedecuesta_simple.pt'}")


if __name__ == '__main__':
    main()
