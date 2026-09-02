# -*- coding: utf-8 -*-
"""
RutaDataset: carga un CSV de una ruta producido por Algoritmo_2_strapdown_basico.m
y construye ventanas temporales (input, target) para entrenar el Bi-LSTM
compensador.

CSV esperado (columnas):
    t, ax, ay, az, wx, wy, wz, axL, ayL, azL,
    q_odo, q_corr, q_truth,
    E_truth, N_truth, U_truth,
    E_base, N_base, U_base,
    res_E, res_N, res_U, mark

Output: ventanas (W, n_features) y targets (n_outputs)
  - features (10 por defecto): axL, ayL, azL, wx, wy, wz, q_corr_norm, mark, E_base, N_base
  - target (2): res_E, res_N en el último instante de la ventana
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FEATURE_COLUMNS = ['axL', 'ayL', 'azL', 'wx', 'wy', 'wz',
                   'q_corr', 'mark', 'E_base', 'N_base']
TARGET_COLUMNS = ['res_E', 'res_N']


def add_synthetic_drift(q_odo: np.ndarray,
                        drift_linear: float = 0.05,
                        drift_sigma_pct: float = 0.02,
                        seed: int = 42) -> np.ndarray:
    """
    Añade deriva sintética a la odometría cruda q_odo para simular
    un odómetro/IMU peor calibrado. Combina:
      - Sesgo lineal: q_odo *= (1 + drift_linear)  (5% típico = calibración imperfecta)
      - Paseo aleatorio: ruido acumulativo con σ_step = drift_sigma_pct·q_final/√N
    Ambos suman deriva ~ (drift_linear ± drift_sigma_pct)·q_final al final.

    Parameters
    ----------
    q_odo : array (N,) de odometría cruda [m]
    drift_linear : sesgo constante (fracción). 0.05 = +5% de exceso.
    drift_sigma_pct : desviación acumulada al final (fracción). 0.02 = ±2%.
    seed : semilla reproducible (típicamente hash del nombre del CSV).

    Returns
    -------
    q_drift : array (N,) con la deriva sintética aplicada
    """
    N = len(q_odo)
    if N == 0:
        return q_odo.copy()
    q_final = float(np.abs(q_odo[-1]))
    if q_final <= 0:
        return q_odo.copy()
    rng = np.random.RandomState(seed)
    sigma_step = drift_sigma_pct * q_final / np.sqrt(max(N, 1))
    random_walk = np.cumsum(rng.standard_normal(N)) * sigma_step
    q_drift = q_odo * (1.0 + drift_linear) + random_walk
    # Mantener monotonicidad (los odómetros no retroceden)
    q_drift = np.maximum.accumulate(q_drift)
    return q_drift


class RutaDataset(Dataset):
    """Dataset de UNA ruta. Lee CSV → downsamplea → ventanas deslizantes."""

    def __init__(self,
                 csv_path: str | Path,
                 window: int = 200,
                 stride: int = 20,
                 downsample_factor: int = 20,
                 feature_cols: Sequence[str] = FEATURE_COLUMNS,
                 target_cols: Sequence[str] = TARGET_COLUMNS,
                 normalize: bool = True,
                 stats: dict | None = None,
                 strict_baseline: bool = False,
                 synthetic_drift: bool = False,
                 drift_linear: float = 0.05,
                 drift_sigma_pct: float = 0.02):
        """
        Parameters
        ----------
        csv_path : ruta al CSV producido por MATLAB
        window : número de muestras por ventana (después del downsample)
        stride : paso entre ventanas (1 = todas, mayor = solapamiento menor)
        downsample_factor : factor de decimación (CSV a 2 kHz → /20 = 100 Hz)
        feature_cols : columnas usadas como input
        target_cols : columnas usadas como target
        normalize : normalización z-score a features
        stats : dict con 'mean' y 'std' para normalización (si None, se calculan
                del propio CSV; útil para train_set vs test_set)
        strict_baseline : si True, recomputa E_base/N_base/res_E/res_N usando
                q_odo (odometría cruda SIN map-matching) en lugar de q_corr.
                Este es el escenario "INS + odometría cruda" tipo Fang (2020).
                False = usa las columnas tal como vienen del CSV (con map-matching).
        """
        self.csv_path = Path(csv_path)
        self.window = window
        self.stride = stride
        self.downsample_factor = downsample_factor
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.normalize = normalize
        self.strict_baseline = strict_baseline

        # 1) Carga
        print(f"Cargando: {self.csv_path.name}"
              f"{' [STRICT baseline: q_odo]' if strict_baseline else ''}")
        df = pd.read_csv(self.csv_path)
        print(f"  Filas originales: {len(df)}")

        # 2) Downsample (decimación simple — toma 1 de cada N)
        if downsample_factor > 1:
            df = df.iloc[::downsample_factor].reset_index(drop=True)
            print(f"  Filas tras downsample x{downsample_factor}: {len(df)}")

        # 2b) Recomputar baseline usando q_odo (sin map-matching) si se pide,
        #     opcionalmente con deriva sintética añadida
        if strict_baseline:
            required_strict = ['q_odo', 'q_truth', 'E_truth', 'N_truth']
            missing_strict = [c for c in required_strict if c not in df.columns]
            if missing_strict:
                raise ValueError(f"strict_baseline requiere columnas ausentes: "
                                 f"{missing_strict}")
            # Aplicar deriva sintética a q_odo si se pide
            if synthetic_drift:
                seed_from_name = abs(hash(self.csv_path.name)) % (2**31)
                q_odo_used = add_synthetic_drift(
                    df['q_odo'].values,
                    drift_linear=drift_linear,
                    drift_sigma_pct=drift_sigma_pct,
                    seed=seed_from_name,
                )
                print(f"  synthetic_drift aplicado: "
                      f"lin={drift_linear:+.1%}, σ={drift_sigma_pct:.1%}")
            else:
                q_odo_used = df['q_odo'].values
            # Reconstruir el mapa (forma del GPS) desde (q_truth, E_truth, N_truth)
            df_map = (df[['q_truth', 'E_truth', 'N_truth']]
                      .drop_duplicates(subset='q_truth')
                      .sort_values('q_truth'))
            q_path = df_map['q_truth'].values
            E_path = df_map['E_truth'].values
            N_path = df_map['N_truth'].values
            # Recomputar posición baseline con q_odo (opcionalmente derivado)
            df['E_base'] = np.interp(q_odo_used, q_path, E_path)
            df['N_base'] = np.interp(q_odo_used, q_path, N_path)
            # Recomputar residuales (target del modelo)
            df['res_E'] = df['E_truth'] - df['E_base']
            df['res_N'] = df['N_truth'] - df['N_base']
            print(f"  strict_baseline aplicado: E_base/N_base/res_E/res_N "
                  f"recomputados desde q_odo{'+deriva' if synthetic_drift else ''}")

        # 3) Verificación de columnas
        missing = [c for c in self.feature_cols + self.target_cols
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Columnas ausentes en CSV: {missing}")

        # 4) Limpieza de NaN/Inf (pueden venir de la interpolación)
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=self.feature_cols + self.target_cols).reset_index(drop=True)
        print(f"  Filas tras limpiar NaN: {len(df)}")

        # 5) Extraer arrays
        self.X = df[self.feature_cols].values.astype(np.float32)
        self.y = df[self.target_cols].values.astype(np.float32)
        self.t = df['t'].values.astype(np.float32) if 't' in df.columns else None

        # 6) Normalización z-score
        if normalize:
            if stats is None:
                self.mean = self.X.mean(axis=0)
                self.std = self.X.std(axis=0) + 1e-8
            else:
                self.mean = stats['mean']
                self.std = stats['std']
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = np.zeros(self.X.shape[1])
            self.std = np.ones(self.X.shape[1])

        # 7) Construir índices de ventanas
        N = len(self.X)
        if N < window:
            raise ValueError(f"Ruta más corta que la ventana: {N} < {window}")
        self.indices = list(range(0, N - window, stride))
        print(f"  Ventanas generadas: {len(self.indices)} "
              f"(W={window}, stride={stride}, fs_eff={2000//downsample_factor} Hz)")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        window = self.X[i:i + self.window]          # (W, n_features)
        target = self.y[i + self.window - 1]         # target = última muestra
        return torch.from_numpy(window), torch.from_numpy(target)

    def stats(self) -> dict:
        """Retorna los stats de normalización (para reutilizar en test set)."""
        return {'mean': self.mean.copy(), 'std': self.std.copy()}

    def info(self) -> dict:
        """Resumen para sanity-check."""
        return {
            'csv_path': str(self.csv_path),
            'num_windows': len(self.indices),
            'window_size': self.window,
            'stride': self.stride,
            'downsample_factor': self.downsample_factor,
            'fs_effective_hz': 2000 // self.downsample_factor,
            'num_features': len(self.feature_cols),
            'num_targets': len(self.target_cols),
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols,
            'feature_mean': self.mean.tolist(),
            'feature_std': self.std.tolist(),
            'target_range_E': [float(self.y[:, 0].min()),
                               float(self.y[:, 0].max())],
            'target_range_N': [float(self.y[:, 1].min()),
                               float(self.y[:, 1].max())],
            'target_rmse_baseline': float(np.sqrt(np.mean(self.y ** 2))),
        }


if __name__ == '__main__':
    # Sanity check: carga la ruta Piedecuesta y muestra info
    import sys
    csv_default = '../export_python/ruta_ParqueTematico_Piedecuesta_012.csv'
    csv_path = sys.argv[1] if len(sys.argv) > 1 else csv_default

    ds = RutaDataset(csv_path, window=200, stride=20, downsample_factor=20)
    print("\n--- INFO ---")
    for k, v in ds.info().items():
        if isinstance(v, list) and len(v) > 5:
            # Detectar si la lista contiene números o strings y formatear acorde
            if v and isinstance(v[0], (int, float)):
                print(f"  {k}: [{v[0]:.3f}, ..., {v[-1]:.3f}] ({len(v)} elementos)")
            else:
                print(f"  {k}: [{v[0]}, ..., {v[-1]}] ({len(v)} elementos)")
        else:
            print(f"  {k}: {v}")

    # Mostrar una muestra
    X0, y0 = ds[0]
    print(f"\nVentana 0: X shape={tuple(X0.shape)}, y shape={tuple(y0.shape)}")
    print(f"X[0,:5]: {X0[0,:5].numpy()}")
    print(f"y: {y0.numpy()}")
