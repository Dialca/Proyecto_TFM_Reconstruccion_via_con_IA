# -*- coding: utf-8 -*-
"""
Salidas en ../results/:
    leaveoneout_summary.csv     : tabla resumen (una fila por fold)
    fold_XX_YYY/predictions.csv : predicciones de cada fold
    fold_XX_YYY/model.pt        : modelo entrenado
    fold_XX_YYY/train_history.csv : loss por época
    figures/*.png               : gráficas comparativas
"""
from __future__ import annotations

# =============================================================================
# ═══════════════════ HIPERPARÁMETROS Y CONFIGURACIÓN ═══════════════════════
# =============================================================================

# ---- 1) MODELO Bi-LSTM ----
HIDDEN_DIM  = 32      # tamaño del estado oculto 
NUM_LAYERS  = 1       # capas apiladas del Bi-LSTM 
DROPOUT     = 0.5     # dropout entre capas 
OUTPUT_DIM  = 2       # 2 = (Δp_E, Δp_N)

# ---- 2) VENTANAS TEMPORALES ----
WINDOW      = 200     # muestras por ventana (a 100 Hz = 2 segundos)
STRIDE      = 20      # paso entre ventanas 
DOWNSAMPLE  = 20      # decimación de 2000 Hz → 100 Hz (factor 20)

# ---- 3) ENTRENAMIENTO ----
EPOCHS      = 30      # número de épocas
BATCH_SIZE  = 64      # tamaño del batch
LR          = 5e-4    # learning rate
WEIGHT_DECAY = 1e-4    # L2 regularization 
GRAD_CLIP   = 1.0     # gradient clipping máximo
VAL_FRAC    = 0.10    # fracción del TRAIN 
EARLY_STOP_PATIENCE = 20   # épocas sin mejorar val_loss antes de parar 

# ---- 4) REPRODUCIBILIDAD ----
SEED        = 42

# ---- 5) DISPOSITIVO ----
DEVICE      = 'auto'  

# ---- 6) CRITERIO SMART ----
SMART_THRESHOLD_PCT = 30.0   # umbral de mejora porcentual del objetivo general

# ---- 6b) MODO DE BASELINE ----

STRICT_BASELINE = True

# ---- 6c) DERIVA
SYNTHETIC_DRIFT       = True    # habilitar/deshabilitar
DRIFT_LINEAR          = 0.05    # sesgo lineal (0.05 = +5% de exceso)
DRIFT_SIGMA_PCT       = 0.02    # ruido acumulativo (0.02 = ±2% al final)

# ---- 6d) FILTRO DE RUTAS POR LONGITUD ----

ROUTE_MODE               = 'cortas'     # 'todas' | 'cortas' | 'largas'
ROUTE_LENGTH_THRESHOLD_M = 200.0        # umbral en metros (Camino 3: solo rutas > 200 m)

# Longitud media (m) de cada ruta física, medida sobre q_truth final del CSV.
# Se usa para clasificar cortas vs largas en tiempo de ejecución.
ROUTE_LENGTHS_M = {
    'UIS_Guatiguara_5':           43.1,   # corta
    'UIS_Guatiguara_2':           52.8,   # corta
    'UIS_Guatiguara_1':           61.0,   # corta
    'UIS_Guatiguara_3':           71.7,   # corta
    'QuintaGranada_Parqueadero':  77.3,   # corta
    'Bomba_Altoque':              96.7,   # corta
    'EscuelaConduccion':         151.8,   # larga
    'Mazda_ICP':                 280.7,   # larga
    'Chircal_icp':               354.0,   # larga
    'ParqueTematico_Piedecuesta': 415.5,  # larga
}

# ---- 7) CSVs VÁLIDOS SELECCIONADOS (los 26 tras filtrado) ----
VALID_CSVS = [
    # Bomba_Altoque (2)
    "ruta_Bomba_Altoque_012.csv",
    "ruta_Bomba_Altoque_014.csv",
    # Chircal_icp (3)
    "ruta_Chircal_icp_013.csv",
    "ruta_Chircal_icp_014.csv",
    "ruta_Chircal_icp_015.csv",
    # EscuelaConduccion (2)
    "ruta_EscuelaConduccion_012.csv",
    "ruta_EscuelaConduccion_013.csv",
    # Mazda_ICP (3)
    "ruta_Mazda_ICP_012.csv",
    "ruta_Mazda_ICP_013.csv",
    "ruta_Mazda_ICP_014.csv",
    # ParqueTematico_Piedecuesta (2, la 014 se descartó por RMSE=34.6 m)
    "ruta_ParqueTematico_Piedecuesta_012.csv",
    "ruta_ParqueTematico_Piedecuesta_013.csv",
    # QuintaGranada_Parqueadero (2)
    "ruta_QuintaGranada_Parqueadero_012.csv",
    "ruta_QuintaGranada_Parqueadero_013.csv",
    # UIS_Guatiguara_1 (4)
    "ruta_UIS_Guatiguara_1_012.csv",
    "ruta_UIS_Guatiguara_1_013.csv",
    "ruta_UIS_Guatiguara_1_014.csv",
    "ruta_UIS_Guatiguara_1_015.csv",
    # UIS_Guatiguara_2 (4 OK; 018/019/020 descartados por outliers)
    "ruta_UIS_Guatiguara_2_013.csv",
    "ruta_UIS_Guatiguara_2_014.csv",
    "ruta_UIS_Guatiguara_2_016.csv",
    "ruta_UIS_Guatiguara_2_012.csv",
    # UIS_Guatiguara_3 (2 OK; 018/019 descartados por outliers/duración)
    "ruta_UIS_Guatiguara_3_020.csv",
    "ruta_UIS_Guatiguara_3_022.csv",
    # UIS_Guatiguara_5 (2 OK; 012/014/019 descartados por duración)
    "ruta_UIS_Guatiguara_5_013.csv",
    "ruta_UIS_Guatiguara_5_018.csv",
]

# Ruta a la carpeta que contiene los CSVs
INTERMEDIATE_DIR = "../intermediate"

# Ruta a la carpeta donde guardar los resultados
RESULTS_DIR = "../results"

# =============================================================================
# ═══════════════════ FIN DE HIPERPARÁMETROS ═══════════════════════════════
# =============================================================================


import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset

from model import BiLSTMCompensador
from dataset import FEATURE_COLUMNS, TARGET_COLUMNS, add_synthetic_drift

warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------

def extract_route_name(csv_name: str) -> str:
    """De 'ruta_Bomba_Altoque_012.csv' → 'Bomba_Altoque'"""
    m = re.match(r'ruta_(.+?)_(\d+)\.csv$', csv_name)
    if not m:
        raise ValueError(f"No puedo extraer nombre de ruta de: {csv_name}")
    return m.group(1)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(device_arg)


def rmse_2d(pred: np.ndarray, target: np.ndarray) -> float:
    """RMSE 2D: sqrt(mean(sum(err^2, axis=1)))"""
    err = pred - target
    return float(np.sqrt((err ** 2).sum(axis=1).mean()))


# ----------------------------------------------------------------------------
# Dataset multi-CSV
# ----------------------------------------------------------------------------

class MultiCSVDataset(Dataset):
 

    def __init__(self, csv_paths: list[str], window: int, stride: int,
                 downsample: int, mean: np.ndarray | None = None,
                 std: np.ndarray | None = None,
                 strict_baseline: bool = False,
                 synthetic_drift: bool = False,
                 drift_linear: float = 0.05,
                 drift_sigma_pct: float = 0.02):
        self.csv_paths = [Path(p) for p in csv_paths]
        self.window = window
        self.strict_baseline = strict_baseline
        self.synthetic_drift = synthetic_drift

        # 1) Cargar y decimar cada CSV
        Xs, ys = [], []
        for p in self.csv_paths:
            df = pd.read_csv(p)
            if downsample > 1:
                df = df.iloc[::downsample].reset_index(drop=True)

            # 1b) Recomputar baseline (q_odo, opcionalmente con deriva) si strict_baseline
            if strict_baseline:
                required_strict = ['q_odo', 'q_truth', 'E_truth', 'N_truth']
                missing_strict = [c for c in required_strict if c not in df.columns]
                if missing_strict:
                    raise ValueError(f"strict_baseline requiere columnas ausentes "
                                     f"en {p.name}: {missing_strict}")
                if synthetic_drift:
                    seed_from_name = abs(hash(p.name)) % (2**31)
                    q_odo_used = add_synthetic_drift(
                        df['q_odo'].values,
                        drift_linear=drift_linear,
                        drift_sigma_pct=drift_sigma_pct,
                        seed=seed_from_name,
                    )
                else:
                    q_odo_used = df['q_odo'].values
                df_map = (df[['q_truth', 'E_truth', 'N_truth']]
                          .drop_duplicates(subset='q_truth')
                          .sort_values('q_truth'))
                q_path = df_map['q_truth'].values
                E_path = df_map['E_truth'].values
                N_path = df_map['N_truth'].values
                df['E_base'] = np.interp(q_odo_used, q_path, E_path)
                df['N_base'] = np.interp(q_odo_used, q_path, N_path)
                df['res_E'] = df['E_truth'] - df['E_base']
                df['res_N'] = df['N_truth'] - df['N_base']

            df = df.replace([np.inf, -np.inf], np.nan).dropna(
                subset=list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
            ).reset_index(drop=True)
            X = df[list(FEATURE_COLUMNS)].values.astype(np.float32)
            y = df[list(TARGET_COLUMNS)].values.astype(np.float32)
            Xs.append(X)
            ys.append(y)

        # Cada CSV mantiene sus ventanas separadas 
        self.per_csv = list(zip(Xs, ys))
        all_X_stacked = np.concatenate(Xs, axis=0)

        if mean is None or std is None:
            self.mean = all_X_stacked.mean(axis=0)
            self.std = all_X_stacked.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        # 3) Construir tabla global de índices (csv_id, offset_dentro_del_csv)
        self.indices = []
        for csv_id, (X, y) in enumerate(self.per_csv):
            N = len(X)
            if N < window:
                continue
            for i in range(0, N - window, stride):
                self.indices.append((csv_id, i))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        csv_id, i = self.indices[idx]
        X, y = self.per_csv[csv_id]
        window = (X[i:i + self.window] - self.mean) / self.std
        target = y[i + self.window - 1]
        return (torch.from_numpy(window.astype(np.float32)),
                torch.from_numpy(target))

    def compute_baseline_rmse(self) -> float:
        """RMSE 2D del baseline (predicción = 0) sobre TODAS las ventanas."""
        all_y = np.array([
            self.per_csv[csv_id][1][i + self.window - 1]
            for csv_id, i in self.indices
        ])
        return rmse_2d(np.zeros_like(all_y), all_y)


# ----------------------------------------------------------------------------
# Bucle de entrenamiento de UN fold
# ----------------------------------------------------------------------------

def train_one_fold(train_ds: MultiCSVDataset,
                   val_ds: Subset,
                   test_ds: MultiCSVDataset,
                   device: torch.device,
                   verbose: bool = True) -> dict:
    """Entrena un modelo, evalúa, devuelve dict con métricas y predicciones."""
    n_features = len(FEATURE_COLUMNS)
    model = BiLSTMCompensador(
        input_dim=n_features,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        output_dim=OUTPUT_DIM,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)

    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}

    for ep in range(EPOCHS):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_losses.append(criterion(model(X), y).item())

        tl = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        history['epoch'].append(ep + 1)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)

        if verbose:
            print(f"  Ep {ep+1:3d}/{EPOCHS} | train MSE={tl:8.4f} | "
                  f"val MSE={vl:8.4f} | {time.time()-t0:.1f}s")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if EARLY_STOP_PATIENCE and epochs_no_improve >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"  Early stop en época {ep+1} (paciencia={EARLY_STOP_PATIENCE}).")
                break

    # Reload best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Evaluación en test
    model.eval()
    all_pred, all_target = [], []
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            all_pred.append(model(X).cpu().numpy())
            all_target.append(y.numpy())

    pred = np.vstack(all_pred)
    target = np.vstack(all_target)
    baseline_rmse = rmse_2d(np.zeros_like(target), target)
    model_rmse = rmse_2d(pred, target)
    if baseline_rmse > 0:
        improvement = (baseline_rmse - model_rmse) / baseline_rmse * 100
    else:
        improvement = 0.0

    return {
        'model_state': best_state,
        'history': history,
        'pred': pred,
        'target': target,
        'baseline_rmse': baseline_rmse,
        'model_rmse': model_rmse,
        'improvement_pct': improvement,
        'smart_ok': improvement >= SMART_THRESHOLD_PCT,
    }


# ----------------------------------------------------------------------------
# Bucle principal: leave-one-out
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Solo lista los folds y sale sin entrenar')
    parser.add_argument('--folds', default=None,
                        help='Índices de folds a ejecutar (ej. 1,3,5). Por defecto: todos.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Sobrescribe EPOCHS')
    args = parser.parse_args()

    global EPOCHS
    if args.epochs is not None:
        EPOCHS = args.epochs

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = pick_device(DEVICE)
    print(f"Device: {device}")
    print(f"Baseline mode: "
          f"{'STRICT (q_odo, sin map-matching)' if STRICT_BASELINE else 'HYBRID (q_corr, con map-matching)'}")
    if STRICT_BASELINE and SYNTHETIC_DRIFT:
        print(f"Deriva sintética ON: linear={DRIFT_LINEAR:+.1%}, "
              f"σ={DRIFT_SIGMA_PCT:.1%}")
    print(f"Modo rutas: {ROUTE_MODE.upper()} (umbral = {ROUTE_LENGTH_THRESHOLD_M} m)")

    intermediate_dir = Path(INTERMEDIATE_DIR)
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'figures').mkdir(exist_ok=True)

    # 1) Verificar existencia de todos los CSVs
    missing = [c for c in VALID_CSVS
               if not (intermediate_dir / c).exists()]
    if missing:
        print(f"ERROR: no encuentro {len(missing)} CSVs:")
        for c in missing:
            print(f"  - {c}")
        sys.exit(1)

    # 1b) Filtrar CSVs según ROUTE_MODE (cortas/largas/todas)
    if ROUTE_MODE == 'cortas':
        keep_routes = {r for r, L in ROUTE_LENGTHS_M.items()
                       if L <= ROUTE_LENGTH_THRESHOLD_M}
    elif ROUTE_MODE == 'largas':
        keep_routes = {r for r, L in ROUTE_LENGTHS_M.items()
                       if L > ROUTE_LENGTH_THRESHOLD_M}
    elif ROUTE_MODE == 'todas':
        keep_routes = set(ROUTE_LENGTHS_M.keys())
    else:
        sys.exit(f"ROUTE_MODE inválido: {ROUTE_MODE!r} "
                 f"(usa 'cortas', 'largas' o 'todas')")

    filtered_csvs = [c for c in VALID_CSVS
                     if extract_route_name(c) in keep_routes]
    if not filtered_csvs:
        sys.exit(f"El filtro ROUTE_MODE={ROUTE_MODE!r} dejó 0 CSVs.")
    print(f"Filtro ROUTE_MODE={ROUTE_MODE!r}: {len(filtered_csvs)}/{len(VALID_CSVS)} "
          f"CSVs seleccionados ({len(keep_routes)} rutas físicas)")

    # 2) Agrupar por ruta física
    groups: dict[str, list[str]] = {}
    for csv in filtered_csvs:
        route = extract_route_name(csv)
        groups.setdefault(route, []).append(csv)

    routes = sorted(groups.keys())
    n_folds = len(routes)
    print(f"\n{n_folds} rutas físicas detectadas, {len(filtered_csvs)} CSVs válidos.")
    print("Grupos:")
    for r in routes:
        print(f"  {r:<30} ← {len(groups[r])} ejec: {groups[r]}")

    # Filtro de folds a ejecutar
    if args.folds:
        want = {int(x) for x in args.folds.split(',')}
        selected_routes = [routes[i - 1] for i in sorted(want)
                           if 1 <= i <= n_folds]
    else:
        selected_routes = routes

    if args.dry_run:
        print(f"\n[DRY-RUN] Folds a ejecutar: {selected_routes}")
        return

    # 3) Loop de folds
    all_results = []
    t_start = time.time()

    for fold_idx, test_route in enumerate(selected_routes, start=1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{len(selected_routes)}: TEST = {test_route}")
        print(f"{'='*70}")

        train_csvs = []
        for r in routes:
            if r != test_route:
                train_csvs.extend([str(intermediate_dir / c) for c in groups[r]])
        test_csvs = [str(intermediate_dir / c) for c in groups[test_route]]

        print(f"  Train: {len(train_csvs)} CSVs ({len(routes)-1} rutas)")
        print(f"  Test:  {len(test_csvs)} CSVs ({test_route})")

        # Construir train dataset (calcula sus stats de normalización)
        train_full = MultiCSVDataset(train_csvs, window=WINDOW, stride=STRIDE,
                                     downsample=DOWNSAMPLE,
                                     strict_baseline=STRICT_BASELINE,
                                     synthetic_drift=SYNTHETIC_DRIFT,
                                     drift_linear=DRIFT_LINEAR,
                                     drift_sigma_pct=DRIFT_SIGMA_PCT)
        print(f"  Ventanas train (total): {len(train_full)}")

        # Split train/val (secuencial: última VAL_FRAC como validación)
        n_val = int(len(train_full) * VAL_FRAC)
        train_indices = list(range(len(train_full) - n_val))
        val_indices = list(range(len(train_full) - n_val, len(train_full)))
        train_ds = Subset(train_full, train_indices)
        val_ds = Subset(train_full, val_indices)
        print(f"  Ventanas train/val: {len(train_ds)} / {len(val_ds)}")

        # Construir test dataset con las STATS de train (importante!)
        test_ds = MultiCSVDataset(test_csvs, window=WINDOW, stride=STRIDE,
                                  downsample=DOWNSAMPLE,
                                  mean=train_full.mean, std=train_full.std,
                                  strict_baseline=STRICT_BASELINE,
                                  synthetic_drift=SYNTHETIC_DRIFT,
                                  drift_linear=DRIFT_LINEAR,
                                  drift_sigma_pct=DRIFT_SIGMA_PCT)
        print(f"  Ventanas test: {len(test_ds)}")

        # Entrenar
        fold_result = train_one_fold(train_ds, val_ds, test_ds, device,
                                     verbose=True)

        # 3e) Guardar resultados del fold (sufijo para no sobrescribir escenarios)
        mode_tag_baseline = 'strict' if STRICT_BASELINE else 'hybrid'
        mode_tag_drift = 'drift' if (STRICT_BASELINE and SYNTHETIC_DRIFT) else 'nodrift'
        mode_tag_routes = ROUTE_MODE   # 'todas' | 'cortas' | 'largas'
        run_tag = f"{mode_tag_baseline}_{mode_tag_drift}_{mode_tag_routes}"
        fold_dir = results_dir / f"fold_{fold_idx:02d}_{test_route}_{run_tag}"
        fold_dir.mkdir(exist_ok=True)
        torch.save({'model_state': fold_result['model_state'],
                    'hyperparams': dict(HIDDEN_DIM=HIDDEN_DIM,
                                        NUM_LAYERS=NUM_LAYERS,
                                        DROPOUT=DROPOUT,
                                        WINDOW=WINDOW,
                                        DOWNSAMPLE=DOWNSAMPLE,
                                        LR=LR,
                                        EPOCHS=EPOCHS,
                                        STRICT_BASELINE=STRICT_BASELINE,
                                        SYNTHETIC_DRIFT=SYNTHETIC_DRIFT,
                                        DRIFT_LINEAR=DRIFT_LINEAR,
                                        DRIFT_SIGMA_PCT=DRIFT_SIGMA_PCT,
                                        ROUTE_MODE=ROUTE_MODE)},
                   fold_dir / 'model.pt')
        pd.DataFrame(fold_result['history']).to_csv(
            fold_dir / 'train_history.csv', index=False)
        pd.DataFrame({
            'pred_E': fold_result['pred'][:, 0],
            'pred_N': fold_result['pred'][:, 1],
            'target_E': fold_result['target'][:, 0],
            'target_N': fold_result['target'][:, 1],
        }).to_csv(fold_dir / 'predictions.csv', index=False)

        summary_row = {
            'fold': fold_idx,
            'test_route': test_route,
            'n_train_csvs': len(train_csvs),
            'n_test_csvs': len(test_csvs),
            'n_train_windows': len(train_ds),
            'n_val_windows': len(val_ds),
            'n_test_windows': len(test_ds),
            'baseline_rmse_m': fold_result['baseline_rmse'],
            'model_rmse_m': fold_result['model_rmse'],
            'improvement_pct': fold_result['improvement_pct'],
            'smart_ok': fold_result['smart_ok'],
        }
        all_results.append(summary_row)

        print(f"\n  → Baseline: {fold_result['baseline_rmse']:.3f} m"
              f" | Modelo: {fold_result['model_rmse']:.3f} m"
              f" | Mejora: {fold_result['improvement_pct']:+.1f}%"
              f" | SMART: {'✓' if fold_result['smart_ok'] else '✗'}")

    # Resumen global
    df_summary = pd.DataFrame(all_results)
    # Sufijo compacto para distinguir escenarios en los archivos guardados
    mode_tag_baseline = 'strict' if STRICT_BASELINE else 'hybrid'
    mode_tag_drift = 'drift' if (STRICT_BASELINE and SYNTHETIC_DRIFT) else 'nodrift'
    suffix = f"_{mode_tag_baseline}_{mode_tag_drift}_{ROUTE_MODE}"
    summary_csv = results_dir / f'leaveoneout_summary{suffix}.csv'
    df_summary.to_csv(summary_csv, index=False)

    print(f"\n{'='*70}")
    print(f"RESUMEN GLOBAL "
          f"[baseline={mode_tag_baseline} | drift={'ON' if SYNTHETIC_DRIFT and STRICT_BASELINE else 'OFF'} | rutas={ROUTE_MODE}]")
    print(f"{'='*70}")
    print(df_summary.to_string(index=False))
    print()
    print(f"Baseline RMSE 2D media   : {df_summary['baseline_rmse_m'].mean():.3f} m")
    print(f"Modelo   RMSE 2D media   : {df_summary['model_rmse_m'].mean():.3f} m")
    print(f"Mejora media             : {df_summary['improvement_pct'].mean():+.1f}%")
    print(f"Folds que superan {SMART_THRESHOLD_PCT}% : "
          f"{df_summary['smart_ok'].sum()}/{len(df_summary)}")
    print(f"\nTiempo total: {(time.time() - t_start)/60:.1f} min")
    print(f"Resumen guardado en: {summary_csv}")

    # Gráficas resumen
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Barras: baseline vs modelo por fold
        x = np.arange(len(df_summary))
        w = 0.35
        axes[0].bar(x - w/2, df_summary['baseline_rmse_m'], w,
                    label='Baseline', color='#3b7dd6')
        axes[0].bar(x + w/2, df_summary['model_rmse_m'], w,
                    label='Bi-LSTM', color='#e07b3a')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(df_summary['test_route'], rotation=45,
                                ha='right', fontsize=8)
        axes[0].set_ylabel('RMSE 2D [m]')
        axes[0].set_title('RMSE por fold (test ruta)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Barras: mejora porcentual por fold
        colors = ['#4a9c4a' if x >= SMART_THRESHOLD_PCT else '#cc4a4a'
                  for x in df_summary['improvement_pct']]
        axes[1].bar(x, df_summary['improvement_pct'], color=colors)
        axes[1].axhline(SMART_THRESHOLD_PCT, ls='--', color='k',
                        label=f'umbral SMART ({SMART_THRESHOLD_PCT}%)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(df_summary['test_route'], rotation=45,
                                ha='right', fontsize=8)
        axes[1].set_ylabel('Mejora [%]')
        axes[1].set_title('Mejora % Bi-LSTM vs Baseline')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = results_dir / 'figures' / f'leaveoneout_summary{suffix}.png'
        plt.savefig(fig_path, dpi=120, bbox_inches='tight')
        print(f"Gráfica guardada en: {fig_path}")
    except Exception as e:
        print(f"(Aviso: gráfica no generada: {e})")


if __name__ == '__main__':
    main()
