# src/prepare_rl_data.py
"""
Prepara el dataset para RL: features + predicciones de modelos base + vol GARCH.
Salida: data/processed/rl_dataset.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _compute_garch_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute vol_forecast por fold. Tarda ~10s."""
    from arch import arch_model
    from models_predictive import make_walk_forward_folds

    df = df.copy()
    df["vol_forecast_garch"] = np.nan

    folds = make_walk_forward_folds(df["Date"])
    for fold in folds:
        train = df[df["Date"] <= fold.train_end]
        test_mask = (df["Date"] >= fold.test_start) & (df["Date"] <= fold.test_end)
        n_days_test = df.loc[test_mask, "Date"].nunique()

        for t in df["Ticker"].unique():
            tr = train[train["Ticker"] == t].dropna(subset=["log_ret"]).sort_values("Date")
            if len(tr) < 500:
                continue
            returns = (tr["log_ret"].values * 100).astype(float)
            try:
                am = arch_model(returns, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
                res = am.fit(disp="off", show_warning=False)
                sigma = np.sqrt(
                    np.maximum(res.forecast(horizon=n_days_test, reindex=False).variance.values[-1], 0.0)
                ) / 100.0
                mask_t = test_mask & (df["Ticker"] == t)
                if mask_t.sum() == len(sigma):
                    df.loc[mask_t, "vol_forecast_garch"] = sigma
            except Exception as e:
                print(f"GARCH fallo en {t}/{fold.fold_id}: {e}")
    df["vol_forecast_garch"] = df["vol_forecast_garch"].fillna(df["vol_63d"])
    return df


def build_rl_dataset(force_recompute_garch: bool = False) -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"

    features = pd.read_csv(proc / "features.csv", parse_dates=["Date"])
    preds = pd.read_csv(proc / "predictions_walkforward.csv", parse_dates=["Date"])

    # Pivot predicciones wide
    preds_wide = preds.pivot_table(
        index=["Date", "Ticker"], columns="model", values="pred"
    ).reset_index()
    preds_wide.columns.name = None
    preds_wide = preds_wide.rename(columns={c: f"pred_{c}" for c in preds_wide.columns if c not in ("Date", "Ticker")})

    df = features.merge(preds_wide, on=["Date", "Ticker"], how="left")

    # GARCH vol (recompute o cargar)
    garch_path = proc / "garch_vol_forecast.csv"
    if garch_path.exists() and not force_recompute_garch:
        vol = pd.read_csv(garch_path, parse_dates=["Date"])
        df = df.merge(vol, on=["Date", "Ticker"], how="left")
    else:
        df = _compute_garch_vol(df)
        df[["Date", "Ticker", "vol_forecast_garch"]].to_csv(garch_path, index=False)

    # Target
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df["target_ret_1d"] = df.groupby("Ticker")["log_ret"].shift(-1)
    df = df.dropna(subset=["target_ret_1d"]).reset_index(drop=True)

    # Recortar a partir del primer dia con predicciones (fold 0 test)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    has_pred = df[pred_cols].notna().any(axis=1)
    first_pred_date = df.loc[has_pred, "Date"].min()
    df = df[df["Date"] >= first_pred_date].reset_index(drop=True)

    # Rellenar NaN residuales en predicciones con 0
    for c in pred_cols:
        df[c] = df[c].fillna(0.0)

    out = proc / "rl_dataset.csv"
    df.to_csv(out, index=False)
    print(f"Guardado: {out} | shape: {df.shape}")
    print(f"Rango: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"Columnas: {list(df.columns)}")
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    build_rl_dataset(force_recompute_garch=True)