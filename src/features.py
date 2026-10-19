# src/features.py
"""
Feature engineering para trading: RSI, MACD, Bollinger, momentum,
volumen relativo, lags y variables de régimen.

Regla: cada feature en t usa solo informacion <= t. Cero look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- Indicadores tecnicos ----------

def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Suavizado de Wilder = EMA con alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + n_std * sd
    lower = ma - n_std * sd
    # %B: posicion del precio dentro de las bandas (0 = lower, 1 = upper)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    # Bandwidth: ancho relativo de las bandas (volatilidad)
    bandwidth = (upper - lower) / ma.replace(0.0, np.nan)
    return upper, lower, pct_b, bandwidth


# ---------- Construccion del dataset de features ----------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade todos los features al dataset en formato largo.
    Espera columnas: Date, Ticker, Open, High, Low, Close, Adj_Close, Volume, log_ret.
    """
    df = df.copy().sort_values(["Ticker", "Date"]).reset_index(drop=True)
    out = []

    for ticker, g in df.groupby("Ticker", sort=False):
        g = g.sort_values("Date").copy()
        close = g["Adj_Close"]
        volume = g["Volume"].astype(float)

        # --- RSI ---
        g["rsi_14"] = _rsi_wilder(close, 14)

        # --- MACD ---
        macd, macd_sig, macd_hist = _macd(close, 12, 26, 9)
        # Normalizo por precio para que sea comparable entre activos
        g["macd_norm"] = macd / close
        g["macd_signal_norm"] = macd_sig / close
        g["macd_hist_norm"] = macd_hist / close

        # --- Bollinger ---
        upper, lower, pct_b, bandwidth = _bollinger(close, 20, 2.0)
        g["bb_pct_b"] = pct_b
        g["bb_bandwidth"] = bandwidth

        # --- Retornos rezagados ---
        for lag in [1, 2, 3, 5, 10, 20]:
            g[f"ret_lag_{lag}"] = g["log_ret"].shift(lag)

        # --- Momentum acumulado ---
        for w in [5, 10, 20]:
            g[f"mom_{w}d"] = g["log_ret"].rolling(w).sum()

        # --- Volumen ---
        vol_mean_20 = volume.rolling(20).mean()
        vol_std_20 = volume.rolling(20).std()
        g["vol_ratio_20"] = volume / vol_mean_20.replace(0.0, np.nan)
        g["vol_zscore_20"] = (volume - vol_mean_20) / vol_std_20.replace(0.0, np.nan)

        # --- Regimen de volatilidad ---
        # Umbral: percentil 70 de vol_21d dentro del propio ticker (expanding)
        vol_21 = g["vol_21d"]
        thr = vol_21.expanding(min_periods=252).quantile(0.70)
        g["vol_regime_high"] = (vol_21 > thr).astype(int)

        # --- Regimen de tendencia: precio vs SMA 200 ---
        sma_200 = close.rolling(200).mean()
        g["trend_up"] = (close > sma_200).astype(int)
        g["dist_sma_200"] = (close - sma_200) / sma_200.replace(0.0, np.nan)

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    result = result.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return result


FEATURE_COLS = [
    "rsi_14",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_pct_b", "bb_bandwidth",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20",
    "mom_5d", "mom_10d", "mom_20d",
    "vol_21d", "vol_63d",
    "vol_ratio_20", "vol_zscore_20",
    "vol_regime_high", "trend_up", "dist_sma_200",
]


def build_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """Pipeline: agrega features y opcionalmente elimina filas con NaN."""
    result = add_features(df)
    if drop_na:
        before = len(result)
        result = result.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        print(f"[features] Filas antes: {before}, despues (sin NaN): {len(result)}")
    return result