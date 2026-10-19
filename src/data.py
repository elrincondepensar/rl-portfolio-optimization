# src/data.py
"""
Carga, validación y limpieza de datos diarios con yfinance.
Activos: SPY, QQQ, TLT, GLD. Periodo: 2015-2025. Ajuste: auto_adjust=False.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = ["SPY", "QQQ", "TLT", "GLD"]
START = "2015-01-01"
END = "2025-12-31"
RAW_DIR = Path("data/raw")
RAW_CSV = RAW_DIR / "prices_raw.csv"
MAX_RETRIES = 3


def _download_with_retry(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Un solo batch + retry con backoff exponencial por rate limiting."""
    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if df is not None and not df.empty:
                return df
            logger.warning("Descarga vacía (intento %d/%d)", attempt + 1, MAX_RETRIES)
        except Exception as exc:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            logger.warning("Rate limit o error: %s. Esperando %ds...", exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"yfinance falló tras {MAX_RETRIES} intentos.")


def _flatten_multiindex(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Convierte el MultiIndex de yfinance a formato largo: Date | Ticker | OHLCV."""
    records = []
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                logger.warning("Ticker %s no encontrado en la descarga.", t)
                continue
            sub = raw[t].copy().reset_index()
            sub["Ticker"] = t
            records.append(sub)
    else:
        raw = raw.reset_index()
        raw["Ticker"] = tickers[0]
        records.append(raw)

    if not records:
        raise ValueError("Ningún ticker descargado correctamente.")

    df = pd.concat(records, ignore_index=True)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validación: tipos, duplicados, NaN en Close y cobertura temporal."""
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Duplicados
    n_dup = df.duplicated(subset=["Ticker", "Date"]).sum()
    if n_dup:
        logger.warning("Eliminando %d duplicados Ticker+Date.", n_dup)
        df = df.drop_duplicates(subset=["Ticker", "Date"])

    # NaN en Close / Adj_Close
    for col in ["Close", "Adj_Close"]:
        n_nan = df[col].isna().sum()
        if n_nan:
            logger.warning("Rellenando %d NaN en %s con forward fill.", n_nan, col)
            df[col] = df.groupby("Ticker")[col].ffill()

    # Cobertura: cada ticker debe tener al menos 2400 sesiones
    counts = df.groupby("Ticker")["Date"].count()
    for t, c in counts.items():
        if c < 2400:
            logger.warning("Ticker %s solo tiene %d sesiones (<2400).", t, c)

    return df


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Carga desde CSV si existe; si no, descarga y guarda."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists() and not force_download:
        logger.info("Cargando datos desde %s", RAW_CSV)
        df = pd.read_csv(RAW_CSV, parse_dates=["Date"])
        return df

    logger.info("Descargando %s desde yfinance...", TICKERS)
    raw = _download_with_retry(TICKERS, START, END)
    df = _flatten_multiindex(raw, TICKERS)
    df = _validate(df)
    df.to_csv(RAW_CSV, index=False)
    logger.info("Guardado: %s (%d filas)", RAW_CSV, len(df))
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columna log_ret = ln(P_t / P_{t-1}) por ticker."""
    df = df.copy()
    df["log_ret"] = (
        df.groupby("Ticker")["Adj_Close"]
        .transform(lambda s: np.log(s / s.shift(1)))
    )
    return df


def compute_rolling_vol(df: pd.DataFrame, windows: list[int] = [21, 63]) -> pd.DataFrame:
    """Añade volatilidad rolling anualizada (21d≈1m, 63d≈3m)."""
    df = df.copy()
    for w in windows:
        col = f"vol_{w}d"
        df[col] = (
            df.groupby("Ticker")["log_ret"]
            .transform(lambda s: s.rolling(w).std() * np.sqrt(252))
        )
    return df


def build_dataset(force_download: bool = False) -> pd.DataFrame:
    """Pipeline completo: raw → log returns → rolling vol."""
    df = load_raw(force_download=force_download)
    df = compute_log_returns(df)
    df = compute_rolling_vol(df)
    return df