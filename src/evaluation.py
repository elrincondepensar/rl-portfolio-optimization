# src/evaluation.py
"""
Fase 7: evaluacion comparativa de modelos RL + benchmarks.
- Reconstruccion de curvas de equity
- Metricas de riesgo ajustadas (Sharpe, Sortino, Calmar, Max DD)
- Bootstrap IC95%
- Analisis por regimen
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rl_env import TradingEnv, EnvConfig, FEATURE_SETS


TICKERS = ["GLD", "QQQ", "SPY", "TLT"]
TEST_START = "2024-10-22"
TEST_END = "2025-10-23"
WINDOW = 20


# ==================== METRICAS ====================

def compute_metrics(daily_returns: np.ndarray, turnover: np.ndarray | None = None) -> dict:
    """Metricas estandar sobre retornos diarios (simples)."""
    r = np.asarray(daily_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return {}
    equity = np.cumprod(1 + r)
    total_ret = equity[-1] - 1.0
    ann_ret = (1 + total_ret) ** (252 / len(r)) - 1.0
    ann_vol = r.std() * np.sqrt(252)
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(252)

    # Sortino
    downside = r[r < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-12
    sortino = r.mean() / (downside_std + 1e-12) * np.sqrt(252)

    # Max DD
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(dd.max())
    calmar = ann_ret / max_dd if max_dd > 1e-8 else np.nan

    out = {
        "n_days": len(r),
        "total_return": float(total_ret),
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": max_dd,
        "calmar": float(calmar) if not np.isnan(calmar) else None,
        "hit_ratio": float((r > 0).mean()),
    }
    if turnover is not None:
        out["turnover_mean"] = float(np.mean(turnover))
    return out


# ==================== RL AGENT ====================

def evaluate_rl_agent(model_path: Path, df: pd.DataFrame, features: str,
                     leverage: float = 1.0, lambda_dd: float = 0.1) -> dict:
    """Carga un modelo PPO y evalua en test. Devuelve serie de retornos diarios y metricas."""
    model = PPO.load(str(model_path))
    test_df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].reset_index(drop=True)

    cfg = EnvConfig(
        episode_len=100000, window=WINDOW, lambda_dd=lambda_dd,
        max_gross=leverage, cost_bps=10.0, borrow_rate_annual=0.005, mu_turnover=0.0,
    )
    env = TradingEnv(df=test_df, tickers=TICKERS, feature_cols=FEATURE_SETS[features], config=cfg)
    obs, _ = env.reset(seed=999, options={"start_idx": WINDOW - 1})

    log_rets, turnovers, weights_hist, dates = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        log_rets.append(info["log_ret"])
        turnovers.append(info["turnover"])
        weights_hist.append(info["weights"].copy())
        dates.append(env.dates[min(env.t, len(env.dates) - 1)])
        done = term or trunc

    log_rets = np.array(log_rets)
    simple_rets = np.exp(log_rets) - 1.0  # equivalente al retorno diario neto del portfolio
    metrics = compute_metrics(simple_rets, np.array(turnovers))
    return {
        "name": f"PPO-{features}",
        "returns": simple_rets,
        "turnovers": np.array(turnovers),
        "weights": np.array(weights_hist),
        "dates": dates,
        "metrics": metrics,
    }


# ==================== BENCHMARKS ====================

def benchmark_returns(df: pd.DataFrame, kind: str) -> dict:
    """Devuelve retornos diarios de un benchmark sobre el mismo periodo de test."""
    test_df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].copy()
    test_dates = sorted(test_df["Date"].unique())[WINDOW - 1:]  # alinear con RL

    R = np.zeros((len(test_dates), len(TICKERS)))
    for i, d in enumerate(test_dates):
        day = test_df[test_df["Date"] == d]
        log_rets = day.set_index("Ticker")["target_ret_1d"].reindex(TICKERS).values
        R[i] = np.exp(log_rets) - 1.0

    if kind == "bh_eq":
        daily = 0.25 * R.sum(axis=1)
        name = "Buy-and-Hold eq."
    elif kind == "gld":
        daily = R[:, 0]
        name = "Always long GLD"
    elif kind == "spy":
        daily = R[:, 2]
        name = "Always long SPY"
    elif kind == "qqq":
        daily = R[:, 1]
        name = "Always long QQQ"
    elif kind == "tlt":
        daily = R[:, 3]
        name = "Always long TLT"
    else:
        raise ValueError(kind)

    return {"name": name, "returns": daily, "metrics": compute_metrics(daily)}


# ==================== BOOTSTRAP ====================

def bootstrap_sharpe(returns: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """IC95% del Sharpe via bootstrap con reemplazo."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        sharpes[i] = sample.mean() / (sample.std() + 1e-12) * np.sqrt(252)
    return {
        "sharpe_mean": float(sharpes.mean()),
        "sharpe_ci_low": float(np.percentile(sharpes, 2.5)),
        "sharpe_ci_high": float(np.percentile(sharpes, 97.5)),
        "sharpe_std": float(sharpes.std()),
    }


# ==================== REGIMENES ====================

def regime_analysis(returns: np.ndarray, dates: list, df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide los retornos del test en regimenes:
    - Alta/baja volatilidad (percentil 70 de vol_21d media de los 4 activos)
    - Alcista/bajista (signo del retorno rolling 60d del benchmark equiponderado)
    """
    test_df = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)]
    vol_mean = test_df.groupby("Date")["vol_21d"].mean().reindex(dates).values
    thresh_vol = np.nanpercentile(vol_mean, 70)

    # Retorno rolling 60d del bh_eq
    bh_daily = benchmark_returns(df, "bh_eq")["returns"]
    n = len(bh_daily)
    roll_ret = np.array([bh_daily[max(0, i - 60):i + 1].sum() for i in range(n)])
    bull = roll_ret > 0

    df_reg = pd.DataFrame({
        "date": dates,
        "return": returns[:len(dates)],
        "high_vol": vol_mean > thresh_vol,
        "bull": bull[:len(dates)],
    })

    rows = []
    for vol_label, vol_mask in [("Alta vol", df_reg["high_vol"]), ("Baja vol", ~df_reg["high_vol"])]:
        for trend_label, trend_mask in [("Alcista", df_reg["bull"]), ("Bajista", ~df_reg["bull"])]:
            sub = df_reg[vol_mask & trend_mask]
            if len(sub) < 5:
                continue
            r = sub["return"].values
            rows.append({
                "regimen": f"{vol_label} / {trend_label}",
                "n_dias": len(r),
                "ret_medio_diario_bps": float(r.mean() * 1e4),
                "sharpe_anualizado": float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)),
                "hit_ratio": float((r > 0).mean()),
            })
    return pd.DataFrame(rows)