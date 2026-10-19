# src/models_predictive.py
"""
Modelos predictivos con validacion walk-forward.
Target: retorno logaritmico t+1 por activo.
Firma comun: (train_df, test_df, features) -> pd.Series(index=test_df.index)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ==================== CONJUNTOS DE FEATURES ====================

LINEAR_FEATURES = [
    "log_ret",                                        # <-- NUEVO: y_t
    "rsi_14", "macd_hist_norm",
    "ret_lag_1", "ret_lag_2", "ret_lag_5",            # y_{t-1}, y_{t-2}, y_{t-5}
    "vol_63d", "vol_zscore_20", "vol_regime_high",
    "dist_sma_200",
]

ALL_FEATURES = [
    "log_ret",                                        # <-- NUEVO
    "rsi_14",
    "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "bb_pct_b", "bb_bandwidth",
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5", "ret_lag_10", "ret_lag_20",
    "mom_5d", "mom_10d", "mom_20d",
    "vol_21d", "vol_63d",
    "vol_ratio_20", "vol_zscore_20",
    "vol_regime_high", "trend_up", "dist_sma_200",
]


# ==================== WALK-FORWARD ====================

@dataclass
class Fold:
    fold_id: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_folds(
    dates: pd.Series,
    train_min_days: int = 756,
    test_days: int = 252,
    step_days: int = 252,
) -> list[Fold]:
    """Ventana expansiva. 3 anios de train minimo, 1 anio de test, 1 anio de paso."""
    unique = np.sort(dates.unique())
    n = len(unique)
    folds: list[Fold] = []
    start_test = train_min_days
    fid = 0
    while start_test + test_days <= n:
        train_end = unique[start_test - 1]
        test_start = unique[start_test]
        test_end = unique[min(start_test + test_days - 1, n - 1)]
        folds.append(Fold(
            fid,
            pd.Timestamp(train_end),
            pd.Timestamp(test_start),
            pd.Timestamp(test_end),
        ))
        fid += 1
        start_test += step_days
    return folds


# ==================== BASELINES ====================

def predict_naive(train_df, test_df, features):
    """Media del target en train, constante por ticker."""
    mu = train_df.groupby("Ticker")["target_ret_1d"].mean()
    out = pd.Series(index=test_df.index, dtype=float)
    for t, val in mu.items():
        out.loc[test_df["Ticker"] == t] = val
    return out


def predict_contrarian(train_df, test_df, features):
    """OLS target ~ log_ret. Coef negativo => mean reversion lag-1."""
    out = pd.Series(index=test_df.index, dtype=float)
    coefs = {}
    for t in test_df["Ticker"].unique():
        tr = train_df[train_df["Ticker"] == t].dropna(subset=["log_ret", "target_ret_1d"])
        te = test_df[test_df["Ticker"] == t].dropna(subset=["log_ret"])
        if len(tr) < 100 or len(te) == 0:
            continue
        x = tr["log_ret"].values.astype(float)
        y = tr["target_ret_1d"].values.astype(float)
        X = np.column_stack([np.ones_like(x), x])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        coefs[t] = float(coef[1])
        out.loc[te.index] = coef[0] + coef[1] * te["log_ret"].values
    out.attrs["coef_lag1"] = coefs
    return out


def predict_ridge(train_df, test_df, features, alpha=1.0):
    """Ridge con estandarizacion, un modelo por ticker."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    out = pd.Series(index=test_df.index, dtype=float)
    for t in test_df["Ticker"].unique():
        tr = train_df[train_df["Ticker"] == t].dropna(subset=features + ["target_ret_1d"])
        te = test_df[test_df["Ticker"] == t].dropna(subset=features)
        if len(tr) < 100 or len(te) == 0:
            continue
        sc = StandardScaler().fit(tr[features])
        model = Ridge(alpha=alpha).fit(sc.transform(tr[features]), tr["target_ret_1d"])
        out.loc[te.index] = model.predict(sc.transform(te[features]))
    return out


# ==================== SARIMAX ====================

def predict_sarimax(train_df, test_df, features, order=(1, 0, 1)):
    """
    SARIMAX por ticker con fallback de orden y sanity check de predicciones.
    Rechaza forecasts cuya magnitud supere 10x la std del target en train.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    out = pd.Series(index=test_df.index, dtype=float)
    orders_to_try = [order, (1, 0, 0), (0, 0, 0)]

    for t in test_df["Ticker"].unique():
        tr = train_df[train_df["Ticker"] == t].dropna(subset=features + ["target_ret_1d"]).sort_values("Date")
        te = test_df[test_df["Ticker"] == t].dropna(subset=features).sort_values("Date")
        if len(tr) < 200 or len(te) == 0:
            continue

        y_train = tr["target_ret_1d"].values
        std_train = float(np.std(y_train))
        # Umbral: 10x la desviacion tipica diaria. Un retorno diario realista nunca lo supera.
        threshold = max(10.0 * std_train, 1e-3)

        fitted = False
        for ord_try in orders_to_try:
            try:
                model = SARIMAX(
                    y_train,
                    exog=tr[features].values,
                    order=ord_try,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=500)
                forecast = res.forecast(steps=len(te), exog=te[features].values)

                # Sanity check: valores finitos y magnitud razonable
                if not np.all(np.isfinite(forecast)):
                    logger.warning("SARIMAX %s orden %s: forecast no finito. Rechazado.", t, ord_try)
                    continue
                if np.max(np.abs(forecast)) > threshold:
                    logger.warning(
                        "SARIMAX %s orden %s: forecast absurdo (max=%.4f > umbral=%.4f). Rechazado.",
                        t, ord_try, np.max(np.abs(forecast)), threshold,
                    )
                    continue

                out.loc[te.index] = forecast
                fitted = True
                break
            except Exception as e:
                logger.warning("SARIMAX %s orden %s fallo: %s", t, ord_try, e)

        if not fitted:
            # Fallback final: prediccion 0 (equivale a naive centrado)
            logger.warning("SARIMAX %s: todos los ordenes rechazados. Fallback a 0.", t)
            out.loc[te.index] = 0.0
    return out


# ==================== GARCH ====================

def predict_garch(train_df, test_df, features):
    """
    AR(1)-GARCH(1,1). Prediccion recursiva del retorno condicional en test.
    Varianza condicional multi-paso guardada en attrs['vol_forecast'].
    """
    from arch import arch_model

    out = pd.Series(index=test_df.index, dtype=float)
    vol_out = pd.Series(index=test_df.index, dtype=float)

    for t in test_df["Ticker"].unique():
        tr = train_df[train_df["Ticker"] == t].dropna(subset=["log_ret"]).sort_values("Date")
        te = test_df[test_df["Ticker"] == t].sort_values("Date")
        if len(tr) < 500 or len(te) == 0:
            continue

        try:
            returns = (tr["log_ret"].values * 100).astype(float)
            am = arch_model(returns, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)

            mu_const = float(res.params.get("Const", 0.0))
            phi = float(res.params.get("ar.L1", 0.0))

            # Varianza condicional multi-paso
            try:
                sigma_fc = res.forecast(horizon=len(te), reindex=False).variance.values[-1]
                sigma_fc = np.sqrt(np.maximum(sigma_fc, 0.0)) / 100.0
            except Exception:
                sigma_fc = np.full(len(te), np.nan)

            last_ret = float(returns[-1])
            preds_te = []
            for _ in range(len(te)):
                pred = (mu_const + phi * last_ret) / 100.0
                preds_te.append(pred)
                last_ret = pred * 100.0

            out.loc[te.index] = preds_te
            vol_out.loc[te.index] = sigma_fc
        except Exception as e:
            logger.warning("GARCH fallo en %s: %s", t, e)

    out.attrs["vol_forecast"] = vol_out
    return out


# ==================== PyMC ====================

def predict_pymc(
    train_df, test_df, features,
    draws=1000, tune=500, chains=4, random_seed=42,
):
    """
    Modelo jerarquico con pooling parcial entre los 4 activos.
    Non-centered parametrization para evitar divergencias.
    """
    import pymc as pm

    tr = train_df.dropna(subset=features + ["target_ret_1d"]).copy()
    te = test_df.dropna(subset=features).copy()
    if len(tr) < 500 or len(te) == 0:
        return pd.Series(index=test_df.index, dtype=float)

    tickers = sorted(tr["Ticker"].unique())
    tmap = {t: i for i, t in enumerate(tickers)}
    idx_tr = tr["Ticker"].map(tmap).values
    idx_te = te["Ticker"].map(tmap).values

    X_tr = tr[features].values.astype(float)
    y_tr = tr["target_ret_1d"].values.astype(float)

    mu_x = X_tr.mean(axis=0)
    sd_x = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mu_x) / sd_x
    X_te_s = (te[features].values.astype(float) - mu_x) / sd_x

    with pm.Model():
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=0.005)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.005)

        # Non-centered
        alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, shape=len(tickers))
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)

        beta = pm.Normal("beta", mu=0.0, sigma=0.005, shape=len(features))
        sigma = pm.HalfNormal("sigma", sigma=0.01)

        mu = alpha[idx_tr] + pm.math.dot(X_tr_s, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_tr)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=random_seed, progressbar=False,
            target_accept=0.95, cores=min(chains, 4),
        )

    beta_post = trace.posterior["beta"].values.reshape(-1, len(features)).mean(axis=0)
    alpha_post = trace.posterior["alpha"].values.reshape(-1, len(tickers)).mean(axis=0)

    out = pd.Series(index=test_df.index, dtype=float)
    out.loc[te.index] = alpha_post[idx_te] + X_te_s @ beta_post
    return out


# ==================== GRU ====================

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """X_seq[k] = X[k : k+seq_len], y_seq[k] = y[k+seq_len-1]."""
    if len(X) < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xs = np.stack([X[k : k + seq_len] for k in range(len(X) - seq_len + 1)])
    ys = y[seq_len - 1:]
    return xs.astype(np.float32), ys.astype(np.float32)


def predict_gru(
    train_df, test_df, features,
    seq_len=20, hidden=32, epochs=30, batch_size=64,
    lr=1e-3, device="cuda", random_seed=42,
):
    """
    GRU 1 capa + dropout 0.2. Un modelo por ticker.
    Predice target_ret_1d[t] con ventana features[t-seq_len+1 : t+1].
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")

    class GRUNet(nn.Module):
        def __init__(self, n_feat, hidden):
            super().__init__()
            self.gru = nn.GRU(n_feat, hidden, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

    out = pd.Series(index=test_df.index, dtype=float)

    for t in test_df["Ticker"].unique():
        tr = (
            train_df[train_df["Ticker"] == t]
            .dropna(subset=features + ["target_ret_1d"])
            .sort_values("Date")
        )
        te = (
            test_df[test_df["Ticker"] == t]
            .dropna(subset=features)
            .sort_values("Date")
        )
        if len(tr) < seq_len + 100 or len(te) == 0:
            continue

        # Normalizacion con stats de train
        mu_x = tr[features].mean().values
        sd_x = tr[features].std().values + 1e-8

        X_tr = ((tr[features].values - mu_x) / sd_x).astype(np.float32)
        y_tr = tr["target_ret_1d"].values.astype(np.float32)
        X_te = ((te[features].values - mu_x) / sd_x).astype(np.float32)

        # Secuencias de train
        xs, ys = _build_sequences(X_tr, y_tr, seq_len)
        if len(xs) < 100:
            continue

        ds = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        net = GRUNet(len(features), hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        net.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()

        # Construccion de secuencias de test con contexto de train
        X_all = np.concatenate([X_tr, X_te], axis=0)
        n_tr = len(X_tr)
        seqs = []
        positions = []  # posiciones dentro de te
        for i in range(n_tr, len(X_all)):
            start = i - seq_len + 1
            if start < 0:
                continue
            seqs.append(X_all[start : i + 1])
            positions.append(i - n_tr)

        if not seqs:
            continue

        net.eval()
        with torch.no_grad():
            X_in = torch.from_numpy(np.stack(seqs).astype(np.float32)).to(dev)
            preds_arr = net(X_in).cpu().numpy()

        # positions son indices posicionales en te. Preservamos index original.
        te_aligned = te.iloc[positions]
        out.loc[te_aligned.index] = preds_arr

    return out


# ==================== WRAPPER SEGURO ====================

def safe_predict(name, fn, *args, **kwargs):
    """Ejecuta un modelo capturando errores. Devuelve (name, series_o_None, tiempo_s)."""
    import time
    t0 = time.time()
    try:
        res = fn(*args, **kwargs)
        dt = time.time() - t0
        n_ok = int(res.notna().sum()) if res is not None else 0
        n_tot = len(args[1]) if len(args) > 1 else 0
        print(f"  {name:<12} OK   {dt:6.1f}s   cobertura={n_ok}/{n_tot}")
        return name, res, dt
    except Exception as e:
        dt = time.time() - t0
        print(f"  {name:<12} FALLO {dt:6.1f}s   {type(e).__name__}: {e}")
        return name, None, dt