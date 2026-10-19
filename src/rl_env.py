# src/rl_env.py
"""
Entorno Gymnasium para trading multi-activo.
Soporta modo 4D (sin cash explicito) y modo 5D (cash como 5o activo).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


FEATURES_A = ["log_ret", "rsi_14", "macd_hist_norm", "vol_63d", "dist_sma_200"]
FEATURES_B = FEATURES_A + ["pred_pymc", "pred_ridge", "pred_contrarian", "pred_gru", "vol_forecast_garch"]
FEATURES_C = ["log_ret", "ret_lag_1", "vol_63d"]

FEATURE_SETS = {"A": FEATURES_A, "B": FEATURES_B, "C": FEATURES_C}


@dataclass
class EnvConfig:
    episode_len: int = 252
    window: int = 20
    initial_capital: float = 1.0
    cost_bps: float = 10.0
    lambda_dd: float = 0.1
    max_gross: float = 1.0
    borrow_rate_annual: float = 0.005
    mu_turnover: float = 0.0
    use_cash: bool = False   # <-- NUEVO


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, tickers: list[str], feature_cols: list[str], config: EnvConfig):
        super().__init__()
        self.cfg = config
        self.tickers = list(tickers)
        self.n_assets = len(tickers)
        self.n_feat = len(feature_cols)
        self.feature_cols = list(feature_cols)
        self.use_cash = bool(config.use_cash)

        df = df.copy()
        df = df[df["Ticker"].isin(self.tickers)].sort_values(["Date", "Ticker"]).reset_index(drop=True)
        self.dates = np.sort(df["Date"].unique())
        self.T = len(self.dates)

        feat = np.zeros((self.T, self.n_assets, self.n_feat), dtype=np.float32)
        rets = np.zeros((self.T, self.n_assets), dtype=np.float32)
        ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}
        date_to_idx = {d: i for i, d in enumerate(self.dates)}

        for _, row in df.iterrows():
            ti = date_to_idx[row["Date"]]
            ai = ticker_to_idx[row["Ticker"]]
            feat[ti, ai] = row[self.feature_cols].values.astype(np.float32)
            rets[ti, ai] = float(row["target_ret_1d"])

        self.features = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        self.returns = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Espacios ---
        action_dim = self.n_assets + (1 if self.use_cash else 0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        # Obs: W*A*F + pesos drifted (A) + [cash_weight si use_cash] + V/V0
        obs_dim = self.cfg.window * self.n_assets * self.n_feat + self.n_assets + 1
        if self.use_cash:
            obs_dim += 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Estado
        self.t = 0
        self.start_idx = 0
        self.portfolio_value = self.cfg.initial_capital
        self.drifted_weights = np.zeros(self.n_assets, dtype=np.float32)  # solo riesgosos
        self.cash_weight = 0.0
        self.peak_value = self.cfg.initial_capital
        self.prev_dd = 0.0
        self.step_count = 0

    # ---------- helpers ----------

    def _enforce_risky_constraint(self, w_risky: np.ndarray) -> np.ndarray:
        gross = float(np.sum(np.abs(w_risky)))
        if gross > self.cfg.max_gross and gross > 1e-8:
            w_risky = w_risky * (self.cfg.max_gross / gross)
        return w_risky.astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        start = max(0, self.t - self.cfg.window + 1)
        window = self.features[start : self.t + 1]
        if window.shape[0] < self.cfg.window:
            pad = np.zeros((self.cfg.window - window.shape[0], self.n_assets, self.n_feat), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)

        obs_feats = window.reshape(-1)
        parts = [
            obs_feats,
            self.drifted_weights.astype(np.float32),
        ]
        if self.use_cash:
            parts.append(np.array([self.cash_weight], dtype=np.float32))
        parts.append(np.array([self.portfolio_value / self.cfg.initial_capital - 1.0], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    # ---------- gym API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "start_idx" in options:
            self.start_idx = int(options["start_idx"])
        else:
            lo = self.cfg.window - 1
            hi = max(lo + 1, self.T - self.cfg.episode_len - 1)
            self.start_idx = int(self.np_random.integers(lo, hi))

        self.t = self.start_idx
        self.portfolio_value = self.cfg.initial_capital
        self.drifted_weights = np.zeros(self.n_assets, dtype=np.float32)
        self.cash_weight = 0.0
        self.peak_value = self.cfg.initial_capital
        self.prev_dd = 0.0
        self.step_count = 0

        return self._get_obs(), {"start_idx": self.start_idx}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        # --- Separar riesgosos y cash ---
        if self.use_cash:
            w_risky_raw = action[: self.n_assets]
            w_cash_raw = float(action[self.n_assets])
        else:
            w_risky_raw = action
            w_cash_raw = None

        # --- Constraint sobre riesgosos ---
        w_risky = self._enforce_risky_constraint(w_risky_raw)
        gross_risky = float(np.sum(np.abs(w_risky)))

        # --- Cash explicito (solo use_cash) ---
        if self.use_cash:
            w_cash = float(np.clip(w_cash_raw, 0.0, 1.0))
            # constraint: gross_risky + w_cash <= max_gross
            total = gross_risky + w_cash
            if total > self.cfg.max_gross and total > 1e-8:
                scale = self.cfg.max_gross / total
                w_risky = (w_risky * scale).astype(np.float32)
                w_cash = w_cash * scale
                gross_risky = float(np.sum(np.abs(w_risky)))
        else:
            w_cash = max(0.0, 1.0 - gross_risky)  # residual implicito

        # --- Retorno del portfolio ---
        r_next = self.returns[self.t]  # (A,)
        port_log_ret_gross = float(np.sum(w_risky * r_next))  # cash aporta 0

        # --- Costes ---
        turnover_risky = float(np.sum(np.abs(w_risky - self.drifted_weights)))
        # turnover de cash: cambio entre el cash target y el drifted actual
        prev_cash_drifted = self.cash_weight if self.use_cash else max(0.0, 1.0 - float(np.sum(np.abs(self.drifted_weights))))
        turnover_cash = abs(w_cash - prev_cash_drifted) if self.use_cash else 0.0
        turnover = turnover_risky + turnover_cash

        cost = self.cfg.cost_bps * 1e-4 * turnover
        short_exposure = float(np.sum(np.maximum(-w_risky, 0.0)))
        borrow = self.cfg.borrow_rate_annual / 252.0 * short_exposure

        net_log_ret = port_log_ret_gross - cost - borrow
        new_value = self.portfolio_value * np.exp(net_log_ret)

        # --- Drawdown ---
        self.peak_value = max(self.peak_value, new_value)
        dd = (self.peak_value - new_value) / max(self.peak_value, 1e-8)
        dd_increment = max(0.0, dd - self.prev_dd)

        # --- Reward ---
        reward = net_log_ret - self.cfg.lambda_dd * dd_increment - self.cfg.mu_turnover * turnover

        # --- Drift de pesos ---
        simple_r = np.exp(r_next) - 1.0
        risky_values = w_risky * (1.0 + simple_r)   # valor por activo tras retorno
        cash_value = w_cash                          # cash no cambia
        total_value = float(np.sum(risky_values) + cash_value)
        if abs(total_value) < 1e-8:
            total_value = 1.0
        self.drifted_weights = (risky_values / total_value).astype(np.float32)
        self.cash_weight = float(cash_value / total_value)

        # --- Avance ---
        self.portfolio_value = new_value
        self.prev_dd = dd
        self.t += 1
        self.step_count += 1

        truncated = (self.step_count >= self.cfg.episode_len) or (self.t >= self.T - 1)
        terminated = False

        info = {
            "portfolio_value": float(new_value),
            "log_ret": float(net_log_ret),
            "turnover": float(turnover),
            "turnover_risky": float(turnover_risky),
            "turnover_cash": float(turnover_cash),
            "cost": float(cost),
            "borrow": float(borrow),
            "drawdown": float(dd),
            "dd_increment": float(dd_increment),
            "weights": self.drifted_weights.copy(),
            "cash_weight": float(self.cash_weight),
            "gross_risky": float(np.sum(np.abs(self.drifted_weights))),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self):
        pass