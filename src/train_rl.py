# src/train_rl.py
"""
Entrena PPO sobre TradingEnv. Config via CLI.

Soporta modo 4D (sin cash explicito) y modo 5D (cash como 5o activo).

Metricas reportadas:
  - Sharpe sobre retorno neto del portfolio (exp(log_ret) - 1), NO sobre reward RL.
  - Reward RL solo se usa internamente para el aprendizaje.

Uso:
    python src/train_rl.py --features A --lambda_dd 0.1 --mu_turnover 0.0 \
        --leverage 1.0 --steps 500000 --seed 42

    python src/train_rl.py --features A --lambda_dd 0.1 --mu_turnover 0.0 \
        --leverage 1.0 --steps 500000 --seed 42 --use_cash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    print("Falta stable-baselines3. Instala con: pip install stable-baselines3")
    raise

sys.path.append(str(Path(__file__).resolve().parent))
from rl_env import TradingEnv, EnvConfig, FEATURE_SETS


TICKERS = ["GLD", "QQQ", "SPY", "TLT"]

TRAIN_END  = "2023-10-19"
VAL_START  = "2023-10-20"
VAL_END    = "2024-10-21"
TEST_START = "2024-10-22"
TEST_END   = "2025-10-23"


# ==================== HELPERS ====================

def _sharpe_from_log_returns(log_rets: np.ndarray) -> float:
    """Sharpe anualizado sobre retorno neto del portfolio."""
    log_rets = np.asarray(log_rets, dtype=float)
    simple = np.exp(log_rets) - 1.0
    if simple.std() < 1e-12:
        return 0.0
    return float(simple.mean() / simple.std() * np.sqrt(252))


def _sortino_from_log_returns(log_rets: np.ndarray) -> float | None:
    log_rets = np.asarray(log_rets, dtype=float)
    simple = np.exp(log_rets) - 1.0
    downside = simple[simple < 0]
    if len(downside) < 2:
        return None
    return float(simple.mean() / (downside.std() + 1e-12) * np.sqrt(252))


# ==================== CALLBACK DE VALIDACION ====================

class ValCallback(BaseCallback):
    """Evaluacion periodica en val. Selecciona mejor modelo por Sharpe NETO."""

    def __init__(self, val_df, tickers, feature_cols, config, eval_every=20000, verbose=0):
        super().__init__(verbose)
        self.val_df = val_df
        self.tickers = tickers
        self.feature_cols = feature_cols
        self.config = config
        self.eval_every = eval_every
        self.history = []
        self.best_mean = -np.inf
        self.best_model_state = None

    def _evaluate(self):
        cfg_eval = EnvConfig(
            episode_len=100000,
            window=self.config.window,
            lambda_dd=self.config.lambda_dd,
            max_gross=self.config.max_gross,
            cost_bps=self.config.cost_bps,
            borrow_rate_annual=self.config.borrow_rate_annual,
            mu_turnover=self.config.mu_turnover,
            use_cash=self.config.use_cash,
        )
        env = TradingEnv(
            df=self.val_df, tickers=self.tickers,
            feature_cols=self.feature_cols, config=cfg_eval,
        )
        obs, _ = env.reset(seed=1234, options={"start_idx": self.config.window - 1})

        log_rets, rewards, cash_weights = [], [], []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            log_rets.append(info["log_ret"])
            rewards.append(r)
            cash_weights.append(info.get("cash_weight", 0.0))
            done = term or trunc

        log_rets = np.array(log_rets)
        rewards = np.array(rewards)
        v_final = env.portfolio_value
        sharpe_net = _sharpe_from_log_returns(log_rets)

        return {
            "val_v_final": float(v_final),
            "val_return": float(v_final - 1.0),
            "val_sharpe": float(sharpe_net),
            "val_sum_reward": float(rewards.sum()),
            "val_max_dd": float(env.prev_dd),
            "val_cash_weight_mean": float(np.mean(cash_weights)),
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_every != 0:
            return True

        stats = self._evaluate()
        stats["step"] = int(self.num_timesteps)
        self.history.append(stats)

        if self.verbose:
            msg = (
                f"  [val @ {self.num_timesteps:>7}] "
                f"ret={stats['val_return']:+.4f} "
                f"sharpe={stats['val_sharpe']:+.3f} "
                f"dd={stats['val_max_dd']:.4f}"
            )
            if self.config.use_cash:
                msg += f" cash={stats['val_cash_weight_mean']:.3f}"
            print(msg)

        if stats["val_sharpe"] > self.best_mean:
            self.best_mean = stats["val_sharpe"]
            self.best_model_state = {k: v.clone() for k, v in self.model.policy.state_dict().items()}

        return True


# ==================== ENTRENAMIENTO ====================

def train(
    rl_csv: Path,
    features: str,
    lambda_dd: float,
    mu_turnover: float,
    leverage: float,
    total_steps: int,
    seed: int,
    output_dir: Path,
    eval_every: int = 20000,
    use_cash: bool = False,
):
    df = pd.read_csv(rl_csv, parse_dates=["Date"])
    train_df = df[df["Date"] <= TRAIN_END].reset_index(drop=True)
    val_df   = df[(df["Date"] >= VAL_START) & (df["Date"] <= VAL_END)].reset_index(drop=True)
    test_df  = df[(df["Date"] >= TEST_START) & (df["Date"] <= TEST_END)].reset_index(drop=True)

    print(f"Train: {train_df['Date'].min().date()} -> {train_df['Date'].max().date()} ({train_df['Date'].nunique()} dias)")
    print(f"Val:   {val_df['Date'].min().date()} -> {val_df['Date'].max().date()} ({val_df['Date'].nunique()} dias)")
    print(f"Test:  {test_df['Date'].min().date()} -> {test_df['Date'].max().date()} ({test_df['Date'].nunique()} dias)")

    feature_cols = FEATURE_SETS[features]
    print(f"\nFeature set: {features} ({len(feature_cols)} features x 4 activos x 20 dias)")
    print(f"Config: lambda_dd={lambda_dd} | mu_turnover={mu_turnover} | max_gross={leverage} | "
          f"seed={seed} | steps={total_steps} | use_cash={use_cash}")

    cfg = EnvConfig(
        episode_len=252, window=20, lambda_dd=lambda_dd,
        max_gross=leverage, cost_bps=10.0, borrow_rate_annual=0.005,
        mu_turnover=mu_turnover, use_cash=use_cash,
    )

    train_env = Monitor(TradingEnv(
        df=train_df, tickers=TICKERS, feature_cols=feature_cols, config=cfg,
    ))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=seed,
        verbose=0,
        device="cpu",
    )

    val_cb = ValCallback(
        val_df=val_df, tickers=TICKERS, feature_cols=feature_cols,
        config=cfg, eval_every=eval_every, verbose=1,
    )

    print(f"\nEntrenando PPO...")
    model.learn(total_timesteps=total_steps, callback=val_cb, progress_bar=False)

    if val_cb.best_model_state is not None:
        model.policy.load_state_dict(val_cb.best_model_state)
        print(f"\nRestaurado mejor modelo (val_sharpe NETO = {val_cb.best_mean:+.4f})")

    # --- Guardar ---
    output_dir.mkdir(parents=True, exist_ok=True)
    # --- Formateo legible de floats en el tag ---
    def _fmt(x: float) -> str:
        s = f"{x:.10f}".rstrip("0").rstrip(".")
        return s if s else "0"

    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{features}_lam{_fmt(lambda_dd)}_mu{_fmt(mu_turnover)}_lev{_fmt(leverage)}_seed{seed}"
    if use_cash:
        tag += "_cash"
    model_path = output_dir / f"ppo_{tag}.zip"
    model.save(str(model_path))
    print(f"Modelo: {model_path}")

    hist_path = output_dir / f"ppo_{tag}_val_history.csv"
    pd.DataFrame(val_cb.history).to_csv(hist_path, index=False)
    print(f"Historial: {hist_path}")

    # --- Evaluacion final en test ---
    cfg_test = EnvConfig(
        episode_len=100000, window=20, lambda_dd=lambda_dd,
        max_gross=leverage, cost_bps=10.0, borrow_rate_annual=0.005,
        mu_turnover=mu_turnover, use_cash=use_cash,
    )
    test_env = TradingEnv(
        df=test_df, tickers=TICKERS, feature_cols=feature_cols, config=cfg_test,
    )
    obs, _ = test_env.reset(seed=999, options={"start_idx": 19})

    test_log_rets = []
    test_rewards = []
    test_turnover = []
    test_gross_exposure = []
    test_costs = []
    test_cash_weights = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = test_env.step(action)
        test_log_rets.append(info["log_ret"])
        test_rewards.append(r)
        test_turnover.append(info["turnover"])
        test_gross_exposure.append(float(np.abs(info["weights"]).sum()))
        test_costs.append(info["cost"])
        test_cash_weights.append(info.get("cash_weight", 0.0))
        done = term or trunc

    test_log_rets = np.array(test_log_rets)
    test_rewards = np.array(test_rewards)
    test_turnover = np.array(test_turnover)
    test_gross_exposure = np.array(test_gross_exposure)
    test_costs = np.array(test_costs)
    test_cash_weights = np.array(test_cash_weights)

    v_final = test_env.portfolio_value
    simple_rets = np.exp(test_log_rets) - 1.0
    sharpe_net = _sharpe_from_log_returns(test_log_rets)
    sortino_net = _sortino_from_log_returns(test_log_rets)

    print(f"\n=== TEST (out-of-sample) ===")
    print(f"  Dias ejecutados:      {len(test_log_rets)}")
    print(f"  V_final:              {v_final:.4f}  ({(v_final-1)*100:+.2f}%)")
    print(f"  Sharpe (neto):        {sharpe_net:+.3f}")
    if sortino_net is not None:
        print(f"  Sortino (neto):       {sortino_net:+.3f}")
    print(f"  Max drawdown:         {test_env.prev_dd:.4f}")
    print(f"  Sum rewards (RL):     {test_rewards.sum():+.4f}")
    print(f"  Turnover medio:       {test_turnover.mean():.4f}")
    print(f"  Gross exposure media: {test_gross_exposure.mean():.4f}")
    print(f"  Coste medio diario:   {test_costs.mean()*1e4:.2f} bps")
    if use_cash:
        print(f"  Cash weight medio:    {test_cash_weights.mean():.4f}")
        print(f"  Cash weight max:      {test_cash_weights.max():.4f}")

    # --- Meta ---
    meta = {
        "features": features,
        "n_features": len(feature_cols),
        "lambda_dd": lambda_dd,
        "mu_turnover": mu_turnover,
        "leverage": leverage,
        "seed": seed,
        "steps": total_steps,
        "use_cash": use_cash,
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "val_best_sharpe": float(val_cb.best_mean),
        "val_final_stats": val_cb.history[-1] if val_cb.history else None,
        "test_v_final": float(v_final),
        "test_return": float(v_final - 1.0),
        "test_sharpe_net": sharpe_net,
        "test_sortino_net": sortino_net,
        "test_max_dd": float(test_env.prev_dd),
        "test_sum_reward_rl": float(test_rewards.sum()),
        "test_n_days": int(len(test_log_rets)),
        "test_turnover_mean": float(test_turnover.mean()),
        "test_gross_exposure_mean": float(test_gross_exposure.mean()),
        "test_cost_mean_bps": float(test_costs.mean() * 1e4),
        "test_cash_weight_mean": float(test_cash_weights.mean()),
        "test_cash_weight_max": float(test_cash_weights.max()),
    }
    meta_path = output_dir / f"ppo_{tag}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta: {meta_path}")

    return meta


def main():
    project_root = Path(__file__).resolve().parents[1]

    p = argparse.ArgumentParser()
    p.add_argument("--rl_csv", type=str,
                   default=str(project_root / "data" / "processed" / "rl_dataset.csv"))
    p.add_argument("--features", type=str, choices=["A", "B", "C"], default="A")
    p.add_argument("--lambda_dd", type=float, default=0.1)
    p.add_argument("--mu_turnover", type=float, default=0.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_every", type=int, default=20_000)
    p.add_argument("--output_dir", type=str, default=str(project_root / "models"))
    p.add_argument("--use_cash", action="store_true",
                   help="Modo 5D: cash como quinto activo explicito.")
    args = p.parse_args()

    meta = train(
        rl_csv=Path(args.rl_csv).resolve(),
        features=args.features,
        lambda_dd=args.lambda_dd,
        mu_turnover=args.mu_turnover,
        leverage=args.leverage,
        total_steps=args.steps,
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
        eval_every=args.eval_every,
        use_cash=args.use_cash,
    )
    print("\nMeta final:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()