# scripts/train_all_seeds.py
"""
Orquestador: entrena todas las combinaciones (feature set x use_cash x mu_turnover)
con seeds 1-20. Resume-friendly.

Consolida todos los resultados en un unico CSV: reports/all_training_runs.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE     = REPORTS_DIR / "training_all_seeds.log"
MASTER_CSV   = REPORTS_DIR / "all_training_runs.csv"


# ==================== CONFIGURACION EXPERIMENTAL ====================

FEATURE_SETS = ["A", "B", "C"]
USE_CASH     = [False, True]
MU_TURNOVER_VALUES = [0.0, 0.0001, 0.00001]
LEVERAGE     = 1.0
LAMBDA_DD    = 0.1
STEPS        = 500_000
SEEDS        = list(range(1, 21))   # 1..20


# ==================== HELPERS ====================

def _fmt(x: float) -> str:
    """Formatea floats para tags consistentes con train_rl.py."""
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def _tag(features: str, use_cash: bool, mu_turnover: float, seed: int) -> str:
    tag = f"{features}_lam{_fmt(LAMBDA_DD)}_mu{_fmt(mu_turnover)}_lev{_fmt(LEVERAGE)}_seed{seed}"
    if use_cash:
        tag += "_cash"
    return tag


def model_exists(features: str, use_cash: bool, mu_turnover: float, seed: int) -> bool:
    tag = _tag(features, use_cash, mu_turnover, seed)
    return (MODELS_DIR / f"ppo_{tag}.zip").exists() and (MODELS_DIR / f"ppo_{tag}_meta.json").exists()


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def train_one(features: str, use_cash: bool, mu_turnover: float, seed: int):
    cmd = [
        sys.executable, str(PROJECT_ROOT / "src" / "train_rl.py"),
        "--features", features,
        "--lambda_dd", str(LAMBDA_DD),
        "--mu_turnover", str(mu_turnover),
        "--leverage", str(LEVERAGE),
        "--steps", str(STEPS),
        "--seed", str(seed),
    ]
    if use_cash:
        cmd.append("--use_cash")

    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    dt = time.time() - t0
    if result.returncode == 0:
        return True, dt, None
    return False, dt, (result.stderr or "")[-2000:]


# ==================== MASTER CSV ====================

MASTER_COLS = [
    "timestamp", "model_file",
    "features", "n_features", "use_cash", "leverage",
    "lambda_dd", "mu_turnover", "seed", "steps",
    "val_best_sharpe", "val_final_sharpe",
    "test_sharpe_net", "test_sortino_net", "test_return",
    "test_max_dd", "test_turnover_mean",
    "test_gross_exposure_mean", "test_cash_weight_mean",
    "test_cost_mean_bps", "test_n_days",
    "test_v_final", "test_sum_reward_rl",
]


def meta_to_row(meta: dict, model_file: str) -> dict:
    """Convierte un meta JSON a fila del CSV unificado."""
    val_stats = meta.get("val_final_stats") or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_file": model_file,
        "features": meta.get("features"),
        "n_features": meta.get("n_features"),
        "use_cash": bool(meta.get("use_cash", False)),
        "leverage": meta.get("leverage"),
        "lambda_dd": meta.get("lambda_dd"),
        "mu_turnover": meta.get("mu_turnover"),
        "seed": meta.get("seed"),
        "steps": meta.get("steps"),
        "val_best_sharpe": meta.get("val_best_sharpe"),
        "val_final_sharpe": val_stats.get("val_sharpe") if val_stats else None,
        "test_sharpe_net": meta.get("test_sharpe_net"),
        "test_sortino_net": meta.get("test_sortino_net"),
        "test_return": meta.get("test_return"),
        "test_max_dd": meta.get("test_max_dd"),
        "test_turnover_mean": meta.get("test_turnover_mean"),
        "test_gross_exposure_mean": meta.get("test_gross_exposure_mean"),
        "test_cash_weight_mean": meta.get("test_cash_weight_mean", 0.0),
        "test_cost_mean_bps": meta.get("test_cost_mean_bps"),
        "test_n_days": meta.get("test_n_days"),
        "test_v_final": meta.get("test_v_final"),
        "test_sum_reward_rl": meta.get("test_sum_reward_rl"),
    }


def rebuild_master_csv() -> pd.DataFrame:
    """Reconstruye el CSV unificado leyendo TODOS los meta JSON existentes."""
    rows = []
    n_ok = n_bad = 0
    for meta_file in sorted(MODELS_DIR.glob("ppo_*_meta.json")):
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            rows.append(meta_to_row(meta, meta_file.stem))
            n_ok += 1
        except Exception as e:
            n_bad += 1
            log(f"  [rebuild] error leyendo {meta_file.name}: {e}")

    if not rows:
        log("  [rebuild] no hay metas para consolidar")
        return pd.DataFrame(columns=MASTER_COLS)

    df = pd.DataFrame(rows)
    # ordenar columnas
    cols = [c for c in MASTER_COLS if c in df.columns]
    df = df[cols].sort_values(["features", "use_cash", "mu_turnover", "seed"]).reset_index(drop=True)
    df.to_csv(MASTER_CSV, index=False)
    log(f"  [rebuild] master CSV reconstruido: {len(df)} filas ({n_ok} OK, {n_bad} fallos) -> {MASTER_CSV}")
    return df


def update_master_csv_incremental(new_row: dict) -> None:
    """Anade una fila al CSV unificado. Si el modelo ya estaba, sobreescribe esa fila."""
    if MASTER_CSV.exists():
        df = pd.read_csv(MASTER_CSV)
    else:
        df = pd.DataFrame(columns=MASTER_COLS)

    # Eliminar fila previa del mismo model_file (por si se reentrena)
    df = df[df["model_file"] != new_row["model_file"]]
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    cols = [c for c in MASTER_COLS if c in df.columns]
    df = df[cols].sort_values(["features", "use_cash", "mu_turnover", "seed"]).reset_index(drop=True)
    df.to_csv(MASTER_CSV, index=False)


# ==================== MAIN ====================

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true",
                   help="Reentrenar incluso si el modelo ya existe.")
    p.add_argument("--rebuild-only", action="store_true",
                   help="Solo reconstruir el CSV unificado desde los metas existentes.")
    args = p.parse_args()

    if args.rebuild_only:
        log("MODO rebuild-only: consolidando metas existentes...")
        rebuild_master_csv()
        return

    combos = list(product(FEATURE_SETS, USE_CASH, MU_TURNOVER_VALUES, SEEDS))
    total = len(combos)

    log("=" * 80)
    log(f"INICIO | {total} runs planificados")
    log(f"Feature sets:  {FEATURE_SETS}")
    log(f"use_cash:      {USE_CASH}")
    log(f"mu_turnover:   {MU_TURNOVER_VALUES}")
    log(f"Seeds:         {SEEDS}")
    log(f"Lambda_dd={LAMBDA_DD} | leverage={LEVERAGE} | steps={STEPS}")
    log(f"Modo:          {'FORCE' if args.force else 'RESUME'}")
    log("=" * 80)

    done = skipped = failed = 0
    t_start = time.time()

    for idx, (features, use_cash, mu, seed) in enumerate(combos, 1):
        cash_str = "_cash" if use_cash else ""
        label = f"{features}{cash_str}_mu{_fmt(mu)}_seed{seed}"
        prefix = f"[{idx}/{total}]"

        if (not args.force) and model_exists(features, use_cash, mu, seed):
            skipped += 1
            log(f"{prefix} SKIP  {label}")
            # Asegurar que el CSV tiene la fila (por si se interrumpio antes)
            tag = _tag(features, use_cash, mu, seed)
            meta_path = MODELS_DIR / f"ppo_{tag}_meta.json"
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                update_master_csv_incremental(meta_to_row(meta, f"ppo_{tag}"))
            except Exception as e:
                log(f"       error actualizando CSV para {label}: {e}")
            continue

        log(f"{prefix} TRAIN {label} ...")
        ok, dt, err = train_one(features, use_cash, mu, seed)

        if ok:
            done += 1
            log(f"{prefix} OK    {label} ({dt/60:.1f} min)")
            # Actualizar CSV con la nueva fila
            tag = _tag(features, use_cash, mu, seed)
            meta_path = MODELS_DIR / f"ppo_{tag}_meta.json"
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                update_master_csv_incremental(meta_to_row(meta, f"ppo_{tag}"))
            except Exception as e:
                log(f"       error actualizando CSV para {label}: {e}")
        else:
            failed += 1
            log(f"{prefix} FAIL  {label} ({dt/60:.1f} min)")
            if err:
                for line in err.strip().splitlines()[-5:]:
                    log(f"       {line}")

    t_total = time.time() - t_start
    log("=" * 80)
    log(f"FIN | Entrenados: {done} | Skipped: {skipped} | Fallidos: {failed}")
    log(f"Tiempo total: {t_total/3600:.2f} horas")

    # Consolidacion final completa (por robustez)
    log("Consolidando master CSV final...")
    rebuild_master_csv()
    log("=" * 80)


if __name__ == "__main__":
    main()