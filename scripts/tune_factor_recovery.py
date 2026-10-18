#!/usr/bin/env python3
"""Optuna hyperparameter study for iVDFM on the factor-recovery dynamic suite (maximize mean MCC over K seeds).

Objective: mean MCC over K fixed seeds (aligned with factor recovery), so tuning maximizes the same metric we report.
Fixed for fairness: window 200, max_epochs 200, learning_rate 0.002 (from factor_recovery common config).
Tunable: capacity, regularization, optimizer details (dropout, dims, layers, beta_kl, batch_size, patience,
scheduler_type, innovation_distribution, factor_order, time_context, use_layer_norm, prior_*, optimizer_weight_decay).

Run from project root: uv run python scripts/tune_factor_recovery.py --n-trials 20 [--n-seeds 5] [--timeout 7200]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna iVDFM tuning on dynamic suite (maximize mean MCC over K seeds)")
    p.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    p.add_argument("--n-seeds", type=int, default=5, help="Number of fixed seeds per trial (objective = mean MCC over these)")
    p.add_argument("--timeout", type=float, default=None, help="Total timeout in seconds")
    p.add_argument("--out", type=str, default=None, help="Path to save best_params JSON")
    p.add_argument("--seed", type=int, default=2026, help="Base seed; seeds used are seed+0, seed+1, ... seed+(n_seeds-1)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    from omegaconf import OmegaConf
    import numpy as np
    import pandas as pd
    import torch
    import optuna

    exp_cfg = OmegaConf.load(PROJECT_ROOT / "config" / "experiment" / "factor_recovery.yaml")
    T = int(exp_cfg.get("T", 200))
    N = int(exp_cfg.get("N", 20))
    r = int(exp_cfg.get("r", 5))
    rho = float(exp_cfg.get("rho", 0.7))
    alpha = float(exp_cfg.get("alpha", 0.2))
    poly_degree = int(exp_cfg.get("poly_degree", 1))
    sign_features = int(exp_cfg.get("sign_features", 0))
    portion_missings = float(exp_cfg.get("portion_missings", 0.0))

    base_overrides = (exp_cfg.get("suite_model_overrides") or {}).get("dynamic") or {}
    ivdfm_base = (base_overrides.get("ivdfm") or {})
    ivdfm_base = OmegaConf.to_container(OmegaConf.create(ivdfm_base), resolve=True) or {}
    # Fixed for fairness: window, max_epochs, learning_rate (from factor_recovery common config)
    fixed_fair = {
        "window": ivdfm_base.get("window", 200),
        "max_epochs": ivdfm_base.get("max_epochs", 200),
        "learning_rate": ivdfm_base.get("learning_rate", 0.002),
    }
    config_root = PROJECT_ROOT / "config"
    model_base = OmegaConf.load(config_root / "model" / "ivdfm.yaml")
    model_base = OmegaConf.merge(model_base, OmegaConf.create(ivdfm_base))

    # Seeds for objective: aligned with factor recovery (seed+0, seed+1, ...)
    n_seeds = max(1, min(args.n_seeds, 10))
    eval_seeds = [args.seed + i for i in range(n_seeds)]

    from synthetic.dynamic import DDFMSynthetic
    from utils.model_fitting import fit_model
    from utils.metrics import mean_corr_coef

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        # Tunable only (no window, max_epochs, learning_rate)
        trial_params = {
            "dropout": trial.suggest_float("dropout", 0.05, 0.25),
            "encoder_hidden_dim": trial.suggest_categorical("encoder_hidden_dim", [64, 100, 128]),
            "decoder_hidden_dim": trial.suggest_categorical("decoder_hidden_dim", [64, 100, 128]),
            "encoder_n_hidden_layers": trial.suggest_int("encoder_n_hidden_layers", 1, 3),
            "decoder_n_hidden_layers": trial.suggest_int("decoder_n_hidden_layers", 1, 3),
            "beta_kl": trial.suggest_float("beta_kl", 0.5, 1.5),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "patience": trial.suggest_int("patience", 15, 35),
            "optimizer_weight_decay": trial.suggest_float("optimizer_weight_decay", 1e-3, 0.05, log=True),
            "use_layer_norm": trial.suggest_categorical("use_layer_norm", [True, False]),
            "prior_hidden_dim": trial.suggest_categorical("prior_hidden_dim", [32, 64, 96]),
            "prior_n_hidden_layers": trial.suggest_int("prior_n_hidden_layers", 1, 2),
            "scheduler_type": trial.suggest_categorical("scheduler_type", ["cosine", "step", "plateau", "exponential"]),
            "innovation_distribution": trial.suggest_categorical("innovation_distribution", ["laplace", "gaussian", "student_t"]),
            "factor_order": trial.suggest_int("factor_order", 1, 3),
            "time_context": trial.suggest_int("time_context", 1, 3),
        }
        trial_params.update(fixed_fair)
        cfg_model = OmegaConf.merge(model_base, OmegaConf.create(trial_params))
        cfg = OmegaConf.create({"model": cfg_model, "dataset": {"r": r}})
        OmegaConf.resolve(cfg)

        mcc_list = []
        for data_seed in eval_seeds:
            gen = DDFMSynthetic(
                n_obs=T,
                n_series=N,
                factor_dim=r,
                rho=rho,
                alpha=alpha,
                poly_degree=poly_degree,
                sign_features=sign_features,
                missing_rate=portion_missings,
                seed=data_seed,
            )
            dyn = gen.generate()
            x_dyn = dyn["x"]
            f_true = dyn["latent"]
            df_dyn = pd.DataFrame(x_dyn, columns=[f"series_{i}" for i in range(N)])
            df_dyn.index = pd.RangeIndex(start=0, stop=T)

            try:
                model = fit_model(
                    model_name="ivdfm",
                    df_train=df_dyn,
                    cfg=cfg,
                    r=r,
                    u_train=None,
                    device=device,
                    scaler=None,
                )
            except Exception as e:
                trial.set_user_attr("error", str(e))
                mcc_list.append(float("nan"))
                continue

            f_hat = model.factors()
            if f_hat.shape[1] != f_true.shape[1]:
                min_dim = min(f_hat.shape[1], f_true.shape[1])
                f_hat = f_hat[:, :min_dim]
                f_true_eval = f_true[:, :min_dim]
            else:
                f_true_eval = f_true
            if f_hat.shape[0] != f_true_eval.shape[0]:
                T_common = min(f_true_eval.shape[0], f_hat.shape[0])
                f_true_eval = f_true_eval[-T_common:, :]
                f_hat = f_hat[-T_common:, :]
            f_true_t = torch.from_numpy(f_true_eval).float()
            f_hat_t = torch.from_numpy(f_hat.astype(np.float32)).float()
            mcc = mean_corr_coef(f_true_t, f_hat_t).item()
            mcc_list.append(mcc if np.isfinite(mcc) else float("nan"))

        valid = [x for x in mcc_list if np.isfinite(x)]
        if not valid:
            return float("nan")
        mean_mcc = float(np.mean(valid))
        return -mean_mcc

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=5))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    best_params = study.best_params
    # Best params are tunable only; merge with fixed_fair when writing to factor_recovery config
    best_mcc = -study.best_value if np.isfinite(study.best_value) else float("nan")

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "results" / "optuna" / "dynamic_ivdfm_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best mean MCC (n_seeds={n_seeds}): {best_mcc:.4f}")
    print(f"Best params (tunable only) written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
