#!/usr/bin/env python3
"""Optuna hyperparameter search for iVDFM forecasting on one benchmark dataset (minimize sMSE).

Structural settings (window, stride, factor order, number of regimes) are fixed per dataset;
Optuna searches the encoder/decoder/prior sizes, learning rate, epochs, decoder variance,
beta_kl and dropout.

Run from the project root:
    uv run python scripts/tune_forecasting.py --dataset etth1 --n-trials 20 [--timeout 7200] [--out path.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Per-dataset structural settings (aligned with config/experiment/forecasting.yaml).
DATASETS = {
    "etth1": dict(horizon=96, time_context=2, factor_order=2, num_regimes=7, max_hidden_dim=200),
    "etth2": dict(horizon=96, time_context=2, factor_order=2, num_regimes=7, max_hidden_dim=200),
    "ettm1": dict(horizon=96, time_context=2, factor_order=3, num_regimes=7, max_hidden_dim=200),
    "ettm2": dict(horizon=96, time_context=2, factor_order=3, num_regimes=7, max_hidden_dim=200),
    "illness": dict(horizon=7, time_context=2, factor_order=3, num_regimes=4, max_hidden_dim=200),
    "weather": dict(horizon=96, time_context=4, factor_order=3, num_regimes=7, max_hidden_dim=512),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna iVDFM hyperparameter search for forecasting (minimize sMSE)")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="Dataset key (config/dataset/<key>.yaml)")
    p.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    p.add_argument("--timeout", type=float, default=None, help="Total timeout in seconds")
    p.add_argument("--horizon", type=int, default=None, help="Forecast horizon (default: per-dataset)")
    p.add_argument("--window", type=int, default=96, help="Window length (fixed in the study)")
    p.add_argument("--stride", type=int, default=25, help="Stride (fixed in the study)")
    p.add_argument("--fixed-params-json", type=str, default=None, help="Evaluate these parameters once instead of searching")
    p.add_argument("--subset-ratio", type=float, default=None, help="Fraction of the training set to use")
    p.add_argument("--train-window-ratio", type=float, default=1.0, help="Proportion of the latest windows to use, in (0, 1]")
    p.add_argument("--max-epochs-per-trial", type=int, default=None, help="Cap epochs per trial")
    p.add_argument("--out", type=str, default=None, help="Where to write the best parameters (JSON)")
    p.add_argument("--seed", type=int, default=2026, help="Random seed")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    spec = DATASETS[args.dataset]
    horizon = args.horizon if args.horizon is not None else spec["horizon"]

    from omegaconf import OmegaConf

    ds_cfg = OmegaConf.load(PROJECT_ROOT / "config" / "dataset" / f"{args.dataset}.yaml")
    from data_provider.data_factory import data_provider

    df_train, _ = data_provider(OmegaConf.create({"dataset": ds_cfg}), "train")

    import torch
    from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
    from dfm_python.models.ivdfm.optuna import run_hyperparameter_optimization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = iVDFMDataset(
        data=df_train,
        window=args.window,
        stride=args.stride,
        time_context=spec["time_context"],
        scaler=None,
        device=device,
    )
    structural = {
        "time_context": dataset.context_length,
        "num_factors": int(getattr(ds_cfg, "r", 3)),
        "window": args.window,
        "stride": args.stride,
        "factor_order": spec["factor_order"],
        "num_regimes": spec["num_regimes"],
        "mixing": False,
        "regime_temperature": 0.2,
    }

    initial_params = None
    n_trials = args.n_trials
    if args.fixed_params_json:
        with open(args.fixed_params_json, encoding="utf-8") as f:
            initial_params = json.load(f)
        n_trials = 1

    best_params, study = run_hyperparameter_optimization(
        dataset,
        n_trials=n_trials,
        timeout=args.timeout,
        max_window=args.window,
        min_window=args.window,
        max_regimes=spec["num_regimes"],
        horizon=horizon,
        metric="sMSE",
        max_epochs_per_trial=args.max_epochs_per_trial,
        device=device,
        seed=args.seed,
        study_name=f"{args.dataset}_ivdfm",
        fixed=structural,
        subset_ratio=args.subset_ratio,
        train_window_ratio=args.train_window_ratio,
        max_hidden_dim=spec["max_hidden_dim"],
        search_scope="focused",
        initial_params=initial_params,
    )

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "results" / "optuna" / f"{args.dataset}_ivdfm_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Best sMSE: {study.best_value:.6f}")
    print(f"Best params written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
