"""
Application experiment (Section 5): fit iVDFM to a real multivariate panel such as daily
exchange rates or weekly influenza-like-illness surveillance, and report factor loadings.

This follows the project pattern:
- reads parameters from cfg.experiment.*
- uses utils.model_fitting.fit_model
- writes results under cfg.experiment.output_dir
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from experiments.exp_basic import Exp_Basic
from utils.model_fitting import fit_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


class Exp_Application(Exp_Basic):
    """Fit a factor model to one real data panel and save factors and loadings."""

    def _build_model(self):
        # Model is fit inside run() to allow simple config overrides.
        return None

    def train(self):
        return None

    def test(self):
        return None

    def run(self) -> None:
        exp_cfg = self.cfg.experiment

        data_path = str(exp_cfg.get("data_path", "./data/exchange/exchange_rate.csv"))
        time_col = str(exp_cfg.get("time_col", "date"))
        series_cols = _as_list(exp_cfg.get("series_cols", []))
        if not series_cols:
            series_cols = ["AUD_USD", "JPY_USD", "CAD_USD", "CNY_USD", "NZD_USD"]

        n_rows = exp_cfg.get("n_rows")
        n_rows_i: Optional[int] = int(n_rows) if n_rows is not None else None

        # Optional slicing by date range. When provided, this takes precedence
        # over `n_rows` (so we can zoom into crisis regimes).
        start_time = exp_cfg.get("start_time")
        end_time = exp_cfg.get("end_time")
        start_time_i = start_time if start_time is not None else None
        end_time_i = end_time if end_time is not None else None

        model_name = str(exp_cfg.get("model", "ivdfm")).lower()
        r = int(exp_cfg.get("r", 2))
        window = int(exp_cfg.get("window", 300))
        stride = int(exp_cfg.get("stride", 1))
        max_epochs = int(exp_cfg.get("max_epochs", 50))
        batch_size = int(exp_cfg.get("batch_size", 128))

        out_root = Path(str(exp_cfg.get("output_dir", "./results/application")))
        name = str(exp_cfg.get("name", "exchange"))
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Load and standardize
        # -------------------------
        df_raw = pd.read_csv(data_path, parse_dates=[time_col])
        if time_col not in df_raw.columns:
            raise ValueError(f"Expected time column {time_col!r} in {data_path}")

        missing = [c for c in series_cols if c not in df_raw.columns]
        if missing:
            raise ValueError(f"Missing columns in {data_path}: {missing}")

        df_raw = df_raw[[time_col] + series_cols].dropna()
        df_raw = df_raw.sort_values(time_col)
        df_raw = df_raw.set_index(time_col)

        if start_time_i is not None or end_time_i is not None:
            # Pandas accepts ISO strings and converts them using the index dtype.
            if start_time_i is not None:
                df_raw = df_raw.loc[str(start_time_i):]
            if end_time_i is not None:
                df_raw = df_raw.loc[:str(end_time_i)]
            if df_raw.empty:
                raise ValueError(
                    f"Date slice resulted in empty dataframe: start_time={start_time_i}, end_time={end_time_i}"
                )
        elif n_rows_i is not None:
            df_raw = df_raw.iloc[:n_rows_i].copy()

        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_raw.values),
            index=df_raw.index,
            columns=df_raw.columns,
        )

        # -------------------------
        # Fit iVDFM (via unified fitter)
        # -------------------------
        # Use the repo's iVDFM model config as the base (avoid inheriting the
        # default `config/model/dfm.yaml` from config/config.yaml).
        cfg_local = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        ivdfm_base = OmegaConf.load(CONFIG_ROOT / "model" / "ivdfm.yaml")
        cfg_local.model = OmegaConf.merge(ivdfm_base, OmegaConf.create({
            "num_factors": r,
            "window": window,
            "stride": stride,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        }))

        model = fit_model(model_name, df_scaled, cfg_local, r=r, device=self.device)
        F = np.asarray(model.factors())  # (window, r) or (T, r)

        # Align outputs to the same window used for factors when fitting.
        n = int(F.shape[0])
        Y = df_scaled.values[-n:]

        # Variance explained: R^2 of linear projection Y ~ F (per-window).
        lr = LinearRegression().fit(F, Y)
        r2 = float(lr.score(F, Y))

        # Simple interpretability: correlation of each factor with each series (on window).
        corr = np.corrcoef(np.hstack([F, Y]).T)
        # corr block: [F,F], [F,Y], [Y,F], [Y,Y]
        corr_FY = corr[:r, r:]
        corr_rows: List[Dict[str, Any]] = []
        for i in range(r):
            for j, col in enumerate(series_cols):
                corr_rows.append({"factor": i, "series": col, "corr": float(corr_FY[i, j])})
        corr_df = pd.DataFrame(corr_rows)

        summary = {
            "name": name,
            "data_path": data_path,
            "time_col": time_col,
            "series_cols": series_cols,
            "n_rows_used": int(len(df_raw)),
            "window": window,
            "stride": stride,
            "r": r,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "variance_explained_r2": r2,
            "factor_std": [float(x) for x in np.std(F, axis=0)],
        }

        # Optional metadata for plotting in downstream scripts (e.g., gray crisis band).
        crisis_start_time = exp_cfg.get("crisis_start_time")
        crisis_end_time = exp_cfg.get("crisis_end_time")
        if crisis_start_time is not None:
            summary["crisis_start_time"] = str(crisis_start_time)
        if crisis_end_time is not None:
            summary["crisis_end_time"] = str(crisis_end_time)

        # Record slicing metadata for downstream visualization.
        if start_time_i is not None or end_time_i is not None:
            summary["start_time_used"] = str(df_raw.index.min())
            summary["end_time_used"] = str(df_raw.index.max())
        else:
            summary["start_time_used"] = str(df_raw.index.min())
            summary["end_time_used"] = str(df_raw.index.max())

        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        corr_df.to_csv(out_dir / "factor_series_corr.csv", index=False)

        # Save factors for optional plotting downstream
        np.save(out_dir / "factors.npy", F)
        df_scaled.iloc[-n:].to_csv(out_dir / "window_scaled.csv", index=True)

        print(f"[example] wrote {out_dir}/summary.json")

