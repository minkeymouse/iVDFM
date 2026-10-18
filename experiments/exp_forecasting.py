"""
Forecasting experiment: MSE/MAE (standardized-space) and CRPS per dataset, horizon, and model.

Pipeline: for each (dataset, horizon), load data once; fit iVDFM ref if needed;
for each model run fit+eval or use ref (iVDFM); write {output_dir}/{dataset}/{model}/h{horizon}.json;
aggregate to forecasting_results.json; optionally write LaTeX tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.evaluation import evaluate_forecasting_crps

LOG = logging.getLogger(__name__)

CRPS_QUANTILE_LEVELS_PERCENT = [10, 50, 90]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

FORECASTING_DATASETS_ORDER = ["etth1", "etth2", "ettm1", "ettm2", "weather"]
FORECASTING_HORIZONS = [96, 192, 336, 720]
FORECASTING_MODELS_ORDER = ["ivdfm", "itransformer", "timemixer", "timexer", "ddfm", "mlpmultivariate"]
# Result keys:
# - mse/mae: computed in standardized space (no inverse transform) so values are typically O(1).
# - crps: computed from quantiles when available (lower better).
FORECASTING_RESULT_KEYS = ["dataset", "model", "horizon", "mse", "mae", "crps"]

IVDFM_KEY = "ivdfm"
DDFM_KEY = "ddfm"
FACTOR_MODELS = {IVDFM_KEY, DDFM_KEY}
NEURAL_MODELS = {"itransformer", "timemixer", "timexer", "mlpmultivariate"}
REFERENCE_KEYS = frozenset({"model", "scaler"})

FORECASTING_RESULTS_JSON = "forecasting_results.json"


# -----------------------------------------------------------------------------
# Config and data loaders
# -----------------------------------------------------------------------------
def _load_dataset_config(name: str) -> DictConfig:
    """Load dataset config by name (e.g. etth1)."""
    path = CONFIG_ROOT / "dataset" / f"{name.lower()}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return OmegaConf.load(path)


def _load_model_config(name: str) -> DictConfig:
    """Load model config by name (e.g. ivdfm)."""
    key = name.lower()
    path = CONFIG_ROOT / "model" / f"{key}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    return OmegaConf.load(path)


def _get_model_config_optional(name: str) -> Optional[DictConfig]:
    """Load model config if file exists; otherwise None."""
    try:
        return _load_model_config(name)
    except FileNotFoundError:
        return None


def _get_model_window(exp_cfg: DictConfig, model_key: str) -> int:
    """Resolve lookback window: exp <model>_window, exp window, model config, else 96."""
    mkey = model_key.lower()
    w = exp_cfg.get(f"{mkey}_window") or (exp_cfg.get("ddfm_window_size") if mkey == "ddfm" else None) or exp_cfg.get("window")
    if w is not None:
        return int(w)
    model_cfg = _get_model_config_optional(mkey)
    if model_cfg is not None:
        w = model_cfg.get("window") or model_cfg.get("window_size")
        if w is not None:
            return int(w)
    return 96


def _get_model_stride(exp_cfg: DictConfig, model_key: str) -> int:
    """Resolve stride: exp <model>_stride, exp stride, model config, else 25."""
    mkey = model_key.lower()
    s = exp_cfg.get(f"{mkey}_stride") or exp_cfg.get("stride")
    if s is not None:
        return int(s)
    model_cfg = _get_model_config_optional(mkey)
    if model_cfg is not None:
        s = model_cfg.get("stride")
        if s is not None:
            return int(s)
    return 25


def _set_seed(seed: int) -> None:
    """Set global random seed (numpy, torch, random)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Model fitting and representation extraction
# -----------------------------------------------------------------------------
def _fit_factor_model(
    model_name: str,
    df_train: pd.DataFrame,
    cfg: DictConfig,
    scaler: Any,
) -> Any:
    """Fit iVDFM or DDFM; return wrapper with .factors() and .update()."""
    model_name = model_name.lower()
    # Resolve against full cfg so dataset.r etc. are available
    full = OmegaConf.to_container(cfg, resolve=True) or {}
    config_dict = dict(full.get("model") or {})
    device_str = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    if model_name == "ivdfm":
        from model.ivdfm import Model
        r = int(cfg.dataset.get("r", config_dict.get("num_factors", 5)))
        config_dict.setdefault("num_factors", r)
        config_dict.setdefault("window", int(cfg.experiment.get("window", 96)))
        config_dict.setdefault("stride", 25)
        m = Model(config_dict=config_dict, device=device_str)
        m.fit(df_train)
        return m
    if model_name == "ddfm":
        from model.ddfm import Model
        exp = OmegaConf.to_container(cfg.get("experiment") or {}, resolve=True) or {}
        config_dict.setdefault("window_size", int(exp.get("window", 96)))
        if exp.get("ddfm_max_epoch_pretrain") is not None:
            config_dict["max_epoch_pretrain"] = int(exp["ddfm_max_epoch_pretrain"])
        if exp.get("ddfm_max_epoch") is not None:
            config_dict["max_epoch"] = int(exp["ddfm_max_epoch"])
        if exp.get("ddfm_n_mc_samples") is not None:
            config_dict["n_mc_samples"] = int(exp["ddfm_n_mc_samples"])
        encoder_layers = list(exp["ddfm_encoder_layers"]) if exp.get("ddfm_encoder_layers") is not None else list(cfg.model.get("encoder_layers", [64, 32]))
        m = Model(encoder_layers=encoder_layers, config_dict=config_dict)
        m.fit(df_train, target_scaler=scaler)
        return m
    raise ValueError(f"Unknown factor model: {model_name}")


def _fit_neural_model(
    model_name: str,
    df_train: pd.DataFrame,
    cfg: DictConfig,
    seed: Optional[int] = None,
) -> Any:
    """Fit neural forecasting model (iTransformer, TimeMixer, TimeXer, MLPMultivariate). seed controls Lightning/worker RNG for reproducibility."""
    model_name = model_name.lower()
    full = OmegaConf.to_container(cfg, resolve=True) or {}
    config_dict = dict(full.get("model") or {})
    window = int(cfg.experiment.get("window", 96))
    horizon = int(cfg.model.get("h", 96))
    config_dict.setdefault("h", horizon)
    config_dict.setdefault("input_size", 2 * horizon)
    config_dict.setdefault("max_steps", 1000)
    config_dict.setdefault("scaler_type", None)
    if model_name == "mlpmultivariate":
        config_dict.setdefault("num_layers", 2)
        config_dict.setdefault("hidden_size", 1024)
    n_series = len([c for c in df_train.columns if c not in {"date", "Date"}])
    config_dict.setdefault("n_series", n_series)
    from model.neuralforecast import Model
    nf_type = model_name
    m = Model(model_type=nf_type, config_dict=config_dict)
    m.fit(df_train, seed=seed)
    return m


# -----------------------------------------------------------------------------
# iVDFM reference (for forecast sMSE)
# -----------------------------------------------------------------------------
def train_ivdfm_reference(
    cfg_ds: DictConfig,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: Any,
    window: int,
    stride: int,
    horizon: int,
) -> Dict[str, Any]:
    """
    Train reference iVDFM once; return fitted model and scaler for forecast sMSE.
    """
    _set_seed(0)
    model = _fit_factor_model("ivdfm", df_train, cfg_ds, scaler)
    return {"scaler": scaler, "model": model}


def _forecasting_horizons(exp_cfg: DictConfig) -> List[int]:
    """Horizons for forecasting: from config or default [96, 192, 336, 720] for ETT* and weather."""
    h = exp_cfg.get("horizons")
    if h is not None:
        return [int(x) for x in list(h)]
    return list(FORECASTING_HORIZONS)


def _prepare_ddfm_cfg(cfg_ds: DictConfig, exp_cfg: DictConfig, window: int, horizon: int) -> None:
    """Set DDFM model config from experiment overrides. Mutates cfg_ds.model."""
    cfg_ds.model = _load_model_config("ddfm")
    cfg_ds.model.window = window
    cfg_ds.model.window_size = window
    cfg_ds.model.h = horizon
    if exp_cfg.get("ddfm_max_epoch") is not None:
        cfg_ds.model.max_epoch = int(exp_cfg.ddfm_max_epoch)
    if exp_cfg.get("ddfm_encoder_layers") is not None:
        cfg_ds.model.encoder_layers = list(exp_cfg.ddfm_encoder_layers)
    if exp_cfg.get("ddfm_window_size") is not None:
        cfg_ds.model.window_size = int(exp_cfg.ddfm_window_size)
    if exp_cfg.get("ddfm_max_epoch_pretrain") is not None:
        cfg_ds.model.max_epoch_pretrain = int(exp_cfg.ddfm_max_epoch_pretrain)


def _make_result_row(
    dataset: str,
    model: str,
    horizon: int,
    mse: float = float("nan"),
    mae: float = float("nan"),
    crps: float = float("nan"),
) -> Dict[str, Any]:
    """Build one result row; missing values as nan."""
    return {
        "dataset": dataset,
        "model": model,
        "horizon": horizon,
        "mse": mse,
        "mae": mae,
        "crps": crps,
    }


def _result_row_for_json(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert result row for JSON: nan -> None, numpy scalars -> Python types."""
    out = {}
    for k in FORECASTING_RESULT_KEYS:
        if k not in row:
            continue
        v = row[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[k] = None
        elif isinstance(v, (np.integer, np.int64, np.int32)):
            out[k] = int(v)
        elif isinstance(v, (np.floating, np.float64, np.float32)):
            out[k] = float(v) if not np.isnan(v) else None
        else:
            out[k] = v
    return out


def _write_result_json(run_dir: Path, row: Dict[str, Any], horizon: int) -> None:
    """Write one result row to run_dir/h{horizon}.json."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"h{horizon}.json", "w", encoding="utf-8") as f:
        json.dump(_result_row_for_json(row), f, indent=2)


def _save_model_weights(run_dir: Path, horizon: int, model_obj: Any) -> None:
    """Best-effort: save model weights/checkpoint next to metrics JSON."""
    try:
        if model_obj is None:
            return
        if hasattr(model_obj, "save") and callable(getattr(model_obj, "save", None)):
            out_path = run_dir / f"h{horizon}_model.pt"
            model_obj.save(out_path)
            LOG.info("Saved model weights: %s", out_path)
    except Exception as e:
        LOG.warning("Could not save model weights dataset_dir=%s h=%s: %s", run_dir, horizon, e)


def _read_result_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read one result JSON; return dict or None on missing/invalid file."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning("Failed to read %s: %s", path, e)
        return None


def _aggregate_results_from_json(base_dir: Path) -> pd.DataFrame:
    """Glob */*/h*.json, parse horizon from filename, dedupe by (dataset, model, horizon), sort."""
    rows: List[Dict[str, Any]] = []
    for path in sorted(base_dir.glob("*/*/h*.json")):
        try:
            h = int(path.stem[1:])
        except ValueError:
            continue
        data = _read_result_json(path)
        if not data or "dataset" not in data or "model" not in data:
            continue
        data = {**data, "horizon": h}
        rows.append(data)
    if not rows:
        return pd.DataFrame(columns=FORECASTING_RESULT_KEYS)
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["dataset", "model", "horizon"], keep="last")
    df = df.sort_values(["dataset", "horizon", "model"]).reset_index(drop=True)
    return df[[c for c in FORECASTING_RESULT_KEYS if c in df.columns]]


def _ref_has_required_keys(ref: Optional[Dict[str, Any]]) -> bool:
    """True if ref is not None and has model and scaler for forecast sMSE."""
    if ref is None:
        return False
    return REFERENCE_KEYS.issubset(ref.keys())


# -----------------------------------------------------------------------------
# Single (dataset, model, horizon) run
# -----------------------------------------------------------------------------
def _prepare_ivdfm_cfg(
    cfg_ds: DictConfig, ds_key: str, window: int, stride: int, horizon: int
) -> None:
    """Set cfg_ds.model for iVDFM. Base from ivdfm.yaml; merge Hydra model.* overrides, then ivdfm_dataset_params (reproducible; no Optuna). Mutates cfg_ds."""
    base = _load_model_config("ivdfm")
    overrides = cfg_ds.get("model")
    if overrides is not None:
        cfg_ds.model = OmegaConf.merge(base, overrides)
    else:
        cfg_ds.model = base
    exp = cfg_ds.get("experiment") or {}
    dataset_params = exp.get("ivdfm_dataset_params") or {}
    params = dataset_params.get(ds_key) or dataset_params.get(ds_key.lower())
    if params is not None:
        for k, v in OmegaConf.to_container(OmegaConf.create(params), resolve=True).items():
            cfg_ds.model[k] = v
        LOG.info("Using fixed iVDFM params for %s", ds_key)
    cfg_ds.model.window = window
    cfg_ds.model.window_size = window
    cfg_ds.model.stride = stride
    cfg_ds.model.h = horizon
    cfg_ds.model.input_size = cfg_ds.model.get("input_size") or (2 * window)
    if exp.get("ivdfm_max_epochs") is not None:
        cfg_ds.model.max_epochs = int(exp["ivdfm_max_epochs"])


def _run_single_model(
    ds_key: str,
    model_name: str,
    cfg_ds: DictConfig,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    scaler: Any,
    ref: Optional[Dict[str, Any]],
    window: int,
    stride: int,
    horizon: int,
    exp_cfg: DictConfig,
) -> Dict[str, Any]:
    """Run MSE/MAE (standardized-space) and CRPS for one (dataset, model, horizon). Returns result row dict."""
    mkey = model_name.lower()
    max_origins = int(exp_cfg.get("forecasting_max_origins", 10))
    quantiles = tuple(q / 100.0 for q in CRPS_QUANTILE_LEVELS_PERCENT)
    _set_seed(0)

    def forecast_metrics(m, df_tr, df_te, h, sc, mo: int) -> Tuple[float, float, float]:
        try:
            metrics, _, _, _ = evaluate_forecasting_crps(
                m, df_tr, df_te, h, scaler=sc,
                rolling_stride=h, max_origins=mo, use_factor_updates=True,
                quantiles=quantiles,
            )
            return (
                float(metrics.get("MSE", float("nan"))),
                float(metrics.get("MAE", float("nan"))),
                float(metrics.get("CRPS", float("nan"))),
            )
        except Exception as e:
            LOG.warning("Forecast MSE/MAE/CRPS failed: %s", e)
            return float("nan"), float("nan"), float("nan")

    def row(mse: float = float("nan"), mae: float = float("nan"), crps: float = float("nan")) -> Dict[str, Any]:
        return _make_result_row(ds_key, mkey, horizon, mse=mse, mae=mae, crps=crps)

    if mkey == IVDFM_KEY:
        if not _ref_has_required_keys(ref):
            return row()
        try:
            # Standardized-space MSE/MAE: do not inverse-transform
            run_dir = Path(exp_cfg.get("output_dir", "./results/forecasting")) / ds_key / mkey
            mse_val, mae_val, crps_val = forecast_metrics(
                ref["model"], df_train, df_test, horizon, None, max_origins
            )
            _save_model_weights(run_dir, horizon, ref["model"])
            return row(mse=mse_val, mae=mae_val, crps=crps_val)
        except Exception as e:
            LOG.warning("iVDFM run failed dataset=%s: %s", ds_key, e)
            return row()

    if mkey == DDFM_KEY:
        _prepare_ddfm_cfg(cfg_ds, exp_cfg, window, horizon)
        try:
            model = _fit_factor_model(mkey, df_train, cfg_ds, scaler)
            run_dir = Path(exp_cfg.get("output_dir", "./results/forecasting")) / ds_key / mkey
            mse_val, mae_val, crps_val = forecast_metrics(
                model, df_train, df_test, horizon, None, max_origins
            )
            _save_model_weights(run_dir, horizon, model)
            return row(mse=mse_val, mae=mae_val, crps=crps_val)
        except Exception as e:
            LOG.warning("Forecasting run failed dataset=%s model=%s: %s", ds_key, mkey, e)
            return row()

    if mkey in NEURAL_MODELS:
        cfg_ds.model = _load_model_config(mkey)
        cfg_ds.model.window = cfg_ds.model.window_size = window
        cfg_ds.model.h = horizon
        cfg_ds.model.input_size = window
        cfg_ds.model.quantile_levels = CRPS_QUANTILE_LEVELS_PERCENT
        try:
            model = _fit_neural_model(mkey, df_train, cfg_ds, seed=0)
            run_dir = Path(exp_cfg.get("output_dir", "./results/forecasting")) / ds_key / mkey
            mse_val, mae_val, crps_val = forecast_metrics(
                model, df_train, df_test, horizon, None, max_origins
            )
            _save_model_weights(run_dir, horizon, model)
            return row(mse=mse_val, mae=mae_val, crps=crps_val)
        except Exception as e:
            LOG.warning("Forecasting run failed dataset=%s model=%s: %s", ds_key, mkey, e)
            return row()

    return row()


# -----------------------------------------------------------------------------
# Data load and placeholders
# -----------------------------------------------------------------------------
def _write_placeholder_results(base_dir: Path, ds_key: str, models: List[str], horizons: List[int]) -> None:
    """Write placeholder (nan) result JSON for each (ds_key, model, horizon) when dataset is skipped."""
    for model_name in models:
        mkey = model_name.lower()
        run_dir = base_dir / ds_key / mkey
        for h in horizons:
            _write_result_json(run_dir, _make_result_row(ds_key, mkey, h), horizon=h)


def _load_dataset_splits(
    cfg: DictConfig, exp_cfg: DictConfig, ds_key: str, horizon: int,
) -> Optional[Tuple[DictConfig, Any, Any, Any, Any]]:
    """
    Load dataset config and train/val/test splits. Returns (cfg_ds, df_train, df_val, df_test, scaler)
    or None on load failure. horizon is set on cfg_ds.experiment for downstream use.
    """
    ds_cfg = _load_dataset_config(ds_key)
    cfg_ds = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_ds.dataset = ds_cfg
    cfg_ds.experiment = exp_cfg
    cfg_ds.experiment.horizon = horizon
    try:
        df_train, scaler = data_provider(cfg_ds, "train")
        df_val, _ = data_provider(cfg_ds, "val")
        df_test, _ = data_provider(cfg_ds, "test")
    except Exception as e:
        LOG.warning("data_provider failed dataset=%s: %s", ds_key, e)
        return None
    return (cfg_ds, df_train, df_val, df_test, scaler)


def run_forecasting(cfg: DictConfig) -> pd.DataFrame:
    """Thin wrapper: Exp_Forecasting(cfg).run()."""
    return Exp_Forecasting(cfg).run()


# -----------------------------------------------------------------------------
# LaTeX table writer
# -----------------------------------------------------------------------------
def _dataset_display_name(ds_key: str) -> str:
    """Map dataset key to table display name."""
    names = {
        "etth1": "ETTh1", "etth2": "ETTh2",
        "ettm1": "ETTm1", "ettm2": "ETTm2",
        "illness": "Illness", "weather": "Weather",
    }
    return names.get(ds_key.lower(), ds_key)


def _model_display_name(mkey: str) -> str:
    """Map model key to table display name."""
    names = {
        "ivdfm": "iVDFM", "ddfm": "DDFM",
        "itransformer": "iTransformer", "timemixer": "TimeMixer",
        "timexer": "TimeXer", "mlpmultivariate": "MLPMultivariate",
    }
    return names.get(mkey.lower(), mkey)


def _load_results_df(path: Path) -> Optional[pd.DataFrame]:
    """Load aggregated results from forecasting_results.json; return DataFrame or None."""
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return None
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning("Failed to read %s: %s", path, e)
        return None


def write_forecasting_latex_tables(
    results_path: Path,
    tables_dir: Path,
    datasets_order: Optional[List[str]] = None,
    horizons_order: Optional[List[int]] = None,
) -> None:
    """Read aggregated forecasting_results.json and write forecasting.tex and forecasting_full.tex.
    Table: row1 = model names (each spanning 2 cols), row2 = CRPS, MSE (standardized) under each model, row3+ = dataset (horizon) and values.
    """
    df = _load_results_df(results_path)
    if df is None:
        LOG.warning("Results not found or unreadable: %s; skipping LaTeX write.", results_path)
        return
    if df.empty:
        return
    required = ["dataset", "model", "horizon", "mse"]
    if any(c not in df.columns for c in required):
        LOG.warning("Results missing columns %s; skipping LaTeX write.", required)
        return
    if "crps" not in df.columns:
        df = df.assign(crps=float("nan"))

    datasets_found = list(datasets_order or FORECASTING_DATASETS_ORDER)
    horizons_found = list(horizons_order or FORECASTING_HORIZONS)
    models = list(FORECASTING_MODELS_ORDER)
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    def lookup(ds: str, model: str, h: int, col: str) -> float:
        row = df[(df["dataset"] == ds) & (df["model"] == model) & (df["horizon"] == h)]
        if row.empty or col not in df.columns:
            return float("nan")
        try:
            return float(row[col].iloc[0])
        except (TypeError, ValueError):
            return float("nan")

    def fmt_val(x: float) -> str:
        return "---" if pd.isna(x) or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"

    n_models = len(models)
    caption = "CRPS $\\downarrow$ and MSE (standardized) $\\downarrow$. Rows: dataset (horizon); columns: per-model CRPS and MSE (standardized)."
    # Row 1: model names, each spanning 2 columns
    header_row1 = "Dataset (H) & " + " & ".join(
        "\\multicolumn{2}{c}{" + _model_display_name(m) + "}" for m in models
    ) + " \\\\"
    # Row 2: CRPS, MSE (standardized) under each model
    header_row2 = " & " + " & ".join(["CRPS $\\downarrow$", "MSE (std.) $\\downarrow$"] * n_models) + " \\\\"

    row_keys: List[Tuple[str, int]] = []
    for ds in datasets_found:
        for h in horizons_found:
            row_keys.append((ds, h))

    def row_label(ds: str, h: int) -> str:
        return _dataset_display_name(ds) + f" ({h})"

    body_lines = [
        "    "
        + " & ".join(
            [row_label(ds, h)]
            + [y for m in models for y in (fmt_val(lookup(ds, m, h, "crps")), fmt_val(lookup(ds, m, h, "mse")))]
        )
        + " \\\\"
        for (ds, h) in row_keys
    ]
    table_body = "\n".join(body_lines) + "\n    \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    tabular_spec = "l" + "c" * (2 * n_models)
    table_specs = [
        ("forecasting.tex", "tab:forecasting", "3pt", "H"),
        ("forecasting_full.tex", "tab:forecasting_full", "4pt", "p"),
    ]
    for filename, label, tabcolsep, float_placement in table_specs:
        header = (
            f"\\begin{{table}}[{float_placement}]"
            "\n    \\centering"
            "\n    \\scriptsize"
            f"\n    \\setlength{{\\tabcolsep}}{{{tabcolsep}}}"
            "\n    \\renewcommand{\\arraystretch}{0.95}"
            f"\n    \\caption{{{caption}}}"
            f"\n    \\label{{{label}}}"
            f"\n    \\begin{{tabular}}{{{tabular_spec}}}"
            "\n    \\toprule"
            "\n    " + header_row1 + "\n    " + header_row2 + "\n    \\midrule\n"
        )
        (tables_dir / filename).write_text(header + table_body, encoding="utf-8")
        LOG.info("Wrote %s", tables_dir / filename)


# -----------------------------------------------------------------------------
# Experiment class (Exp_Basic entry point)
# -----------------------------------------------------------------------------
class Exp_Forecasting(Exp_Basic):
    """
    Forecasting experiment: CRPS and MSE (standardized) per dataset, horizon and model.

    No single model; multiple models are fit per (dataset, horizon) inside run().
    Results under output_dir as {dataset}/{model}/h{horizon}.json; aggregated
    forecasting_results.json for LaTeX tables (row = dataset (horizon), columns = per-model CRPS / MSE (std.)).
    """

    def _build_model(self):
        # Forecasting evaluates multiple models per dataset; no single persistent model.
        return None

    def train(self):
        """Not used; forecasting runs multiple models in run()."""
        return None

    def test(self):
        """Not used; metrics computed inside run()."""
        return None

    def _load_dataset_splits(self, ds_key: str, horizon: int) -> Optional[Tuple[DictConfig, Any, Any, Any, Any]]:
        """Load dataset config and train/val/test splits. Returns (cfg_ds, df_train, df_val, df_test, scaler) or None."""
        return _load_dataset_splits(self.cfg, self.exp_cfg, ds_key, horizon)

    def _run_single_model(
        self,
        ds_key: str,
        model_name: str,
        cfg_ds: DictConfig,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        scaler: Any,
        ref: Optional[Dict[str, Any]],
        window_m: int,
        stride_m: int,
        horizon: int,
        exp_cfg: DictConfig,
    ) -> Dict[str, Any]:
        """Run sMSE for one (dataset, model). Returns result row dict."""
        return _run_single_model(
            ds_key, model_name, cfg_ds,
            df_train, df_val, df_test, scaler,
            ref, window_m, stride_m, horizon, exp_cfg,
        )

    def _write_tables(self, results_path: Path) -> None:
        """Write LaTeX tables from aggregated results if forecasting_tables_dir is set."""
        tables_dir = self.exp_cfg.get("forecasting_tables_dir")
        if not tables_dir or not results_path.exists():
            return
        write_forecasting_latex_tables(
            results_path,
            Path(tables_dir),
            datasets_order=FORECASTING_DATASETS_ORDER,
            horizons_order=FORECASTING_HORIZONS,
        )

    def run(self) -> pd.DataFrame:
        """Run full forecasting experiment (all datasets × horizons × models) and optionally write LaTeX tables."""
        self.exp_cfg = self.cfg.experiment
        datasets = list(self.exp_cfg.get("datasets", FORECASTING_DATASETS_ORDER))
        models = list(self.exp_cfg.get("models", FORECASTING_MODELS_ORDER))
        horizons = _forecasting_horizons(self.exp_cfg)
        if not datasets or not models or not horizons:
            LOG.warning("run_forecasting: empty datasets, models or horizons; returning empty DataFrame.")
            return pd.DataFrame(columns=FORECASTING_RESULT_KEYS)

        base_dir = Path(self.exp_cfg.get("output_dir", "./results/forecasting"))
        base_dir.mkdir(parents=True, exist_ok=True)

        for ds in datasets:
            ds_key = ds.lower()
            # Load once per dataset (use first horizon for initial cfg; horizon is overwritten per run)
            loaded = self._load_dataset_splits(ds_key, horizons[0])
            if loaded is None:
                _write_placeholder_results(base_dir, ds_key, models, horizons)
                continue

            cfg_ds, df_train, df_val, df_test, scaler = loaded

            for horizon in horizons:
                cfg_ds.experiment.horizon = horizon
                ref: Optional[Dict[str, Any]] = None

                if "ivdfm" in [m.lower() for m in models]:
                    w_ivdfm = _get_model_window(self.exp_cfg, "ivdfm")
                    s_ivdfm = _get_model_stride(self.exp_cfg, "ivdfm")
                    if len(df_train) >= w_ivdfm:
                        try:
                            cfg_ds.experiment.window = w_ivdfm
                            cfg_ds.experiment.stride = s_ivdfm
                            _prepare_ivdfm_cfg(cfg_ds, ds_key, w_ivdfm, s_ivdfm, horizon)
                            ref = train_ivdfm_reference(
                                cfg_ds, df_train, df_val, df_test, scaler,
                                w_ivdfm, s_ivdfm, horizon,
                            )
                        except Exception as e:
                            LOG.warning("iVDFM reference failed dataset=%s horizon=%s: %s", ds_key, horizon, e)

                for model_name in models:
                    mkey = model_name.lower()
                    window_m = _get_model_window(self.exp_cfg, mkey)
                    stride_m = _get_model_stride(self.exp_cfg, mkey)
                    run_dir = base_dir / ds_key / mkey

                    if len(df_train) < window_m:
                        LOG.warning(
                            "dataset=%s model=%s horizon=%s: train length %d < window %d; skipping.",
                            ds_key, mkey, horizon, len(df_train), window_m,
                        )
                        _write_result_json(run_dir, _make_result_row(ds_key, mkey, horizon), horizon=horizon)
                        continue

                    cfg_ds.experiment.window = window_m
                    cfg_ds.experiment.stride = stride_m

                    row = self._run_single_model(
                        ds_key, model_name, cfg_ds,
                        df_train, df_val, df_test, scaler,
                        ref, window_m, stride_m, horizon,
                        self.exp_cfg,
                    )
                    _write_result_json(run_dir, row, horizon=horizon)

        df_results = _aggregate_results_from_json(base_dir)
        results_path = base_dir / FORECASTING_RESULTS_JSON
        if not df_results.empty:
            records = [_result_row_for_json(r) for r in df_results.to_dict("records")]
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            LOG.info("Wrote %s (%d rows)", results_path, len(df_results))
        self._write_tables(results_path)
        return df_results
