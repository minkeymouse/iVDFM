"""
Forecast evaluation: rolling-origin protocol with sMSE, sMAE and CRPS from quantile forecasts.
"""

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Logger for forecasting pipeline (evaluation protocol, metrics)
FORECAST_LOG = logging.getLogger("psdfm.forecasting")

# Default quantile levels for CRPS (symmetric)
DEFAULT_CRPS_QUANTILES = (0.1, 0.5, 0.9)


# ---- Rolling-origin forecasting evaluation (sMSE / sMAE / CRPS) ----

def _compute_forecasting_metrics(
    preds: np.ndarray,
    trues: np.ndarray,
    scaler: Optional[StandardScaler | RobustScaler] = None,
) -> Dict[str, float]:
    """
    Multivariate sMSE/sMAE: per-variable MSE/MAE normalized by Var(y_j)/std(y_j), then averaged.
    Inverse-transforms with scaler if provided, then computes in original space.
    """
    preds_flat = preds.reshape(-1, preds.shape[-1])
    trues_flat = trues.reshape(-1, trues.shape[-1])
    n_origins, horizon, n_series = preds.shape

    if scaler is not None:
        try:
            if preds_flat.shape[1] == scaler.n_features_in_:
                preds_flat = scaler.inverse_transform(preds_flat)
                trues_flat = scaler.inverse_transform(trues_flat)
                FORECAST_LOG.info("[METRICS] inverse_transform applied; multivariate sMSE/sMAE in original space.")
            else:
                FORECAST_LOG.warning(
                    "[METRICS] scaler n_features_in_=%s != preds n_series=%d; skipping inverse transform.",
                    getattr(scaler, "n_features_in_", None), preds_flat.shape[1],
                )
        except Exception as e:
            FORECAST_LOG.warning("[METRICS] Could not inverse transform: %s", e)
    else:
        FORECAST_LOG.info("[METRICS] no scaler; multivariate sMSE/sMAE in scaled space.")

    n = preds_flat.shape[1]
    vars_j = [float(np.var(trues_flat[:, j])) for j in range(n)]
    # Floor so low-variance series (e.g. Exchange JPY/CNY) don't blow up sMSE/sMAE
    median_var = float(np.median(vars_j))
    var_floor = max(1e-10, median_var)
    std_floor = max(1e-10, np.sqrt(var_floor))

    smse_per_var = []
    smae_per_var = []
    for j in range(n):
        p_j = preds_flat[:, j]
        t_j = trues_flat[:, j]
        var_j = vars_j[j]
        std_j = float(np.std(t_j)) if var_j >= 1e-20 else std_floor
        mse_j = float(np.mean((p_j - t_j) ** 2))
        mae_j = float(np.mean(np.abs(p_j - t_j)))
        denom_var = max(var_j, var_floor)
        denom_std = max(std_j, std_floor) if np.isfinite(std_j) else std_floor
        if np.isfinite(denom_std) and denom_std >= 1e-10:
            smse_per_var.append(mse_j / denom_var)
            smae_per_var.append(mae_j / denom_std)
        else:
            smse_per_var.append(float("nan"))
            smae_per_var.append(float("nan"))
    smse = float(np.nanmean(smse_per_var)) if smse_per_var else float("nan")
    smae = float(np.nanmean(smae_per_var)) if smae_per_var else float("nan")
    # Raw MSE and MAE (mean over all points) for reporting
    mse = float(np.mean((preds_flat - trues_flat) ** 2))
    mae = float(np.mean(np.abs(preds_flat - trues_flat)))
    return {"MSE": mse, "MAE": mae, "sMSE": smse, "sMAE": smae}


def evaluate_forecasting_crps(
    model: Any,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    horizon: int,
    scaler: Optional[StandardScaler | RobustScaler] = None,
    rolling_stride: Optional[int] = None,
    max_origins: Optional[int] = 10,
    use_factor_updates: bool = True,
    quantiles: Tuple[float, ...] = DEFAULT_CRPS_QUANTILES,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, Optional[List[Dict[float, np.ndarray]]]]:
    """
    Rolling-origin evaluation: refit/update on the history, forecast `horizon` steps, repeat over
    up to `max_origins` origins. When the model has predict_quantiles, also computes CRPS. Returns (metrics, preds, trues, qdicts_per_origin).
    metrics includes 'sMSE' (from median) and 'CRPS' (when quantiles available).
    """
    from utils.metrics import crps_from_quantiles

    if rolling_stride is None:
        rolling_stride = horizon

    has_update = hasattr(model, "update") and callable(getattr(model, "update", None))
    has_predict_quantiles = hasattr(model, "predict_quantiles") and callable(
        getattr(model, "predict_quantiles", None)
    )

    original_factors_backup = None
    original_innovations_backup = None
    if has_update and use_factor_updates and hasattr(model, "_model"):
        try:
            if hasattr(model._model, "factors") and model._model.factors is not None:
                f = model._model.factors
                if isinstance(f, np.ndarray) and f.size > 0:
                    original_factors_backup = f.copy()
            if hasattr(model._model, "innovations") and model._model.innovations is not None:
                inv = model._model.innovations
                if isinstance(inv, np.ndarray) and inv.size > 0:
                    original_innovations_backup = inv.copy()
        except Exception as e:
            FORECAST_LOG.warning("Could not backup factor state for CRPS eval: %s", e)

    n_test = len(df_test)
    if n_test < horizon:
        raise ValueError(f"Test length ({n_test}) < horizon ({horizon})")
    n_origins = (n_test - horizon) // rolling_stride + 1
    if max_origins is not None:
        n_origins = min(n_origins, max_origins)

    preds_list: List[np.ndarray] = []
    trues_list: List[np.ndarray] = []
    qdicts_list: List[Dict[float, np.ndarray]] = []

    for i in range(n_origins):
        origin_idx = i * rolling_stride
        if origin_idx + horizon > n_test:
            break
        df_hist = (
            pd.concat([df_train, df_test.iloc[:origin_idx]], axis=0) if origin_idx > 0 else df_train
        )
        true_vals = df_test.iloc[origin_idx : origin_idx + horizon].values.copy().astype(np.float64)
        if true_vals.ndim == 1:
            true_vals = true_vals.reshape(-1, 1)

        try:
            if has_update and use_factor_updates and origin_idx > 0 and hasattr(model, "_model"):
                try:
                    if original_factors_backup is not None and hasattr(model._model, "factors"):
                        model._model.factors = (
                            original_factors_backup.copy()
                            if isinstance(original_factors_backup, np.ndarray)
                            and original_factors_backup.size > 0
                            else None
                        )
                    if original_innovations_backup is not None and hasattr(
                        model._model, "innovations"
                    ):
                        model._model.innovations = (
                            original_innovations_backup.copy()
                            if isinstance(original_innovations_backup, np.ndarray)
                            and original_innovations_backup.size > 0
                            else None
                        )
                    new_data = df_test.iloc[:origin_idx]
                    if len(new_data) > 0:
                        sig = inspect.signature(model.update)
                        update_kwargs = {}
                        if "append" in sig.parameters:
                            update_kwargs["append"] = False
                        if "store_full_history" in sig.parameters:
                            update_kwargs["store_full_history"] = False
                        if "verbose" in sig.parameters:
                            update_kwargs["verbose"] = False
                        model.update(new_data, **update_kwargs)
                except Exception as e:
                    FORECAST_LOG.warning("Factor update failed at origin %s: %s", origin_idx, e)

            if has_predict_quantiles:
                sig = inspect.signature(model.predict_quantiles)
                if "df" in sig.parameters:
                    median, qdict = model.predict_quantiles(horizon=horizon, df=df_hist)
                else:
                    median, qdict = model.predict_quantiles(horizon=horizon)
                median = np.asarray(median, dtype=np.float64)
                if median.ndim == 1:
                    median = median.reshape(-1, 1)
                qdict = {q: np.asarray(v, dtype=np.float64) for q, v in qdict.items()}
            else:
                sig = inspect.signature(model.predict)
                if "df" in sig.parameters:
                    pred_result = model.predict(horizon=horizon, df=df_hist)
                else:
                    pred_result = model.predict(horizon=horizon)
                median = np.asarray(
                    pred_result[0] if isinstance(pred_result, tuple) else pred_result,
                    dtype=np.float64,
                )
                if median.ndim == 1:
                    median = median.reshape(-1, 1)
                qdict = {}

            if median.shape[0] > horizon:
                median = median[:horizon]
            if median.shape[1] > true_vals.shape[1]:
                median = median[:, : true_vals.shape[1]]
            if true_vals.shape[1] > median.shape[1]:
                true_vals = true_vals[:, : median.shape[1]]

            preds_list.append(median)
            trues_list.append(true_vals)
            if qdict:
                qdicts_list.append(qdict)
        except Exception as e:
            FORECAST_LOG.warning("Prediction failed at origin %s: %s", origin_idx, e)
            continue

    if has_update and use_factor_updates and hasattr(model, "_model"):
        try:
            if original_factors_backup is not None and hasattr(model._model, "factors"):
                model._model.factors = (
                    original_factors_backup.copy()
                    if isinstance(original_factors_backup, np.ndarray)
                    and original_factors_backup.size > 0
                    else None
                )
            if original_innovations_backup is not None and hasattr(model._model, "innovations"):
                model._model.innovations = (
                    original_innovations_backup.copy()
                    if isinstance(original_innovations_backup, np.ndarray)
                    and original_innovations_backup.size > 0
                    else None
                )
        except Exception:
            pass

    if not preds_list:
        raise ValueError("No successful predictions.")

    preds = np.stack(preds_list, axis=0)
    trues = np.stack(trues_list, axis=0)
    metrics = _compute_forecasting_metrics(preds, trues, scaler=scaler)

    if qdicts_list and len(qdicts_list) == len(preds_list):
        crps_vals = []
        for j, qdict in enumerate(qdicts_list):
            try:
                c = crps_from_quantiles(trues[j], qdict, list(quantiles))
                crps_vals.append(c)
            except (KeyError, ValueError) as e:
                FORECAST_LOG.warning("CRPS failed origin %s: %s", j, e)
        metrics["CRPS"] = float(np.mean(crps_vals)) if crps_vals else float("nan")
    else:
        metrics["CRPS"] = float("nan")

    return metrics, preds, trues, qdicts_list if qdicts_list else None


