"""
Thin DDFM wrapper used by this repo.

Keep it simple: expose dfm-python's original API (`fit`, `predict`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class IdentityScaler:
    """
    sklearn-like scaler that performs identity transforms.

    dfm-python's DDFM forecasting requires a dataset scaler to exist, but in this repo the
    inputs are already scaled by our data loader. This avoids double-scaling and (more
    importantly) avoids refitting/re-scaling covariates on every rolling-origin update.
    """

    n_features_in_: int

    def fit(self, X):
        X = np.asarray(X)
        self.n_features_in_ = int(X.shape[1]) if X.ndim == 2 else 1
        return self

    def transform(self, X):
        return np.asarray(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X)


class Model:
    """Minimal wrapper around `dfm_python.DDFM`."""

    def __init__(
        self,
        *,
        encoder_layers: Optional[List[int]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        self.encoder_layers = encoder_layers
        self.config_dict = config_dict
        self._model = None
        self._dataset = None

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        target_scaler: Optional[StandardScaler | RobustScaler] = None,
    ) -> "Model":
        from dfm_python.config import DDFMConfig
        from dfm_python.dataset.ddfm_dataset import DDFMDataset

        try:
            # `dfm_python.DDFM` can be set to None if optional imports fail.
            from dfm_python.models.ddfm.ddfm import DDFM  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "dfm-python DDFM is unavailable (likely missing PyTorch-related deps)."
            ) from e

        # IMPORTANT:
        # Our repo's data loader already returns *scaled* DataFrames.
        #
        # dfm-python's DDFMDataset requires a scaler to be present for forecasting, but if we pass
        # a real scaler (e.g., StandardScaler), it will refit/rescale covariates on every dataset
        # creation (including rolling-origin updates), which makes covariate scaling inconsistent.
        #
        # Use an identity scaler to satisfy dfm-python requirements while keeping everything in the
        # already-scaled space provided by our data loader.
        identity_scaler = IdentityScaler().fit(df_train.to_numpy())

        # Determine time_idx: DDFMDataset can handle datetime index directly
        # If DataFrame has datetime index, use 'index' as time_idx
        # Otherwise, look for 'date' column or create one
        df_work = df_train.copy()
        if isinstance(df_work.index, pd.DatetimeIndex):
            # DDFMDataset handles datetime index: if time_idx='index' and not in columns, uses data.index
            time_idx = 'index'
        elif 'date' in df_work.columns:
            time_idx = 'date'
        else:
            # No datetime index and no 'date' column - create one
            df_work = df_work.reset_index()
            if 'index' in df_work.columns:
                time_idx = 'index'
            else:
                # Create integer index as 'date' column
                df_work['date'] = pd.RangeIndex(start=0, stop=len(df_work))
                time_idx = 'date'

        # Targets vs covariates:
        # - default: multivariate targets (covariates=None)
        # - optional: single-target forecasting (targets={target_col}, covariates=all_except_target)
        target_col = str((self.config_dict or {}).get("target_col", "OT"))
        cov_mode = str((self.config_dict or {}).get("covariates_mode", "none")).lower()
        all_cols = [c for c in df_work.columns if c != time_idx]
        covariates = (
            [c for c in all_cols if c != target_col]
            if cov_mode in ("all_except_target", "all-except-target") and target_col in all_cols and len(all_cols) > 1
            else None
        )
        
        dataset = DDFMDataset(
            data=df_work,
            time_idx=time_idx,
            covariates=covariates,  # None = all columns are targets; list = exclude these from targets
            scaler=identity_scaler,
        )
        # Load DDFM config
        cfg_dict = self.config_dict or {}
        encoder_layers = self.encoder_layers
        if encoder_layers is not None:
            cfg_dict = dict(cfg_dict)
            cfg_dict['encoder_layers'] = encoder_layers
        cfg_obj = DDFMConfig.from_dict(cfg_dict) if cfg_dict else None

        encoder_size = tuple(encoder_layers) if encoder_layers else None
        decoder_type = getattr(cfg_obj, "decoder_type", "linear") if cfg_obj else "linear"
        
        kwargs = {}
        if cfg_obj:
            cfg = self.config_dict or {}
            kwargs = {
                "config": cfg_obj,
                "activation": getattr(cfg_obj, "activation", "relu"),
                "learning_rate": float(getattr(cfg_obj, "learning_rate", 0.001)),
                "n_mc_samples": int(getattr(cfg_obj, "n_mc_samples", 10)),
                "window_size": int(getattr(cfg_obj, "window_size", 100)),
                "max_iter": int(getattr(cfg_obj, "max_epoch", 50)),
                "tolerance": float(getattr(cfg_obj, "tolerance", 1e-5)),
                "disp": int(getattr(cfg_obj, "disp", 10)),
                "seed": int(getattr(cfg_obj, "seed", 0) or 0),
            }
            if cfg.get("max_epoch_pretrain") is not None:
                kwargs["max_epoch_pre_train"] = int(cfg["max_epoch_pretrain"])

        model = DDFM(dataset=dataset, encoder_size=encoder_size, decoder_type=decoder_type, **kwargs)
        model.fit()
        model.build_state_space()

        self._model = model
        self._dataset = dataset
        return self

    def predict(self, horizon: int, *, df: Optional[pd.DataFrame] = None, update: bool = True):
        """
        Predict future values.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon
        df : Optional[pd.DataFrame]
            Historical data for prediction. If provided, uses predict_with_history.
        update : bool
            Whether to update model state with history (default True)
            
        Returns
        -------
        np.ndarray or Tuple[np.ndarray, np.ndarray]
            Predictions. If df is provided, returns (X_forecast, Z_forecast).
            Otherwise returns X_forecast only.
        """
        if self._model is None:
            raise ValueError("Call fit() before predict().")
        if df is not None:
            # Use predict_with_history for rolling-origin evaluation
            return self.predict_with_history(horizon=horizon, df=df, update=update)
        return self._model.predict(horizon=horizon)

    def _dataset_from_history(self, df: pd.DataFrame):
        """Build DDFMDataset from history DataFrame (same columns/scale as training). Used for update and rolling-origin."""
        from dfm_python.dataset.ddfm_dataset import DDFMDataset

        df_work = df.copy()
        if isinstance(df_work.index, pd.DatetimeIndex):
            time_idx = "index"
        elif "date" in df_work.columns:
            time_idx = "date"
        else:
            df_work = df_work.reset_index()
            time_idx = "index" if "index" in df_work.columns else "date"
            if time_idx == "date" and "date" not in df_work.columns:
                df_work["date"] = pd.RangeIndex(start=0, stop=len(df_work))

        if self._dataset is not None:
            covariates = getattr(self._dataset, "covariates", None)
            scaler = getattr(self._dataset, "scaler", None)
        else:
            target_col = str((self.config_dict or {}).get("target_col", "OT"))
            cov_mode = str((self.config_dict or {}).get("covariates_mode", "none")).lower()
            all_cols = [c for c in df_work.columns if c != time_idx]
            covariates = (
                [c for c in all_cols if c != target_col]
                if cov_mode in ("all_except_target", "all-except-target") and target_col in all_cols and len(all_cols) > 1
                else None
            )
            scaler = IdentityScaler().fit(
                df_work.drop(columns=[time_idx]).to_numpy() if time_idx in df_work.columns else df_work.to_numpy()
            )
        return DDFMDataset(
            data=df_work,
            time_idx=time_idx,
            covariates=covariates,
            scaler=scaler,
        )

    def predict_with_history(self, horizon: int, df: pd.DataFrame, *, update: bool = True):
        """
        Forecast using a rolling history.

        This is intended for rolling-origin evaluation: we re-initialize/update the
        underlying dfm-python DDFM state from the provided history, then forecast.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        df : pd.DataFrame
            History up to the forecast origin (same columns/scale as training df).
        update : bool, default True
            If True, update the model's internal state using the provided history.

        Returns
        -------
        Same as dfm-python DDFM.predict(): (X_forecast, Z_forecast) by default.
        """
        if self._model is None:
            raise ValueError("Call fit() before predict().")
        dataset = self._dataset_from_history(df)
        return self._model.predict(horizon=horizon, data=dataset, update=update)

    def predict_quantiles(
        self,
        horizon: int,
        *,
        df: Optional[pd.DataFrame] = None,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        num_samples: int = 200,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """
        Probabilistic forecast via state-space simulation: Z_{t+1} = F@Z_t + w_t, y_t = H@Z_t + v_t,
        with w_t ~ N(0,Q), v_t ~ N(0,R). Returns median and quantile forecasts for CRPS evaluation.
        """
        if self._model is None:
            raise ValueError("Call fit() before predict_quantiles().")
        if df is not None:
            dataset = self._dataset_from_history(df)
            self._model._update(dataset)

        factors_avg = self._model._get_averaged_factors()
        Z_last = np.asarray(factors_avg[-1, :], dtype=np.float64) if len(factors_avg) > 0 else np.asarray(self._model.training_state.mu_0, dtype=np.float64)
        Z_last = np.nan_to_num(Z_last, nan=0.0, posinf=0.0, neginf=0.0)

        F = np.nan_to_num(np.asarray(self._model.training_state.F, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        Q = np.nan_to_num(np.asarray(self._model.training_state.Q, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        H = np.nan_to_num(np.asarray(self._model.training_state.H, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        R = np.nan_to_num(np.asarray(self._model.training_state.R, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

        m = Z_last.shape[0]
        rng = np.random.default_rng(random_seed)

        # Ensure Q is PSD for Cholesky (add small jitter if needed)
        Q_safe = Q + 1e-10 * np.eye(m, dtype=np.float64)
        try:
            LQ = np.linalg.cholesky(Q_safe)
        except np.linalg.LinAlgError:
            LQ = np.diag(np.sqrt(np.maximum(np.diag(Q_safe), 1e-12)))
        R_diag = np.maximum(np.diag(R), 1e-12)

        samples = []
        for _ in range(num_samples):
            Z = Z_last.copy()
            y_horizon = []
            for _ in range(horizon):
                w = LQ @ rng.standard_normal(m)
                Z = F @ Z + w
                v = rng.normal(0, np.sqrt(R_diag))
                y_h = (H @ Z) + v
                y_horizon.append(y_h)
            samples.append(np.stack(y_horizon))
        stack = np.stack(samples, axis=0)

        pct = [float(q * 100) for q in quantiles]
        q_arr = np.percentile(stack, pct, axis=0)
        idx_50 = next((i for i, q in enumerate(quantiles) if q == 0.5), None)
        median = np.asarray(q_arr[idx_50]) if idx_50 is not None else np.median(stack, axis=0)
        qdict = {q: np.asarray(q_arr[i]) for i, q in enumerate(quantiles)}

        if getattr(self._model, "scaler", None) is not None:
            median = self._model.scaler.inverse_transform(median)
            qdict = {q: self._model.scaler.inverse_transform(qdict[q]) for q in qdict}
        return median, qdict

    def save(self, path):
        """Save model checkpoint."""
        if self._model is None:
            raise ValueError("Call fit() before save().")
        return self._model.save(path)

    def factors(self) -> np.ndarray:
        """
        Return extracted factors (encoder latent, averaged across MC samples).
        
        Returns
        -------
        np.ndarray
            Factors (T x num_factors), averaged across MC samples if 3D
        """
        if self._model is None:
            raise ValueError("Call fit() before factors().")
        
        # Try to get factors from model directly (averaged if 3D)
        if hasattr(self._model, "factors") and self._model.factors is not None:
            f = np.asarray(self._model.factors)
            # Average across MC samples if 3D: (n_mc_samples, T, num_factors) -> (T, num_factors)
            return f.mean(axis=0) if f.ndim == 3 else f
        
        # Fallback: get factors from result
        try:
            result = self._model.get_result()
            if hasattr(result, 'Z') and result.Z is not None:
                return np.asarray(result.Z)
        except Exception:
            pass
        
        raise ValueError("Unable to extract factors from dfm-python DDFM.")

    def sample_factors(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Sample factors from MC distribution (randomly select one MC sample).
        
        For clustering visualization, this provides better representation of the
        distribution compared to averaging.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Sampled factors (T x num_factors)
        """
        if self._model is None:
            raise ValueError("Call fit() before sample_factors().")
        
        # Try to get factors from model directly
        if hasattr(self._model, "factors") and self._model.factors is not None:
            f = np.asarray(self._model.factors)
            if f.ndim == 3:
                # (n_mc_samples, T, num_factors) - randomly select one MC sample
                if random_seed is not None:
                    rng = np.random.RandomState(random_seed)
                    sample_idx = rng.randint(0, f.shape[0])
                else:
                    sample_idx = np.random.randint(0, f.shape[0])
                return f[sample_idx]  # (T, num_factors)
            else:
                # Already 2D, return as-is
                return f
        
        # Fallback: get factors from result (no MC samples available)
        try:
            result = self._model.get_result()
            if hasattr(result, 'Z') and result.Z is not None:
                return np.asarray(result.Z)
        except Exception:
            pass
        
        # If no MC samples, fall back to regular factors
        return self.factors()

