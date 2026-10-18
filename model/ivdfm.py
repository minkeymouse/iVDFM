"""
Thin iVDFM wrapper used by this repo.

This wraps dfm-python's `iVDFM` implementation and exposes a minimal interface
compatible with our experiments (`fit`, `factors`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class Model:
    """Minimal wrapper around `dfm_python.models.ivdfm.iVDFM`."""

    def __init__(
        self,
        *,
        config_dict: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        import torch
        
        self.config_dict = dict(config_dict) if config_dict is not None else {}
        
        # Force GPU usage for IVDFM when available, regardless of device parameter
        # This matches the behavior of VAE/iVAE for fair comparison
        if torch.cuda.is_available():
            # Always prefer GPU for IVDFM when available
            if device is not None and "cuda" in str(device).lower():
                self.device = str(device)  # Use provided GPU device
            else:
                self.device = "cuda:0"  # Force GPU even if CPU was passed
            print(f"  [IVDFM] Using GPU: {self.device}")
        else:
            # Fallback to CPU only if GPU is not available
            self.device = device or "cpu"
            print(f"  [IVDFM] Using CPU (GPU not available)")
        
        self._model = None
        self._result = None

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        num_factors: Optional[int] = None,
        context: Optional[list[str]] = None,
    ) -> "Model":
        import torch

        from dfm_python.dataset.ivdfm_dataset import iVDFMDataset
        from dfm_python.models.ivdfm.ivdfm import iVDFM

        # Convert device string to torch.device
        # self.device is already set to "cuda:0" if GPU is available (from __init__)
        torch_device = None
        if self.device:
            try:
                torch_device = torch.device(self.device)
            except Exception:
                # Fallback: try to use GPU if available
                if torch.cuda.is_available():
                    torch_device = torch.device("cuda:0")
                else:
                    torch_device = torch.device("cpu")

        # Config is optional; we allow running with constructor defaults.
        # If config_dict has iVDFM fields, we pass it as **kwargs to iVDFM.
        cfg_kwargs: Dict[str, Any] = dict(self.config_dict or {})

        # Ensure we don't pass unknown keys that collide with our explicit args.
        # (Keep this conservative: iVDFM accepts **kwargs, so extra keys are ok.)
        T = int(len(df_train))
        if T < 2:
            raise ValueError("iVDFM requires at least 2 time steps")

        # iVDFM expects sequences; dataset constructs sliding windows.
        # When window=None, the package will use full T (full series as one sequence).
        window = cfg_kwargs.pop("window", cfg_kwargs.pop("sequence_length", None))  # Backward compat
        # Only clamp if window is explicitly provided and > T
        # If None, pass through to package which will convert to T
        if window is not None:
            window = int(max(2, min(window, T)))
        stride = int(cfg_kwargs.pop("stride", 1))

        # Targets vs covariates:
        # - default: multivariate targets (covariates=None)
        # - optional: single-target forecasting (targets={target_col}, covariates=all_except_target)
        target_col = str(cfg_kwargs.get("target_col", "OT"))
        cov_mode = str(cfg_kwargs.get("covariates_mode", "none")).lower()
        all_cols = [c for c in df_train.columns if c not in ("date", "index")]
        covariates_list = (
            [c for c in all_cols if c != target_col]
            if cov_mode in ("all_except_target", "all-except-target") and target_col in all_cols and len(all_cols) > 1
            else None
        )
        
        # Build dataset:
        # - covariates: excluded from targets (used for factor extraction but not forecasted)
        # - if `context` columns are provided, dataset will extract them as exogenous variables
        # - time_context controls time-based features (default=1: time step only, >1 adds sine features)
        # - Final context = [time_features(time_context dim)] + [exogenous context columns]
        # Keep time_context consistent between dataset and model (prevents misleading warnings)
        time_context = int(cfg_kwargs.get("time_context", cfg_kwargs.get("context_dim", 1)))  # backward compat
        # Avoid passing duplicates via **cfg_kwargs below
        cfg_kwargs.pop("time_context", None)
        dataset = iVDFMDataset(
            data=df_train,
            window=window,
            stride=stride,
            covariates=covariates_list,  # Columns excluded from targets
            context=context,  # Auxiliary context columns (separate from covariates)
            time_context=time_context,
            device=torch_device,
        )

        r = int(cfg_kwargs.pop("num_factors", num_factors if num_factors is not None else 2))

        model = iVDFM(
            dataset=dataset,
            num_factors=r,
            time_context=time_context,
            device=torch_device,
            **cfg_kwargs,
        )
        model.fit()

        self._model = model
        try:
            self._result = model.get_result()
        except Exception:
            self._result = None
        return self

    def predict(
        self,
        horizon: int,
        *,
        df: Optional[pd.DataFrame] = None,
        context_data: Optional[Any] = None,
        context: Optional[list[str]] = None,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict future values.
        
        Parameters
        ----------
        horizon : int
            Number of periods ahead to forecast
            
        Returns
        -------
        np.ndarray
            Forecasted values of shape (horizon, data_dim)
        """
        if self._model is None:
            raise ValueError("Call fit() before predict().")
        # dfm-python iVDFM supports passing `data` (e.g., a DataFrame) to extract
        # the last factor state for prediction. This is needed for rolling-origin
        # evaluation so each origin conditions on the correct history.
        if df is None and context_data is None and context is None:
            return self._model.predict(horizon=horizon)
        return self._model.predict(
            data=df,
            context_data=context_data,
            context=context,
            horizon=horizon,
            deterministic=deterministic,
        )

    def predict_quantiles(
        self,
        horizon: int,
        *,
        df: Optional[pd.DataFrame] = None,
        context_data: Optional[Any] = None,
        context: Optional[list[str]] = None,
        quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
        num_samples: int = 200,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
        """
        Probabilistic forecast via sampling; returns (median, quantile_dict) for CRPS.

        Calls predict(..., deterministic=False) num_samples times with different
        seeds, then computes empirical quantiles. quantile_dict maps q -> (horizon, n_series).
        """
        if self._model is None:
            raise ValueError("Call fit() before predict_quantiles().")
        base = (random_seed if random_seed is not None else 42) & 0xFFFFFFFF
        samples = []
        for i in range(num_samples):
            import torch
            s = base + i
            np.random.seed(s)
            torch.manual_seed(s)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)
            pred = self._model.predict(
                data=df,
                context_data=context_data,
                context=context,
                horizon=horizon,
                deterministic=False,
                num_samples=1,
            )
            samples.append(pred)
        stack = np.stack(samples, axis=0)
        pct = [float(q * 100) for q in quantiles]
        q_arr = np.percentile(stack, pct, axis=0)
        idx_50 = next((i for i, q in enumerate(quantiles) if q == 0.5), None)
        median = q_arr[idx_50] if idx_50 is not None else np.median(stack, axis=0)
        qdict = {q: np.asarray(q_arr[i]) for i, q in enumerate(quantiles)}
        return median, qdict

    def save(self, path, *, weights_only: bool = False) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to save checkpoint
        weights_only : bool, default False
            If True, save only model weights (state_dict) to .pt file.
            If False, save full checkpoint with config and training state.
        """
        if self._model is None:
            raise ValueError("Call fit() before save().")
        return self._model.save(path, weights_only=weights_only)

    def factors(self) -> np.ndarray:
        if self._model is None:
            raise ValueError("Call fit() before factors().")

        res = self._result
        if res is None:
            res = self._model.get_result()

        Z = getattr(res, "Z", None)
        if Z is None:
            raise ValueError("iVDFM result does not contain factors (Z).")

        Z = np.asarray(Z)
        # Handle possible batched output: (batch, T, r) -> (T, r)
        if Z.ndim == 3:
            Z = Z.mean(axis=0)
        return Z

    def sample_factors(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Sample factors from the stochastic distribution.
        
        For iVDFM, factors are stochastic because they are linear combinations of
        stochastic innovations: f_t = A f_{t-1} + B η_t, where η_t ~ p(η_t | u_t).
        
        This method samples innovations (either from prior p(η_t | u_t) or encoder
        posterior q(η_t | y_t, u_t)) and reconstructs factors deterministically using
        the linear dynamics.
        
        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        np.ndarray
            Sampled factors (T x r)
        """
        if self._model is None:
            raise ValueError("Call fit() before sample_factors().")
        
        # Set random seed if provided
        if random_seed is not None:
            import torch
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        
        import torch
        from dfm_python.config.types import to_tensor, to_numpy
        
        self._model.eval()
        with torch.no_grad():
            # Get stored innovations from model (these are from training forward pass)
            # They are already stochastic samples from q(η_t | y_t, u_t)
            stored_innovations = self._model.innovations
            
            if stored_innovations is not None:
                stored_innovations = np.asarray(stored_innovations)
                # Handle batched: (batch, T, r) -> (T, r)
                if stored_innovations.ndim == 3:
                    # For clustering, we want one sample, so randomly pick one batch
                    if random_seed is not None:
                        rng = np.random.RandomState(random_seed)
                        batch_idx = rng.randint(0, stored_innovations.shape[0])
                    else:
                        batch_idx = np.random.randint(0, stored_innovations.shape[0])
                    stored_innovations = stored_innovations[batch_idx]  # (T, r)
                elif stored_innovations.ndim == 2:
                    # Already (T, r) - use directly
                    pass
                else:
                    raise ValueError(f"Unexpected innovations shape: {stored_innovations.shape}")
                
                # Convert to tensor and reconstruct factors using SSM
                eta_tensor = to_tensor(stored_innovations, device=self._model.device)
                if eta_tensor.ndim == 2:
                    eta_tensor = eta_tensor.unsqueeze(0)  # (1, T, r)
                
                # Reconstruct factors deterministically from innovations
                # f_t = convolution of innovations (linear dynamics)
                factors_tensor = self._model.ssm.forward(eta_tensor)  # (1, T, r)
                factors = to_numpy(factors_tensor.squeeze(0))  # (T, r)
                
                return factors
            
            # Fallback: if no stored innovations, return regular factors
            # (This shouldn't happen if model was properly trained)
            return self.factors()

    def update(self, df: pd.DataFrame, **kwargs) -> None:
        """Update model factors with new data (online learning)."""
        if self._model is None:
            raise ValueError("Call fit() before update().")
        return self._model.update(df, **kwargs)

    def predict_irf(
        self,
        do_time: int,
        do_dim: int,
        do_value: float,
        horizon: int,
    ) -> np.ndarray:
        """
        Model-implied impulse response: do(η_{do_time, do_dim} = do_value), return
        observation-level IRF (horizon, N) as difference vs baseline.

        Requires fit() first. Uses stored factors and innovations from the last fit.
        """
        if self._model is None:
            raise ValueError("Call fit() before predict_irf().")
        from dfm_python.config.types import to_tensor, to_numpy

        self._model.eval()
        import torch
        from dfm_python.config.constants import DEFAULT_TORCH_DTYPE
        dtype = DEFAULT_TORCH_DTYPE if DEFAULT_TORCH_DTYPE is not None else torch.float32
        device = self._model.device

        F = self._model.factors
        eta = self._model.innovations
        if F is None or eta is None:
            raise ValueError("Factors/innovations not available; run fit() first.")
        F = np.asarray(F, dtype=np.float32)
        eta = np.asarray(eta, dtype=np.float32)
        if F.ndim == 3:
            F = np.mean(F, axis=0)
        if eta.ndim == 3:
            eta = np.mean(eta, axis=0)
        T, r = F.shape
        if do_time < 0 or do_time >= T or do_dim < 0 or do_dim >= r:
            raise ValueError("do_time or do_dim out of range.")
        end = min(do_time + horizon, T)

        eta_do = eta.copy()
        eta_do[do_time, do_dim] = do_value

        eta_t = to_tensor(eta_do, device=device, dtype=dtype)
        if eta_t.dim() == 2:
            eta_t = eta_t.unsqueeze(0)
        with torch.no_grad():
            factors_do = self._model.ssm.forward(eta_t)
        factors_do = factors_do.squeeze(0)
        F_t = to_tensor(F, device=device, dtype=dtype).unsqueeze(0)
        u = np.arange(T, dtype=np.float32) / max(T - 1, 1)
        context_dim = getattr(self._model, "context_dim", 1)
        if context_dim > 1:
            u = np.stack([u] * context_dim, axis=1)
        else:
            u = u.reshape(-1, 1)
        u_t = to_tensor(u, device=device, dtype=dtype).unsqueeze(0)

        y_baseline = self._model._decode(F_t, u_t).squeeze(0)
        y_do = self._model._decode(factors_do.unsqueeze(0), u_t).squeeze(0)
        y_baseline = to_numpy(y_baseline)
        y_do = to_numpy(y_do)

        irf = y_do[do_time:end] - y_baseline[do_time:end]
        if irf.shape[0] < horizon:
            pad = np.zeros((horizon - irf.shape[0], irf.shape[1]), dtype=irf.dtype)
            irf = np.concatenate([irf, pad], axis=0)
        return irf[:horizon]
