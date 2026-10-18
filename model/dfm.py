"""
Thin DFM wrapper used by this repo.

Keep it simple: expose dfm-python's original API (`fit`, `predict`, `update`).
Matches the API pattern from dfm-python tutorials.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class Model:
    """Minimal wrapper around `dfm_python.DFM`."""

    def __init__(self, *, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        from dfm_python.config import DFMConfig

        if config_path:
            import yaml

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.config = DFMConfig.from_dict(cfg)
        elif config_dict is not None:
            self.config = DFMConfig.from_dict(config_dict)
        else:
            raise ValueError("Provide config_path or config_dict.")

        self._model = None
        self._dataset = None
        # dfm-python DFM can inverse-transform forecasts if dataset.target_scaler is provided.
        self.preds_in_original_units = True

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        target_scaler: Optional[StandardScaler | RobustScaler] = None,
    ) -> "Model":
        """
        Fit DFM model on training data.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            Training data. Must have a datetime index or a 'date' column.
            If index is datetime, it will be converted to a 'date' column.
        target_scaler : StandardScaler | RobustScaler, optional
            Scaler for target series. If None, StandardScaler will be used.
            
        Returns
        -------
        Model
            Self for method chaining
        """
        from dfm_python import DFM, DFMDataset

        # Prepare DataFrame: ensure time_index is a column
        df_work = df_train.copy()
        
        # Check if 'date' column already exists
        if 'date' in df_work.columns:
            time_index_col = 'date'
        # If DataFrame has datetime index, convert to 'date' column
        elif isinstance(df_work.index, pd.DatetimeIndex):
            index_name = df_work.index.name
            df_work = df_work.reset_index()
            # Rename the reset index column to 'date'
            if index_name and index_name in df_work.columns:
                df_work = df_work.rename(columns={index_name: 'date'})
            elif 'index' in df_work.columns:
                df_work = df_work.rename(columns={'index': 'date'})
            time_index_col = 'date'
        # If index has a name, use it as time column
        elif df_work.index.name:
            index_name = df_work.index.name
            df_work = df_work.reset_index()
            time_index_col = index_name
        # Otherwise, create a simple integer index as 'date'
        else:
            df_work = df_work.reset_index()
            if 'index' in df_work.columns:
                df_work = df_work.rename(columns={'index': 'date'})
            else:
                df_work['date'] = pd.RangeIndex(start=0, stop=len(df_work))
            time_index_col = 'date'

        # Allow blocks series to be omitted: interpret [] as all columns.
        # Exclude time_index_col from series list
        numeric_cols = [c for c in df_work.columns if c != time_index_col and pd.api.types.is_numeric_dtype(df_work[c])]
        
        for _block_name, block_cfg in self.config.blocks.items():
            series = block_cfg.get("series", [])
            if isinstance(series, list) and len(series) == 0:
                block_cfg["series"] = numeric_cols

        # Create scaler if not provided
        # Note: dfm-python expects reasonably scaled numeric inputs for stable Kalman smoothing.
        # We therefore fit+transform the numeric columns here (excluding the time index).
        scaler = target_scaler if target_scaler is not None else StandardScaler()

        # Fit/transform numeric columns (excluding time index)
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for DFM fitting.")
        df_work.loc[:, numeric_cols] = scaler.fit_transform(df_work[numeric_cols].values)

        # Create dataset (matches tutorial API)
        dataset = DFMDataset(
            config=self.config,
            data=df_work,
            time_index=time_index_col,
        )

        # Create and fit model (matches tutorial API)
        model = DFM(
            dataset=dataset,
            config=self.config,
            scaler=scaler,
        )
        model.fit()

        self._model = model
        self._dataset = dataset
        return self

    def update(self, data_new: Union[pd.DataFrame, np.ndarray]) -> None:
        """
        Update model state with new observations.
        
        Parameters
        ----------
        data_new : pd.DataFrame or np.ndarray
            New preprocessed observations (T_new x N). Must be preprocessed but NOT standardized.
        """
        if self._model is None:
            raise ValueError("Call fit() before update().")
        self._model.update(data_new)

    def predict(self, horizon: int):
        """
        Forecast future values.
        
        Parameters
        ----------
        horizon : int
            Number of periods ahead to forecast.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (X_forecast, Z_forecast) where:
            - X_forecast: (horizon x N) forecasted series
            - Z_forecast: (horizon x m) forecasted factors
        """
        if self._model is None:
            raise ValueError("Call fit() before predict().")
        return self._model.predict(horizon=horizon)

    def factors(self) -> np.ndarray:
        """
        Return smoothed **primary** factors.
        
        Returns
        -------
        np.ndarray
            Smoothed factors (T x r), where r is the number of
            conceptual factors (sum of r per block).
        """
        if self._model is None:
            raise ValueError("Call fit() before factors().")
        res = self._model.result
        Z = np.asarray(res.Z)
        # For DFM, Z contains the full Kalman state (companion-form factors
        # plus idiosyncratic components). We only want the primary factor
        # block for evaluation.
        n_factors = Z.shape[1]  # Default: return all columns
        
        # Try to get the actual number of factors from result
        try:
            if hasattr(res, 'num_factors'):
                n_factors = res.num_factors()
            elif res.r is not None:
                # res.r is an array of factors per block
                n_factors = int(np.sum(res.r))
            elif hasattr(self._model, 'num_factors'):
                # Fallback to model's num_factors attribute
                n_factors = int(self._model.num_factors)
            elif hasattr(self.config, 'blocks'):
                # Fallback to config: sum num_factors from all blocks
                total = 0
                for block_cfg in self.config.blocks.values():
                    total += int(block_cfg.get('num_factors', 0))
                if total > 0:
                    n_factors = total
        except Exception:
            # If all fallbacks fail, use Z.shape[1] (already set above)
            pass
        
        # Ensure we don't exceed Z's actual dimension
        n_factors = min(n_factors, Z.shape[1])
        # Also ensure we return at least 1 column
        n_factors = max(1, n_factors)
        
        return Z[:, :n_factors]
    
    def sample_factors(self, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Sample factors from the distribution.
        
        For DFM, factors are deterministic (Kalman smoothed), so this returns
        the same as factors() but provides consistent API for clustering visualization.
        
        Parameters
        ----------
        random_seed : int, optional
            Ignored (DFM is deterministic)
            
        Returns
        -------
        np.ndarray
            Factors (T x r) - same as factors() for deterministic DFM
        """
        # DFM factors are deterministic (Kalman smoothed), so just return factors
        return self.factors()

