"""
NeuralForecast model wrapper for forecasting experiments.

Supports iTransformer (ITF), TimeMixer, TimeXer, and MLPMultivariate.
Follows the same API pattern as model/dfm.py and model/ddfm.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


def _convert_to_long_format(
    df: pd.DataFrame,
    target_series: List[str],
    freq: str = "H"
) -> pd.DataFrame:
    """Convert wide DataFrame to NeuralForecast long format (unique_id, ds, y).
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format DataFrame with datetime index or 'date' column
    target_series : List[str]
        List of target series column names
    freq : str, default "H"
        Frequency string for date range (H=hourly, D=daily, W=weekly, etc.)
    
    Returns
    -------
    pd.DataFrame
        Long format DataFrame with columns (unique_id, ds, y)
    """
    df_work = df.copy()
    
    # Get datetime index
    if isinstance(df_work.index, pd.DatetimeIndex):
        dates = df_work.index
    elif 'date' in df_work.columns:
        dates = pd.to_datetime(df_work['date'])
    else:
        raise ValueError("DataFrame must have datetime index or 'date' column")
    
    # Convert to long format
    nf_data = []
    for series_name in target_series:
        if series_name not in df_work.columns:
            logger.warning(f"Series {series_name} not found in DataFrame, skipping")
            continue
        
        series_data = df_work[series_name].values
        for i, (date, value) in enumerate(zip(dates, series_data)):
            nf_data.append({
                'unique_id': series_name,
                'ds': date,
                'y': value if not pd.isna(value) else None
            })
    
    nf_df = pd.DataFrame(nf_data)
    return nf_df


def _extract_forecasts(
    forecast_df: pd.DataFrame,
    target_series: List[str],
    horizon: int
) -> np.ndarray:
    """Extract forecasts from NeuralForecast output DataFrame.
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        NeuralForecast forecast DataFrame with columns (unique_id, ds, model_name, ...)
    target_series : List[str]
        List of target series names
    horizon : int
        Forecast horizon
    
    Returns
    -------
    np.ndarray
        Forecast array of shape (horizon, n_series)
    """
    # NeuralForecast returns forecasts in long format
    # We need to reshape to (horizon, n_series)
    model_col = [col for col in forecast_df.columns if col not in ['unique_id', 'ds']]
    if not model_col:
        raise ValueError("No forecast columns found in forecast DataFrame")
    model_col = model_col[0]
    
    forecasts = []
    for series_name in target_series:
        series_forecasts = forecast_df[forecast_df['unique_id'] == series_name][model_col].values
        if len(series_forecasts) < horizon:
            # Pad with NaN if needed
            series_forecasts = np.pad(
                series_forecasts,
                (0, horizon - len(series_forecasts)),
                constant_values=np.nan
            )
        elif len(series_forecasts) > horizon:
            # Truncate if needed
            series_forecasts = series_forecasts[:horizon]
        forecasts.append(series_forecasts)
    
    return np.array(forecasts).T  # Shape: (horizon, n_series)


# Quantile levels for CRPS: symmetric [0.1, 0.5, 0.9] (percent: 10, 50, 90)
CRPS_QUANTILES = (0.1, 0.5, 0.9)
CRPS_QUANTILE_LEVELS_PERCENT = [10, 50, 90]


def _quantile_col_name(model_name: str, q: float) -> str:
    """NeuralForecast column name for quantile: median, lo-10, hi-90."""
    if q == 0.5:
        return f"{model_name}-median"
    pct = int(round(q * 100))
    if q < 0.5:
        return f"{model_name}-lo-{pct}"
    return f"{model_name}-hi-{pct}"


def _infer_model_name_from_forecast(forecast_df: pd.DataFrame) -> Optional[str]:
    """Infer model name from forecast columns (e.g. 'iTransformer-median' or 'iTransformer-lo-10' -> 'iTransformer')."""
    fc_cols = [c for c in forecast_df.columns if c not in ("unique_id", "ds")]
    if not fc_cols:
        return None
    first = fc_cols[0]
    if first.endswith("-median"):
        return first[:-7]
    if "-lo-" in first:
        return first.split("-lo-")[0]
    if "-hi-" in first:
        return first.split("-hi-")[0]
    if "-" in first:
        return first.rsplit("-", 1)[0]
    return first


def _extract_quantile_forecasts(
    forecast_df: pd.DataFrame,
    target_series: List[str],
    model_name: str,
    quantiles: Tuple[float, ...] = CRPS_QUANTILES,
    horizon: int = 96,
) -> Dict[float, np.ndarray]:
    """Extract per-quantile forecasts from NeuralForecast MQLoss output. Returns dict q -> (horizon, n_series)."""
    out: Dict[float, np.ndarray] = {}
    for q in quantiles:
        col = _quantile_col_name(model_name, q)
        if col not in forecast_df.columns:
            raise ValueError(f"Quantile column '{col}' not in forecast. Columns: {list(forecast_df.columns)}")
        forecasts = []
        for series_name in target_series:
            vals = forecast_df.loc[forecast_df["unique_id"] == series_name, col].values
            if len(vals) < horizon:
                vals = np.pad(vals.astype(float), (0, horizon - len(vals)), constant_values=np.nan)
            elif len(vals) > horizon:
                vals = vals[:horizon].astype(float)
            else:
                vals = vals.astype(float)
            forecasts.append(vals)
        out[q] = np.array(forecasts).T
    return out


class Model:
    """Wrapper for NeuralForecast models: iTransformer, TimeMixer, TimeXer, MLPMultivariate."""

    def __init__(
        self,
        *,
        model_type: str = "itransformer",
        config_dict: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize NeuralForecast model wrapper.

        Parameters
        ----------
        model_type : str
            Model type: 'itransformer', 'timemixer', 'timexer', 'mlpmultivariate'
        config_dict : dict, optional
            Model configuration dictionary
        device : str, optional
            Device to use ('cpu' or 'cuda'). If None, auto-detects.
        """
        self.model_type = model_type.lower()
        valid_types = ['itransformer', 'timemixer', 'timexer', 'mlpmultivariate']
        if self.model_type not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got '{model_type}'")

        self.config_dict = config_dict or {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._nf_model = None  # NeuralForecast wrapper instance
        self._target_series = None
        self._freq = "H"  # Default hourly, can be overridden
        self.preds_in_original_units = True  # NeuralForecast handles scaling internally
        self._quantile_levels: Optional[Tuple[float, ...]] = None  # When set, use MQLoss and quantile predict
        
    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        target_scaler: Optional[StandardScaler | RobustScaler] = None,
        seed: Optional[int] = None,
    ) -> "Model":
        """
        Fit NeuralForecast model on training data.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            Training data with datetime index or 'date' column
        target_scaler : StandardScaler | RobustScaler, optional
            Scaler (for compatibility, but NeuralForecast handles scaling internally)
        seed : int, optional
            Random seed for reproducible training. Sets PyTorch Lightning seed_everything
            before model construction and fit. Required for reproducible training.
            
        Returns
        -------
        Model
            Self for method chaining
        """
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import (
                iTransformer, TimeMixer, TimeXer, MLPMultivariate
            )
            from neuralforecast.losses.pytorch import MQLoss
        except ImportError as e:
            raise ImportError(
                "NeuralForecast is not installed. Install with: pip install neuralforecast"
            ) from e

        if seed is not None:
            from pytorch_lightning import seed_everything
            seed_everything(seed, workers=True)
        
        # Extract configuration
        h = int(self.config_dict.get('h', 96))  # Forecast horizon
        input_size = int(self.config_dict.get('input_size', 2 * h))
        max_steps = int(self.config_dict.get('max_steps', 1000))
        scaler_type = self.config_dict.get('scaler_type')
        quantile_levels_percent = self.config_dict.get("quantile_levels")
        if quantile_levels_percent is not None:
            quantile_levels_percent = list(quantile_levels_percent)
            self._quantile_levels = tuple(q / 100.0 for q in quantile_levels_percent)
        else:
            self._quantile_levels = None
        
        # Get target series
        df_work = df_train.copy()
        if isinstance(df_work.index, pd.DatetimeIndex):
            dates = df_work.index
        elif 'date' in df_work.columns:
            dates = pd.to_datetime(df_work['date'])
            df_work = df_work.set_index('date')
        else:
            raise ValueError("DataFrame must have datetime index or 'date' column")
        
        # Determine frequency
        if len(dates) > 1:
            freq_diff = dates[1] - dates[0]
            if freq_diff <= pd.Timedelta(hours=1):
                self._freq = "H"
            elif freq_diff <= pd.Timedelta(days=1):
                self._freq = "D"
            elif freq_diff <= pd.Timedelta(weeks=1):
                self._freq = "W"
            else:
                self._freq = "M"
        
        # Get all numeric columns as targets (exclude date columns)
        date_cols = {'date', 'Date', 'date_w', 'date_m', 'Date_w', 'Date_m', 'year', 'month', 'day'}
        all_cols = [col for col in df_work.columns if col not in date_cols]
        self._target_series = all_cols
        
        # Convert to long format
        nf_train = _convert_to_long_format(df_work, self._target_series, self._freq)
        
        # Create model instance based on type
        # Note: val_size is handled by NeuralForecast.fit(), not passed to model constructor
        model_kwargs = {
            'h': h,
            'input_size': input_size,
            'max_steps': max_steps,
            'scaler_type': scaler_type,
        }
        if seed is not None:
            model_kwargs['random_seed'] = seed
        if self._quantile_levels is not None:
            levels_pct = [int(round(q * 100)) for q in self._quantile_levels]
            model_kwargs['loss'] = MQLoss(level=levels_pct)

        # Add model-specific parameters
        if self.model_type == 'itransformer':
            model_kwargs.update({
                'n_series': len(self._target_series),
            })
            model = iTransformer(**model_kwargs)
        elif self.model_type == 'timemixer':
            model_kwargs.update({
                'n_series': len(self._target_series),
            })
            model = TimeMixer(**model_kwargs)
        elif self.model_type == 'timexer':
            model_kwargs.update({
                'n_series': len(self._target_series),
            })
            model = TimeXer(**model_kwargs)
        elif self.model_type == 'mlpmultivariate':
            model_kwargs.update({
                'n_series': len(self._target_series),
                'num_layers': int(self.config_dict.get('num_layers', 2)),
                'hidden_size': int(self.config_dict.get('hidden_size', 1024)),
            })
            model = MLPMultivariate(**model_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Create NeuralForecast wrapper (seed_everything above ensures reproducible fit)
        self._nf_model = NeuralForecast(models=[model], freq=self._freq)
        
        # Fit model
        logger.info(f"Fitting {self.model_type.upper()} model on {len(self._target_series)} series...")
        self._nf_model.fit(df=nf_train)
        
        # Store training data for prediction (if needed)
        self._last_train_df = df_work
        
        self._model = model
        logger.info(f"Model fitted successfully")
        
        return self
    
    def predict(self, horizon: Optional[int] = None, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecasts. When fitted with quantile_levels, returns median forecast.
        
        Parameters
        ----------
        horizon : int, optional
            Forecast horizon. If None, uses model's h parameter.
        df : pd.DataFrame, optional
            Data for prediction. If None, tries to use stored data.
            Must be in wide format with datetime index.
            
        Returns
        -------
        np.ndarray
            Forecast array of shape (horizon, n_series)
        """
        median, _ = self.predict_quantiles(horizon=horizon, df=df)
        return median

    def predict_quantiles(
        self,
        horizon: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, Optional[Dict[float, np.ndarray]]]:
        """
        Generate forecasts. When fitted with quantile_levels, returns (median, quantile_dict).
        Otherwise returns (point_forecast, None).
        
        Returns
        -------
        median : np.ndarray
            Shape (horizon, n_series). Median or point forecast.
        quantile_dict : dict or None
            Maps q (e.g. 0.1, 0.5, 0.9) to (horizon, n_series). None when not probabilistic.
        """
        if self._nf_model is None:
            raise ValueError("Call fit() before predict().")
        
        if horizon is None:
            horizon = getattr(self._model, 'h', 96)
        
        if df is not None:
            nf_df = _convert_to_long_format(df, self._target_series, self._freq)
        elif hasattr(self, '_last_train_df'):
            nf_df = _convert_to_long_format(self._last_train_df, self._target_series, self._freq)
        else:
            raise ValueError(
                "NeuralForecast models require data for prediction. "
                "Please provide df parameter with historical data."
            )
        
        forecast_df = self._nf_model.predict(df=nf_df)
        
        if self._quantile_levels is not None:
            model_name = _infer_model_name_from_forecast(forecast_df)
            if model_name is None:
                raise ValueError("Could not infer model name from forecast columns")
            qdict = _extract_quantile_forecasts(
                forecast_df, self._target_series, model_name,
                quantiles=self._quantile_levels, horizon=horizon,
            )
            median = qdict[0.5]
            return median, qdict
        forecasts = _extract_forecasts(forecast_df, self._target_series, horizon)
        return forecasts, None
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to save checkpoint
        """
        if self._nf_model is None:
            raise ValueError("Model not fitted. Call fit() before save().")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # NeuralForecast models can be saved using their save method
        # We'll save the model state and metadata
        checkpoint = {
            'model_type': self.model_type,
            'config_dict': self.config_dict,
            'target_series': self._target_series,
            'freq': self._freq,
            'model_state': self._model.state_dict() if hasattr(self._model, 'state_dict') else None,
        }
        
        import torch
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "Model":
        """
        Load model from checkpoint.
        
        Parameters
        ----------
        path : str or Path
            Path to checkpoint file
            
        Returns
        -------
        Model
            Loaded model instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        import torch
        checkpoint = torch.load(path, map_location='cpu')
        
        # Recreate model instance
        model = cls(
            model_type=checkpoint['model_type'],
            config_dict=checkpoint['config_dict'],
        )
        model._target_series = checkpoint['target_series']
        model._freq = checkpoint['freq']
        
        # Note: NeuralForecast models need to be refitted or loaded differently
        # This is a simplified load - in practice, you may need to refit
        logger.warning(
            "NeuralForecast model loaded. You may need to refit the model "
            "or use NeuralForecast's native loading mechanism."
        )
        
        return model
