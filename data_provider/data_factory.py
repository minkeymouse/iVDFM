"""
Load a benchmark dataset split as a scaled DataFrame (all models consume full series, not windows).
"""

from __future__ import annotations

from typing import Literal, Tuple

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data_loader import load_data, PIPELINE_LOG

Split = Literal["train", "val", "test"]


def data_provider(cfg, flag: Split) -> Tuple[pd.DataFrame, StandardScaler | RobustScaler]:
    """
    Returns:
      - df: preprocessed (scaled) DataFrame indexed by datetime
      - scaler: train-fitted scaler
    """
    from omegaconf import DictConfig
    
    if isinstance(cfg, DictConfig):
        dataset_name = str(cfg.dataset.get("name", "custom"))
        scaler_type = str(cfg.dataset.get("scaler", "standard"))
        PIPELINE_LOG.info("[DATA_PROVIDER] dataset=%s split=%s scaler=%s (unified for all models).", dataset_name, flag, scaler_type)
        df, scaler = load_data(
            name=dataset_name,
            root_path=str(cfg.dataset.get("root_path", "./data")),
            data_path=str(cfg.dataset.get("data_path", "")),
            split=flag,
            target=str(cfg.dataset.get("target", "OT")),
            scaler=scaler_type,
        )
    else:
        dataset_name = getattr(cfg, "data", "custom")
        scaler_type = str(getattr(cfg, "scaler", "standard"))
        PIPELINE_LOG.info("[DATA_PROVIDER] dataset=%s split=%s scaler=%s (unified for all models).", dataset_name, flag, scaler_type)
        df, scaler = load_data(
            name=cfg.data,
            root_path=cfg.root_path,
            data_path=cfg.data_path,
            split=flag,
            target=cfg.target,
            scaler=scaler_type,
        )
    return df, scaler
