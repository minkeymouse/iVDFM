"""
Full-series data loading utilities for DFM / DDFM experiments.

This repository originally copied iTransformer's windowed PyTorch datasets into
`data_provider/data_loader.py`. For our DFM/DDFM-only framework we do NOT use:
- windowing (seq_len/label_len/pred_len slicing)
- time features / `timeenc`
- PyTorch `Dataset` / `DataLoader`

Instead we expose simple helpers to load CSV → split → (optionally) scale, and
return a pandas DataFrame indexed by datetime for direct use with dfm-python's
`DFMDataset` / `DDFMDataset`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal, Tuple

import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

PIPELINE_LOG = logging.getLogger("psdfm.forecasting.pipeline")

Split = Literal["train", "val", "test"]
ScalerKind = Literal["standard", "robust"]


@dataclass
class SeriesData:
    df: pd.DataFrame
    scaler: StandardScaler | RobustScaler

    def inverse_transform(self, data):
        """Inverse-transform scaled data back to original space."""
        return self.scaler.inverse_transform(data)


def _split_borders(n: int) -> dict[str, Tuple[int, int]]:
    """70/10/20 train/val/test split for all datasets."""
    num_train = int(n * 0.7)
    num_test = int(n * 0.2)
    num_val = n - num_train - num_test
    return {
        "train": (0, num_train),
        "val": (num_train, num_train + num_val),
        "test": (n - num_test, n),
    }


def _select_columns(df_raw: pd.DataFrame, target: str) -> list[str]:
    if "date" not in df_raw.columns:
        raise ValueError("Expected 'date' column in CSV.")
    cols = list(df_raw.columns)
    cols.remove("date")
    return cols


def load_full_series(
    *,
    name: str,
    root_path: str,
    data_path: str,
    split: Split,
    target: str = "OT",
    scaler: ScalerKind = "standard",
) -> SeriesData:
    """Load CSV, split, scale (always on); return DataFrame indexed by datetime."""
    csv_path = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(csv_path)

    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values("date")

    cols = _select_columns(df_raw, target=target)
    borders_all = _split_borders(len(df_raw))
    b1, b2 = borders_all[split]

    # Always scale (fit on train split only) — unified for all models
    data_values = df_raw[cols].to_numpy()
    sc = StandardScaler() if scaler == "standard" else RobustScaler()
    tr1, tr2 = borders_all["train"]
    sc.fit(data_values[tr1:tr2])
    data_values = sc.transform(data_values)

    PIPELINE_LOG.info(
        "[DATA] scaler=%s fitted on train only (rows %d:%d), n_features=%d; "
        "all splits (train/val/test) use same transform for fair comparison.",
        scaler, tr1, tr2, len(cols),
    )
    df = pd.DataFrame(data_values[b1:b2], columns=cols, index=df_raw["date"].iloc[b1:b2].to_numpy())
    df.index = pd.to_datetime(df.index)
    PIPELINE_LOG.info("[DATA] split=%s shape=(%d, %d) (scaled).", split, len(df), len(cols))
    return SeriesData(df=df, scaler=sc)

def load_data(
    *,
    name: str,
    root_path: str,
    data_path: str,
    split: Split,
    target: str = "OT",
    scaler: ScalerKind = "standard",
) -> Tuple["pd.DataFrame", StandardScaler | RobustScaler]:
    """
    Load CSV → split → scale (always on). Returns DataFrame indexed by datetime.

    Notes:
    - scaler is always fitted on the train split, even when split != 'train'
    - This loader is multivariate-only (DFM/DDFM consume full-series matrices).
    """
    series: SeriesData = load_full_series(
        name=name,
        root_path=root_path,
        data_path=data_path,
        split=split,
        target=target,
        scaler=scaler,
    )
    return series.df, series.scaler


def inverse_transform(scaler: StandardScaler | RobustScaler, data):
    """
    Convenience helper matching iTransformer's Dataset.inverse_transform().

    Since this repo returns `(df, scaler)` instead of a Dataset object, you can call:
        `inverse_transform(scaler, arr)`
    """
    return scaler.inverse_transform(data)

