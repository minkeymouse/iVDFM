"""
Shared model fitting utilities for experiments.

Provides a unified interface for fitting different models (DFM, DDFM, VAE, iVAE, iVDFM)
across different experiment types.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import StandardScaler


def fit_model(
    model_name: str,
    df_train: pd.DataFrame,
    cfg: Union[DictConfig, dict],
    *,
    r: Optional[int] = None,
    u_train: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    scaler: Optional[StandardScaler] = None,
    encoder_layers: Optional[list] = None,
    activation_override: Optional[str] = None,
) -> object:
    """
    Fit a model on training data.
    
    This is a unified model fitting function used across different experiments.
    It handles model-specific initialization and configuration.
    
    Parameters
    ----------
    model_name : str
        One of "dfm", "ddfm", "vae", "ivae", "ivdfm"
    df_train : pd.DataFrame
        Training data (T x N)
    cfg : DictConfig or dict
        Configuration containing model settings
    r : int, optional
        Number of factors (for DFM block structure, iVDFM). If None, uses config default.
    u_train : np.ndarray, optional
        Auxiliary variables (T x aux_dim). Required for iVAE.
    device : torch.device, optional
        Device to run on. If None, auto-detects or uses CPU.
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, creates and fits a new StandardScaler.
    encoder_layers : list, optional
        Encoder layer configuration for DDFM. If None, uses config default.
    activation_override : str, optional
        Override activation function for DDFM (e.g., "tanh" for factor recovery).
        If None, uses config default.
        
    Returns
    -------
    Model
        Fitted model with factors() method (and optionally forecast() method)
        
    Raises
    ------
    ValueError
        If model_name is unsupported or required parameters are missing
    """
    # Extract model config
    if isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg.model, resolve=True)
        if encoder_layers is None:
            encoder_layers = cfg.model.get("encoder_layers", None)
    else:
        config_dict = getattr(cfg, "config_dict", None) if hasattr(cfg, "config_dict") else cfg.get("model", {}) if isinstance(cfg, dict) else {}
        if encoder_layers is None and hasattr(cfg, "encoder_layers"):
            encoder_layers = cfg.encoder_layers
    
    # Create scaler if not provided
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df_train.values)
    
    # Handle DFM block structure configuration
    if model_name == "dfm" and r is not None:
        if config_dict is None:
            config_dict = {}
        if "blocks" not in config_dict:
            config_dict["blocks"] = {}
        # Update the first block (or create default global block)
        if not config_dict["blocks"]:
            config_dict["blocks"]["global"] = {"num_factors": r, "series": []}
        else:
            # Update first block's num_factors
            first_block_name = list(config_dict["blocks"].keys())[0]
            config_dict["blocks"][first_block_name]["num_factors"] = r
    
    # Handle DDFM activation override (e.g., for factor recovery experiments)
    if model_name == "ddfm" and activation_override is not None:
        if config_dict is None:
            config_dict = {}
        config_dict["activation"] = activation_override
    
    # Fit model based on type
    if model_name == "dfm":
        from model.dfm import Model
        m = Model(config_path=None, config_dict=config_dict)
        m.fit(df_train, target_scaler=scaler)
        return m

    elif model_name == "ivdfm":
        from model.ivdfm import Model
        
        dev_str: Optional[str] = None
        try:
            if device is not None:
                dev_str = str(device)
        except Exception:
            dev_str = None

        m = Model(config_dict=config_dict, device=dev_str)

        m.fit(df_train, num_factors=r)
        return m
    
    elif model_name == "ddfm":
        from model.ddfm import Model
        m = Model(encoder_layers=encoder_layers, config_dict=config_dict)
        m.fit(df_train, target_scaler=scaler)
        return m
    
    elif model_name == "vae":
        from model.ivae import Model
        m = Model(model_type="vae", config_dict=config_dict, device=device)
        m.fit(df_train, u_train=None, target_scaler=scaler)
        return m
    
    elif model_name == "ivae":
        from model.ivae import Model
        if u_train is None:
            raise ValueError("u_train is required for iVAE model")
        m = Model(model_type="ivae", config_dict=config_dict, device=device)
        m.fit(df_train, u_train=u_train, target_scaler=scaler)
        return m
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
