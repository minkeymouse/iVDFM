import random

import numpy as np
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _apply_common_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cleanup_resources() -> None:
    """Clean up GPU resources and processes after experiment."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    except Exception:
        pass


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Let Hydra handle composition; config controls chdir/output dirs.
    print("Hydra config loaded, starting experiment...", flush=True)
    OmegaConf.set_struct(cfg, False)
    _apply_common_seed(int(cfg.get("seed", 2026)))

    try:
        exp_choice = str(HydraConfig.get().runtime.choices.get("experiment"))
    except Exception:
        exp_choice = None

    try:
        if exp_choice == "factor_recovery":
            from experiments.exp_factor_recovery import Exp_Factor_Recovery
            exp = Exp_Factor_Recovery(cfg)
            exp.run()
        elif exp_choice == "causal_intervention":
            from experiments.exp_causal_intervention import Exp_Causal_Intervention
            exp = Exp_Causal_Intervention(cfg)
            exp.run()
        elif exp_choice == "forecasting":
            from experiments.exp_forecasting import Exp_Forecasting
            exp = Exp_Forecasting(cfg)
            exp.run()
        elif exp_choice == "application":
            from experiments.exp_application import Exp_Application
            exp = Exp_Application(cfg)
            exp.run()
        else:
            raise ValueError(f"Unknown experiment: {exp_choice}. Choose factor_recovery, causal_intervention, forecasting, or application.")
    finally:
        # Always cleanup resources, even if experiment fails
        _cleanup_resources()


if __name__ == "__main__":
    main()
