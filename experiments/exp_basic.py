from abc import ABC, abstractmethod

import torch


class Exp_Basic(ABC):
    """Base class for the Hydra-driven experiments; picks the device and holds the config."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self._acquire_device()
        self.model = self._build_model()
        if self.model is not None and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

    def _build_model(self):
        """Override to build a primary model. Experiments that fit several models return None."""
        return None

    def _acquire_device(self) -> torch.device:
        use_gpu = self.cfg.get("use_gpu", True) if hasattr(self.cfg, "get") else True
        if use_gpu and torch.cuda.is_available():
            gpu = int(self.cfg.get("gpu", 0)) if hasattr(self.cfg, "get") else 0
            device = torch.device(f"cuda:{gpu}")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """Main experiment entry point (called from run.py)."""
        raise NotImplementedError
