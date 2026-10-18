# synthetic/base.py

from abc import ABC, abstractmethod
import numpy as np


class BaseSynthetic(ABC):
    """
    Minimal abstract base class for synthetic data generators.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def generate(self) -> dict:
        """
        Generate synthetic data.

        Must return a dict with at least:
            - 'x': observed data (np.ndarray)

        May also include:
            - 'latent': ground-truth latent variables
            - 'u': auxiliary variables
        """
        pass

    # ---- optional shared helpers ----

    def add_noise(self, x: np.ndarray, noise_std: float = 0.0) -> np.ndarray:
        if noise_std <= 0:
            return x
        return x + noise_std * self.rng.randn(*x.shape)

    def add_missing(self, x: np.ndarray, missing_rate: float = 0.0) -> np.ndarray:
        if missing_rate <= 0:
            return x
        x = x.copy()
        n = x.size
        k = int(n * missing_rate)
        idx = self.rng.choice(n, k, replace=False)
        x.flat[idx] = np.nan
        return x
