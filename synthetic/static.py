# synthetic/ivae.py

import numpy as np
from .base import BaseSynthetic


class IVAESynthetic(BaseSynthetic):
    """
    Synthetic data generator for iVAE / nonlinear ICA experiments.

    DGP:
        z_t | u_t ~ prod_i N(mu_i(u_t), sigma_i^2(u_t))
        x_t = f(z_t)

    - Latents are conditionally independent given u
    - Time is only an index (piecewise stationary)
    - Ground truth = latent components z
    """

    def __init__(
        self,
        n_per_seg: int,
        n_seg: int,
        latent_dim: int,
        data_dim: int | None = None,
        n_layers: int = 2,
        activation: str = "tanh",
        noise_std: float = 0.0,
        missing_rate: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        self.n_per_seg = n_per_seg
        self.n_seg = n_seg
        self.latent_dim = latent_dim
        self.data_dim = data_dim or latent_dim
        self.n_layers = n_layers
        self.activation = activation
        self.noise_std = noise_std
        self.missing_rate = missing_rate

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        z, u = self._generate_latents()
        x = self._mix_nonlinear(z)

        x = self.add_noise(x, self.noise_std)
        x = self.add_missing(x, self.missing_rate)

        return {
            "x": x,
            "latent": z,
            "u": u,
        }

    # ------------------------------------------------------------------
    # Latent generation
    # ------------------------------------------------------------------

    def _generate_latents(self):
        """
        Generate piecewise-stationary, conditionally independent sources.
        """
        n = self.n_per_seg * self.n_seg

        # segment labels
        u = np.repeat(np.arange(self.n_seg), self.n_per_seg)

        # segment-specific parameters
        mu = self.rng.uniform(-2.0, 2.0, size=(self.n_seg, self.latent_dim))
        sigma = self.rng.uniform(0.5, 2.0, size=(self.n_seg, self.latent_dim))

        z = np.zeros((n, self.latent_dim))
        for s in range(self.n_seg):
            idx = slice(s * self.n_per_seg, (s + 1) * self.n_per_seg)
            z[idx] = (
                self.rng.randn(self.n_per_seg, self.latent_dim) * sigma[s]
                + mu[s]
            )

        # one-hot encode u (recommended for iVAE)
        u_oh = np.zeros((n, self.n_seg))
        u_oh[np.arange(n), u] = 1.0

        return z, u_oh

    # ------------------------------------------------------------------
    # Nonlinear mixing
    # ------------------------------------------------------------------

    def _mix_nonlinear(self, z: np.ndarray) -> np.ndarray:
        """
        Apply a random nonlinear mixing f(z).
        """
        h = z
        for k in range(self.n_layers):
            W = self._random_matrix(h.shape[1], self.data_dim)
            h = h @ W
            if k < self.n_layers - 1:
                h = self._activation(h)
        return h

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _random_matrix(self, d_in: int, d_out: int) -> np.ndarray:
        W = self.rng.randn(d_in, d_out)
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-8
        return W

    def _activation(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "relu":
            return np.maximum(0.0, x)
        elif self.activation == "none":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
