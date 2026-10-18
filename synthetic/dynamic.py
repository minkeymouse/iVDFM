# synthetic/ddfm.py

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from .base import BaseSynthetic


class DDFMSynthetic(BaseSynthetic):
    """
    Synthetic data generator for Dynamic Factor Model (DFM / DDFM) experiments.

    DGP:
        f_t = A f_{t-1} + u_t
        x_t = g(f_t) Lambda' + eps_t

    - Latent state f_t is dynamic
    - Identification target is the factor *space*
    - Nonlinearity enters through the measurement equation
    """

    def __init__(
        self,
        n_obs: int,
        n_series: int,
        factor_dim: int = 1,
        rho: float = 0.7,
        alpha: float = 0.2,
        poly_degree: int = 1,
        sign_features: int = 0,
        shock_distribution: str = "laplace",
        noise_std: float = 0.0,
        missing_rate: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)

        self.n_obs = n_obs
        self.n_series = n_series
        self.factor_dim = factor_dim
        self.rho = rho
        self.alpha = alpha
        self.poly_degree = poly_degree
        self.sign_features = sign_features
        self.shock_distribution = str(shock_distribution).lower()
        self.noise_std = noise_std
        self.missing_rate = missing_rate

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self) -> dict:
        f = self._generate_factors()
        f_nl = self._nonlinear_features(f)
        x = self._generate_observations(f_nl)

        x = self.add_noise(x, self.noise_std)
        x = self.add_missing(x, self.missing_rate)

        return {
            "x": x,
            "latent": f,      # true dynamic factors
        }

    # ------------------------------------------------------------------
    # Latent dynamics
    # ------------------------------------------------------------------

    def _generate_factors(self) -> np.ndarray:
        """
        AR(1) latent factor dynamics.
        """
        f = np.zeros((self.n_obs, self.factor_dim))
        if self.shock_distribution in {"gaussian", "normal"}:
            shocks = self.rng.randn(self.n_obs, self.factor_dim)
        elif self.shock_distribution == "laplace":
            # Laplace(0, b) with b=1/sqrt(2) has unit variance, matching N(0,1).
            shocks = self.rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=(self.n_obs, self.factor_dim))
        else:
            raise ValueError(f"Unsupported shock_distribution={self.shock_distribution!r} (expected 'laplace' or 'gaussian').")

        A = self.rho * np.eye(self.factor_dim)

        for t in range(self.n_obs):
            if t == 0:
                f[t] = shocks[t]
            else:
                f[t] = f[t - 1] @ A + shocks[t]

        return f

    # ------------------------------------------------------------------
    # Measurement equation
    # ------------------------------------------------------------------

    def _nonlinear_features(self, f: np.ndarray) -> np.ndarray:
        """
        Polynomial + sign expansion of factors.
        """
        z = f

        if self.poly_degree > 1:
            poly = PolynomialFeatures(
                self.poly_degree, include_bias=False
            )
            z = poly.fit_transform(z)

        if self.sign_features > 0:
            z = np.hstack(
                [z, np.sign(z[:, : self.sign_features])]
            )

        return z

    def _generate_observations(self, z: np.ndarray) -> np.ndarray:
        """
        Linear loadings + AR(1) idiosyncratic noise.
        """
        # loadings
        Lambda = self.rng.randn(self.n_series, z.shape[1])

        # idiosyncratic noise
        eps = np.zeros((self.n_obs, self.n_series))
        v = self.rng.randn(self.n_obs, self.n_series)

        for t in range(self.n_obs):
            if t == 0:
                eps[t] = v[t]
            else:
                eps[t] = self.alpha * eps[t - 1] + v[t]

        x = z @ Lambda.T + eps
        return x
