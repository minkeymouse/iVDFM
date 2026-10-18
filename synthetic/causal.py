# synthetic/causal.py
import numpy as np


class CausalSynthetic:
    """
    Minimal synthetic SCM for causality / intervention experiments.

    SCM:
        ε_t  ->  f_t  ->  y_t

        f_t = A f_{t-1} + ε_t
        y_t = C f_t

    - ε_t : exogenous shocks (intervention target)
    - dynamics : deterministic
    - observation : linear (kept simple on purpose)
    """

    def __init__(
        self,
        n_obs: int,
        state_dim: int,
        obs_dim: int | None = None,
        rho: float = 0.8,
        noise_dist: str = "laplace",
        seed: int | None = None,
    ):
        self.n_obs = n_obs
        self.state_dim = state_dim
        self.obs_dim = obs_dim or state_dim
        self.rho = rho
        self.noise_dist = noise_dist

        self.rng = np.random.RandomState(seed)

        # deterministic dynamics
        self.A = rho * np.eye(state_dim)

        # observation matrix
        self.C = self.rng.randn(self.obs_dim, state_dim)

    # ------------------------------------------------------------------
    # SCM components
    # ------------------------------------------------------------------

    def sample_shocks(self) -> np.ndarray:
        """
        Sample exogenous shocks ε_t.
        """
        if self.noise_dist == "laplace":
            return self.rng.laplace(size=(self.n_obs, self.state_dim))
        elif self.noise_dist == "gaussian":
            return self.rng.randn(self.n_obs, self.state_dim)
        else:
            raise ValueError(f"Unknown noise_dist: {self.noise_dist}")

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        """
        Deterministic state transition.
        """
        f = np.zeros((self.n_obs, self.state_dim))
        for t in range(self.n_obs):
            if t == 0:
                f[t] = shocks[t]
            else:
                f[t] = self.A @ f[t - 1] + shocks[t]
        return f

    def emit(self, states: np.ndarray) -> np.ndarray:
        """
        Observation mechanism.
        """
        return states @ self.C.T

    # ------------------------------------------------------------------
    # do-intervention
    # ------------------------------------------------------------------

    def do_intervention(
        self,
        shocks: np.ndarray,
        time: int,
        dim: int,
        value: float,
    ) -> np.ndarray:
        """
        Perform do(ε_{time, dim} = value).

        This *replaces* the shock-generating mechanism at that location.
        """
        shocks = shocks.copy()
        shocks[time, dim] = value
        return shocks

    # ------------------------------------------------------------------
    # Full simulation
    # ------------------------------------------------------------------

    def generate(self, do: dict | None = None) -> dict:
        """
        Parameters
        ----------
        do : dict or None
            Example:
                do = {
                    "time": 50,
                    "dim": 1,
                    "value": 3.0,
                }
        """
        shocks = self.sample_shocks()

        if do is not None:
            shocks = self.do_intervention(
                shocks,
                time=do["time"],
                dim=do["dim"],
                value=do["value"],
            )

        states = self.propagate(shocks)
        x = self.emit(states)

        return {
            "x": x,             # observations
            "state": states,    # ground-truth latent state f_t
            "shock": shocks,    # ground-truth shocks ε_t
        }

    # ------------------------------------------------------------------
    # Ground-truth IRF
    # ------------------------------------------------------------------

    def ground_truth_irf(self, dim: int, horizon: int, do_time: int | None = None) -> np.ndarray:
        """
        Closed-form impulse response function for do(ε_dim).
        do_time is ignored (single mechanism).
        Returns:
            IRF[h] = effect at time h
        """
        irf = []
        Ak = np.eye(self.state_dim)
        e = np.zeros(self.state_dim)
        e[dim] = 1.0

        for h in range(horizon):
            irf.append(self.C @ (Ak @ e))
            Ak = self.A @ Ak

        return np.stack(irf, axis=0)


# -----------------------------------------------------------------------------
# Regime-Switching SCM (same graph ε→f→y; mechanism varies by regime)
# -----------------------------------------------------------------------------


class CausalRegimeSynthetic:
    """
    Regime-switching SCM: same causal graph ε_t → f_t → y_t, but
    f_t = A^{(r_t)} f_{t-1} + ε_t with regime-specific diagonal A.

    Regimes: piecewise constant (fixed segments). r_t = which segment t falls into.
    Ground-truth IRF at do_time uses the active regime at do_time.
    """

    def __init__(
        self,
        n_obs: int,
        state_dim: int,
        obs_dim: int | None = None,
        num_regimes: int = 2,
        regime_rhos: list | None = None,
        noise_dist: str = "laplace",
        seed: int | None = None,
    ):
        self.n_obs = n_obs
        self.state_dim = state_dim
        self.obs_dim = obs_dim or state_dim
        self.num_regimes = num_regimes
        self.noise_dist = noise_dist
        self.rng = np.random.RandomState(seed)

        if regime_rhos is not None and len(regime_rhos) != num_regimes:
            raise ValueError("len(regime_rhos) must equal num_regimes")
        rhos = regime_rhos or [0.5 + 0.4 * (k / max(1, num_regimes - 1)) for k in range(num_regimes)]
        self.A_list = [rho * np.eye(state_dim) for rho in rhos]

        self.C = self.rng.randn(self.obs_dim, state_dim)

        # Fixed segments: regime[t] = segment index for t
        segment_length = max(1, n_obs // num_regimes)
        self.regime = np.zeros(n_obs, dtype=int)
        for t in range(n_obs):
            self.regime[t] = min(t // segment_length, num_regimes - 1)

    def sample_shocks(self) -> np.ndarray:
        if self.noise_dist == "laplace":
            return self.rng.laplace(size=(self.n_obs, self.state_dim))
        elif self.noise_dist == "gaussian":
            return self.rng.randn(self.n_obs, self.state_dim)
        raise ValueError(f"Unknown noise_dist: {self.noise_dist}")

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n_obs, self.state_dim))
        for t in range(self.n_obs):
            if t == 0:
                f[t] = shocks[t]
            else:
                k = self.regime[t]
                f[t] = self.A_list[k] @ f[t - 1] + shocks[t]
        return f

    def emit(self, states: np.ndarray) -> np.ndarray:
        return states @ self.C.T

    def do_intervention(self, shocks: np.ndarray, time: int, dim: int, value: float) -> np.ndarray:
        out = shocks.copy()
        out[time, dim] = value
        return out

    def generate(self, do: dict | None = None) -> dict:
        shocks = self.sample_shocks()
        if do is not None:
            shocks = self.do_intervention(
                shocks, time=do["time"], dim=do["dim"], value=do["value"]
            )
        states = self.propagate(shocks)
        x = self.emit(states)
        return {"x": x, "state": states, "shock": shocks}

    def ground_truth_irf(self, dim: int, horizon: int, do_time: int | None = None) -> np.ndarray:
        """Regime-conditional IRF: use active regime at do_time."""
        if do_time is None:
            do_time = self.n_obs // 2
        k = self.regime[min(do_time, self.n_obs - 1)]
        A = self.A_list[k]
        irf = []
        Ak = np.eye(self.state_dim)
        e = np.zeros(self.state_dim)
        e[dim] = 1.0
        for h in range(horizon):
            irf.append(self.C @ (Ak @ e))
            Ak = A @ Ak
        return np.stack(irf, axis=0)


# -----------------------------------------------------------------------------
# Cross-Factor (Chain) SCM: non-diagonal A, mediated latent causality
# -----------------------------------------------------------------------------


class CausalChainSynthetic:
    """
    Cross-factor causal SCM: ε_t → f_t → y_t with f_t = A f_{t-1} + ε_t.
    A is upper triangular (e.g. f1→f2→f3). Innovations ε remain independent.
    """

    def __init__(
        self,
        n_obs: int,
        state_dim: int,
        obs_dim: int | None = None,
        rho: float = 0.7,
        chain_strength: float = 0.2,
        noise_dist: str = "laplace",
        seed: int | None = None,
    ):
        self.n_obs = n_obs
        self.state_dim = state_dim
        self.obs_dim = obs_dim or state_dim
        self.rho = rho
        self.chain_strength = chain_strength
        self.noise_dist = noise_dist
        self.rng = np.random.RandomState(seed)

        # Upper triangular A: diagonal rho, subdiagonal chain_strength (factor i -> i+1)
        self.A = np.diag([rho] * state_dim)
        for i in range(state_dim - 1):
            self.A[i + 1, i] = chain_strength
        # Ensure spectral radius < 1
        if np.max(np.abs(np.linalg.eigvals(self.A))) >= 1.0:
            self.A = self.A * 0.9 / np.max(np.abs(np.linalg.eigvals(self.A)))

        self.C = self.rng.randn(self.obs_dim, state_dim)

    def sample_shocks(self) -> np.ndarray:
        if self.noise_dist == "laplace":
            return self.rng.laplace(size=(self.n_obs, self.state_dim))
        elif self.noise_dist == "gaussian":
            return self.rng.randn(self.n_obs, self.state_dim)
        raise ValueError(f"Unknown noise_dist: {self.noise_dist}")

    def propagate(self, shocks: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n_obs, self.state_dim))
        for t in range(self.n_obs):
            if t == 0:
                f[t] = shocks[t]
            else:
                f[t] = self.A @ f[t - 1] + shocks[t]
        return f

    def emit(self, states: np.ndarray) -> np.ndarray:
        return states @ self.C.T

    def do_intervention(self, shocks: np.ndarray, time: int, dim: int, value: float) -> np.ndarray:
        out = shocks.copy()
        out[time, dim] = value
        return out

    def generate(self, do: dict | None = None) -> dict:
        shocks = self.sample_shocks()
        if do is not None:
            shocks = self.do_intervention(
                shocks, time=do["time"], dim=do["dim"], value=do["value"]
            )
        states = self.propagate(shocks)
        x = self.emit(states)
        return {"x": x, "state": states, "shock": shocks}

    def ground_truth_irf(self, dim: int, horizon: int, do_time: int | None = None) -> np.ndarray:
        """IRF with mediated effects (non-diagonal A). do_time unused."""
        irf = []
        Ak = np.eye(self.state_dim)
        e = np.zeros(self.state_dim)
        e[dim] = 1.0
        for h in range(horizon):
            irf.append(self.C @ (Ak @ e))
            Ak = self.A @ Ak
        return np.stack(irf, axis=0)
