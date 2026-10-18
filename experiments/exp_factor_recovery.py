"""
Factor recovery experiments.

Evaluates identifiability of temporal concepts by comparing inferred factors
to ground-truth factors in controlled synthetic data generation processes.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import signal
import time
import random

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.exp_basic import Exp_Basic
from synthetic.dynamic import DDFMSynthetic
from synthetic.static import IVAESynthetic
from utils.metrics import mean_corr_coef, subspace_distance, temporal_smoothness, trace_r2
from utils.model_fitting import fit_model


class TimeoutError(Exception):
    """Timeout exception for model fitting."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Model fitting exceeded maximum time limit")


def _fit_model_with_timeout(
    model_name: str,
    df_train: pd.DataFrame,
    cfg: DictConfig,
    r: int = None,
    u_train: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    timeout_seconds: Optional[int] = None,
) -> Optional[Any]:
    """
    Fit a model with optional timeout protection.
    
    Parameters
    ----------
    timeout_seconds : int, optional
        Maximum time in seconds. If None, no timeout. Uses signal-based timeout
        on Unix systems (main thread only). Falls back to elapsed-time check
        if signals are not available.
    """
    if timeout_seconds is None or timeout_seconds <= 0:
        # No timeout - call directly
        return _fit_model(
            model_name=model_name,
            df_train=df_train,
            cfg=cfg,
            r=r,
            u_train=u_train,
            device=device,
        )
    
    # Try signal-based timeout (Unix only, main thread only)
    # This is the most reliable way to interrupt a hanging operation
    try:
        # Check if signal is available (Unix)
        if hasattr(signal, 'SIGALRM') and hasattr(signal, 'alarm'):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                model = _fit_model(
                    model_name=model_name,
                    df_train=df_train,
                    cfg=cfg,
                    r=r,
                    u_train=u_train,
                    device=device,
                )
                return model
            finally:
                # Cancel alarm and restore old handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Signal not available (Windows) - just call without timeout
            # User will need to manually interrupt if it hangs
            return _fit_model(
                model_name=model_name,
                df_train=df_train,
                cfg=cfg,
                r=r,
                u_train=u_train,
                device=device,
            )
    except (TimeoutError, KeyboardInterrupt):
        if hasattr(signal, 'alarm'):
            signal.alarm(0)  # Cancel alarm
        raise
    except Exception:
        if hasattr(signal, 'alarm'):
            signal.alarm(0)  # Cancel alarm on any error
        raise


def _build_model_configs_for_suite(
    exp_cfg: DictConfig,
    suite: str,
    model_names: list[str],
    r: int,
    config_root: Path,
) -> Dict[str, DictConfig]:
    """
    Build per-model config for a given suite. Merges: model yaml + suite_model_overrides[suite][model].
    Per-suite configuration is the single source; no top-level *_overrides.
    """
    resolve_ctx = OmegaConf.create({"dataset": {"r": r}})
    suite_overrides = (exp_cfg.get("suite_model_overrides") or {}).get(suite) or {}
    model_configs: Dict[str, DictConfig] = {}
    for model_name in model_names:
        model_path = config_root / "model" / f"{model_name}.yaml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")
        model_base = OmegaConf.load(model_path)
        suite_model = suite_overrides.get(model_name) if hasattr(suite_overrides, "get") else {}
        if suite_model:
            suite_dict = OmegaConf.to_container(suite_model, resolve=True) if hasattr(suite_model, "__iter__") and not isinstance(suite_model, str) else {}
            if isinstance(suite_dict, dict) and suite_dict:
                # Suite overrides must override base yaml (merge base first, then suite).
                model_base = OmegaConf.merge(model_base, OmegaConf.create(suite_dict))
        root = OmegaConf.merge(OmegaConf.create({"model": model_base}), resolve_ctx)
        OmegaConf.resolve(root)
        model_configs[model_name] = OmegaConf.create({"model": root.model})
    return model_configs


def _suggest_trial_params(model_name: str, trial: Any) -> Dict[str, Any]:
    """
    Suggest a small, model-appropriate hyperparameter subset for fair tuning.
    We intentionally keep the search space small and comparable across models.
    """
    model_name = str(model_name).lower()
    if model_name == "vae":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 100, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [2, 3]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    if model_name == "ivae":
        # Tune KL weight and batch size; keep architecture aligned with VAE for comparability.
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-3, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 100, 128]),
            "n_layers": trial.suggest_categorical("n_layers", [2, 3]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "a": trial.suggest_categorical("a", [0.5, 1.0, 2.0, 5.0, 10.0]),
        }
    if model_name == "ivdfm":
        # Tune only a small subset to keep budgets comparable.
        return {
            "beta_kl": trial.suggest_float("beta_kl", 0.3, 3.0, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "optimizer_weight_decay": trial.suggest_float("optimizer_weight_decay", 1e-5, 5e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def _apply_trial_params(cfg_for_model: DictConfig, model_name: str, params: Dict[str, Any], *, max_epochs: int) -> DictConfig:
    """Return a shallow-updated cfg_for_model with trial params applied."""
    cfg_new = OmegaConf.create(OmegaConf.to_container(cfg_for_model, resolve=True))
    if model_name in {"vae", "ivae"}:
        # VAE/iVAE configs use top-level keys in model yaml (epochs, hidden_dim, ...)
        # but we load them under cfg_new.model.*
        for k, v in params.items():
            cfg_new.model[k] = v
        # Enforce trial epochs
        if model_name == "vae":
            cfg_new.model["epochs"] = max_epochs
        else:
            cfg_new.model["epochs"] = max_epochs
    elif model_name == "ivdfm":
        for k, v in params.items():
            cfg_new.model[k] = v
        cfg_new.model["max_epochs"] = max_epochs
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")
    return cfg_new


def _run_optuna_factor_recovery(
    *,
    suite: str,
    exp_cfg: DictConfig,
    config_root: Path,
    device: Optional[torch.device],
    T: int,
    N: int,
    r: int,
    rho: float,
    alpha: float,
    poly_degree: int,
    sign_features: int,
    portion_missings: float,
    dynamic_shock_distribution: str,
    tune_models: list[str],
    n_trials: int,
    timeout: Optional[int],
    max_epochs_per_trial: int,
    base_seed: int,
    data_seed: int,
) -> Dict[str, Dict[str, Any]]:
    """Tune a small hyperparameter subset per model on a fixed synthetic dataset for one suite."""
    import optuna

    suite = str(suite).lower()
    if suite not in {"dynamic", "static"}:
        raise ValueError(f"Unsupported optuna suite: {suite}")
    print(f"[optuna] using fixed {suite} dataset seed={data_seed}", flush=True)

    if suite == "dynamic":
        gen_dyn = DDFMSynthetic(
            n_obs=T,
            n_series=N,
            factor_dim=r,
            rho=rho,
            alpha=alpha,
            poly_degree=poly_degree,
            sign_features=sign_features,
            shock_distribution=str(dynamic_shock_distribution).lower(),
            missing_rate=portion_missings,
            seed=data_seed,
        )
        dyn = gen_dyn.generate()
        x_tune = dyn["x"]
        f_true_tune = dyn["latent"]
        df_tune = pd.DataFrame(x_tune, columns=[f"series_{i}" for i in range(N)])
        df_tune.index = pd.RangeIndex(start=0, stop=T)
        n_seg_dyn = max(1, int(exp_cfg.get("n_seg", 10)))
        seg_id = np.minimum((np.arange(T) * n_seg_dyn) // max(1, T), n_seg_dyn - 1)
        u_tune = np.eye(n_seg_dyn, dtype=np.float32)[seg_id]
    else:
        n_seg = max(1, int(exp_cfg.get("n_seg", 10)))
        n_per_seg = max(1, int(T // n_seg))
        gen_static = IVAESynthetic(
            n_per_seg=n_per_seg,
            n_seg=n_seg,
            latent_dim=r,
            data_dim=int(exp_cfg.get("static_data_dim") or N),
            n_layers=int(exp_cfg.get("static_n_layers", 2)),
            activation=str(exp_cfg.get("static_activation", "tanh")),
            noise_std=float(exp_cfg.get("static_noise_std", 0.0)),
            missing_rate=portion_missings,
            seed=data_seed,
        )
        stat = gen_static.generate()
        x_tune = stat["x"]
        f_true_tune = stat["latent"]
        u_tune = stat["u"]
        df_tune = pd.DataFrame(x_tune, columns=[f"series_{i}" for i in range(x_tune.shape[1])])
        df_tune.index = pd.RangeIndex(start=0, stop=x_tune.shape[0])

    base_model_configs = _build_model_configs_for_suite(exp_cfg, suite, tune_models, r, config_root)

    tuned: Dict[str, Dict[str, Any]] = {}
    for model_name in tune_models:
        model_name = str(model_name).lower()
        print(f"[optuna] tuning {model_name}...", flush=True)

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_trial_params(model_name, trial)
            cfg_trial = _apply_trial_params(base_model_configs[model_name], model_name, params, max_epochs=max_epochs_per_trial)
            # Vary training randomness per trial but keep data fixed
            trial_seed = base_seed + int(trial.number)
            torch.manual_seed(trial_seed)
            np.random.seed(trial_seed)
            try:
                model = _fit_model_with_timeout(
                    model_name=model_name,
                    df_train=df_tune,
                    cfg=cfg_trial,
                    r=r,
                    u_train=u_tune if model_name == "ivae" else None,
                    device=device,
                    timeout_seconds=600,
                )
                f_hat = model.factors()
                f_true_eval = f_true_tune
                # Align dims
                if f_hat.shape[1] != f_true_eval.shape[1]:
                    min_dim = min(f_hat.shape[1], f_true_eval.shape[1])
                    f_hat = f_hat[:, :min_dim]
                    f_true_eval = f_true_eval[:, :min_dim]
                if f_hat.shape[0] != f_true_eval.shape[0]:
                    T_common = min(f_true_eval.shape[0], f_hat.shape[0])
                    f_true_eval = f_true_eval[-T_common:, :]
                    f_hat = f_hat[-T_common:, :]
                mcc = mean_corr_coef(torch.from_numpy(f_true_eval).float(), torch.from_numpy(f_hat).float()).item()
            except TimeoutError:
                raise optuna.TrialPruned()
            except Exception:
                raise optuna.TrialPruned()

            if not np.isfinite(mcc):
                raise optuna.TrialPruned()
            # maximize MCC => minimize negative
            return -float(mcc)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=base_seed, n_startup_trials=min(5, n_trials)),
            study_name=f"factor_recovery_{suite}_{model_name}",
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        if study.best_trial is None or study.best_trial.state != optuna.trial.TrialState.COMPLETE:
            print(f"[optuna] no completed trials for {model_name}; keeping base params", flush=True)
            tuned[model_name] = {}
        else:
            tuned[model_name] = dict(study.best_trial.params)
            tuned[model_name]["_best_value_neg_mcc"] = float(study.best_value)
            print(f"[optuna] best {model_name}: MCC={-study.best_value:.4f}, params={study.best_trial.params}", flush=True)

    # Persist for reproducibility
    try:
        import json
        out_dir = Path(exp_cfg.get("output_dir", "./results/factor_recovery"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"optuna_best_params_{suite}.json"
        with open(out_path, "w") as f:
            json.dump(tuned, f, indent=2)
        print(f"[optuna] wrote best params to {out_path}", flush=True)
    except Exception as e:
        print(f"[optuna] warning: failed to write best params json: {e}", flush=True)

    return tuned


class Exp_Factor_Recovery(Exp_Basic):
    """
    Factor recovery experiment.

    Why this is a class (vs only a function):
    - keeps a consistent interface with other experiments (device, cfg, entrypoint)
    - allows future reuse of shared hooks (logging, saving, seeding, etc.)
    - still supports multi-model evaluation per run (DFM + DDFM) cleanly
    """

    def _build_model(self):
        # Synthetic concept recovery evaluates multiple models per simulation.
        # We keep `self.model` unused (None) and build models inside `run()`.
        return None

    def train(self):
        # Not used: synthetic experiment doesn't have a training loop distinct from evaluation.
        return None

    def test(self):
        # Not used: synthetic experiment runs end-to-end in `run()`.
        return None

    def run(self) -> None:
        exp_cfg = self.cfg.experiment
        
        # Validate and extract parameters
        n_simulations = int(exp_cfg.get("n_simulations", 10))
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        
        T = int(exp_cfg.get("T", 100))
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        
        N = int(exp_cfg.get("N", 10))
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        
        r = int(exp_cfg.get("r", 2))
        if r < 1:
            raise ValueError(f"r must be >= 1, got {r}")
        if r > N:
            print(f"Warning: r={r} > N={N}, this may cause issues")
        
        rho = float(exp_cfg.get("rho", 0.7))
        if not (0.0 <= rho <= 1.0):
            print(f"Warning: rho={rho} is outside [0, 1], may cause instability")
        
        alpha = float(exp_cfg.get("alpha", 0.2))
        dynamic_shock_distribution = str(exp_cfg.get("dynamic_shock_distribution", "laplace")).lower()
        poly_degree = int(exp_cfg.get("poly_degree", 1))
        if poly_degree < 0:
            raise ValueError(f"poly_degree must be >= 0, got {poly_degree}")
        
        sign_features = int(exp_cfg.get("sign_features", 0))
        if sign_features < 0:
            raise ValueError(f"sign_features must be >= 0, got {sign_features}")
        
        portion_missings = float(exp_cfg.get("portion_missings", 0.0))
        if not (0.0 <= portion_missings <= 1.0):
            raise ValueError(f"portion_missings must be in [0, 1], got {portion_missings}")

        # Static (iVAE-style) generator params (optional; defaults keep backward compatibility)
        static_n_seg = int(exp_cfg.get("n_seg", 10))
        static_n_layers = int(exp_cfg.get("static_n_layers", 2))
        static_activation = str(exp_cfg.get("static_activation", "tanh"))
        static_noise_std = float(exp_cfg.get("static_noise_std", 0.0))
        static_data_dim = exp_cfg.get("static_data_dim", None)
        static_data_dim = int(static_data_dim) if static_data_dim is not None else None
    
        print(f"Running {n_simulations} synthetic experiments")
        print(f"  T={T}, N={N}, r={r}, rho={rho}, alpha={alpha}")
        print(f"  [dynamic] shock_distribution={dynamic_shock_distribution}")
        print(f"  poly_degree={poly_degree}, sign_features={sign_features}, missing_rate={portion_missings}")
        print(
            f"  [static] n_seg={static_n_seg}, n_layers={static_n_layers}, "
            f"activation={static_activation}, noise_std={static_noise_std}, data_dim={static_data_dim}"
        )
    
        suites = list(exp_cfg.get("suites", ["dynamic", "static"]))
        requested_models = exp_cfg.get("models", None)
        if requested_models is not None:
            requested_models = [str(m).lower() for m in list(requested_models)]

        def _models_for_suite(suite: str) -> list[str]:
            # We allow all four models on both suites; iVAE receives `u_train`:
            # - static: `u` from IVAESynthetic
            # - dynamic: piecewise-constant time segments (constructed above)
            default = ["dfm", "ivdfm", "ddfm", "vae", "ivae"]
            if requested_models is None:
                return default
            # Respect user-requested subset, but keep stable ordering (default order).
            return [m for m in default if m in set(requested_models)]

        results: dict = {}
        for suite in suites:
            results[suite] = {}
            for model_name in _models_for_suite(suite):
                results[suite][model_name] = {"mcc": [], "subspace_dist": [], "smoothness": [], "trace_r2": []}

        # Per-(suite, model) config: built per suite so suite_model_overrides (hardcoded for paper table
        # reproducibility) are applied. config_root and resolve_ctx used by _build_model_configs_for_suite.
        config_root = Path(__file__).resolve().parents[1] / "config"
        resolve_ctx = OmegaConf.create({"dataset": {"r": r}})

        # ---------------------------------------------------------------------
        # Optional fair tuning (same budget per model, dynamic suite only)
        # ---------------------------------------------------------------------
        optuna_cfg = exp_cfg.get("optuna") or {}
        run_optuna = bool(optuna_cfg.get("enabled", False))
        tuned_overrides_by_suite: Dict[str, Dict[str, Dict[str, Any]]] = {}
        if run_optuna:
            try:
                import optuna  # noqa: F401
            except Exception as e:
                raise RuntimeError(
                    "optuna is enabled in config but not available in the environment"
                ) from e

            tune_models_default = [str(m).lower() for m in list(optuna_cfg.get("models", ["vae", "ivae", "ivdfm"]))]
            n_trials = int(optuna_cfg.get("n_trials", 10))
            timeout = optuna_cfg.get("timeout")
            max_epochs_per_trial = int(optuna_cfg.get("max_epochs_per_trial", 200))
            optuna_data_seed = int(optuna_cfg.get("data_seed", 2026))
            optuna_suites = [str(s).lower() for s in list(optuna_cfg.get("suites", ["dynamic"]))]
            models_by_suite = optuna_cfg.get("models_by_suite") or {}

            print(
                f"[optuna] enabled: suites={optuna_suites}, default_models={tune_models_default}, "
                f"n_trials={n_trials}, timeout={timeout}, max_epochs_per_trial={max_epochs_per_trial}",
                flush=True,
            )
            for tune_suite in optuna_suites:
                if tune_suite not in {"dynamic", "static"}:
                    print(f"[optuna] warning: skipping unknown suite '{tune_suite}'", flush=True)
                    continue
                suite_specific_models = models_by_suite.get(tune_suite)
                tune_models = (
                    [str(m).lower() for m in list(suite_specific_models)]
                    if suite_specific_models is not None
                    else tune_models_default
                )
                print(f"[optuna] tuning suite={tune_suite}, models={tune_models}", flush=True)
                tuned_overrides_by_suite[tune_suite] = _run_optuna_factor_recovery(
                    suite=tune_suite,
                    exp_cfg=exp_cfg,
                    config_root=config_root,
                    device=self.device,
                    T=T,
                    N=N,
                    r=r,
                    rho=rho,
                    alpha=alpha,
                    poly_degree=poly_degree,
                    sign_features=sign_features,
                    portion_missings=portion_missings,
                    dynamic_shock_distribution=dynamic_shock_distribution,
                    tune_models=tune_models,
                    n_trials=n_trials,
                    timeout=timeout,
                    max_epochs_per_trial=max_epochs_per_trial,
                    base_seed=int(self.cfg.get("seed", 2026)),
                    data_seed=optuna_data_seed,
                )

        def _apply_tuned_overrides(suite: str, model_configs: Dict[str, DictConfig]) -> Dict[str, DictConfig]:
            tuned_overrides = tuned_overrides_by_suite.get(suite, {})
            if not tuned_overrides:
                return model_configs
            merged: Dict[str, DictConfig] = {}
            for m, cfg_m in model_configs.items():
                params = tuned_overrides.get(m, {})
                if not params:
                    merged[m] = cfg_m
                    continue
                params = {k: v for k, v in params.items() if not str(k).startswith("_")}
                cfg_new = OmegaConf.merge(cfg_m, OmegaConf.create({"model": params}))
                OmegaConf.resolve(cfg_new)
                merged[m] = cfg_new
            return merged

        # Store factors from the run with best MCC per model (for visualization)
        factors_for_viz: Dict[str, Dict[str, tuple]] = {}
        best_mcc_for_viz: Dict[str, Dict[str, float]] = {"dynamic": {}, "static": {}}

        # Run Monte Carlo simulations
        for sim_idx in range(n_simulations):
            # Data seed: if factor_recovery_data_seed is set, use it for all sims (same data); else seed+sim_idx
            data_seed_cfg = exp_cfg.get("factor_recovery_data_seed")
            if data_seed_cfg is not None:
                data_seed = int(data_seed_cfg)
                # Keep data fixed but still vary training randomness across simulations.
                seed = int(self.cfg.get("seed", 2026)) + sim_idx
                # NOTE: run.py seeds global RNG once at startup; we must re-seed per simulation
                # when we fix the dataset, otherwise all sims become identical (std -> 0).
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            else:
                seed = int(self.cfg.get("seed", 2026)) + sim_idx
                data_seed = seed
            print(f"\n--- Simulation {sim_idx + 1}/{n_simulations} (seed={seed}) ---")

            if "dynamic" in suites:
                # -------------------------
                # Suite 1: dynamic factors
                # -------------------------
                print("  [dynamic] generating data...")
                gen_dyn = DDFMSynthetic(
                    n_obs=T,
                    n_series=N,
                    factor_dim=r,
                    rho=rho,
                    alpha=alpha,
                    poly_degree=poly_degree,
                    sign_features=sign_features,
                    shock_distribution=dynamic_shock_distribution,
                    missing_rate=portion_missings,
                    seed=data_seed,
                )
                dyn = gen_dyn.generate()
                x_dyn = dyn["x"]  # (T, N)
                f_true_dyn = dyn["latent"]  # (T, r)
                df_dyn = pd.DataFrame(x_dyn, columns=[f"series_{i}" for i in range(N)])
                df_dyn.index = pd.RangeIndex(start=0, stop=T)

                # Optional auxiliary variables for iVAE on the dynamic suite.
                # We define a piecewise-constant segment label over time (similar to IVAESynthetic).
                model_names_dyn = _models_for_suite("dynamic")
                u_dyn: Optional[np.ndarray] = None
                if "ivae" in model_names_dyn:
                    n_seg_dyn = max(1, int(exp_cfg.get("n_seg", static_n_seg)))
                    seg_id = np.minimum(
                        (np.arange(T) * n_seg_dyn) // max(1, T),
                        n_seg_dyn - 1,
                    )
                    u_dyn = np.eye(n_seg_dyn, dtype=np.float32)[seg_id]  # (T, n_seg_dyn)

                model_configs_dyn = _build_model_configs_for_suite(
                    exp_cfg, "dynamic", model_names_dyn, r, config_root
                )
                model_configs_dyn = _apply_tuned_overrides("dynamic", model_configs_dyn)
                factors_dyn = _eval_suite(
                    suite="dynamic",
                    model_configs=model_configs_dyn,
                    df=df_dyn,
                    f_true=f_true_dyn,
                    r_true=r,
                    results=results,
                    u_train=u_dyn,
                    device=self.device,
                    model_names=model_names_dyn,
                )
                # Keep factors from the run with best MCC so far (for visualization)
                for k, v in factors_dyn.items():
                    current_mcc = results["dynamic"][k]["mcc"][-1] if results["dynamic"][k]["mcc"] else np.nan
                    if np.isfinite(current_mcc) and current_mcc > best_mcc_for_viz["dynamic"].get(k, -np.inf):
                        best_mcc_for_viz["dynamic"][k] = float(current_mcc)
                        if "dynamic" not in factors_for_viz:
                            factors_for_viz["dynamic"] = {}
                        factors_for_viz["dynamic"][k] = (np.copy(v[0]), np.copy(v[1]))

            if "static" in suites:
                # -------------------------
                # Suite 2: static (iVAE)
                # -------------------------
                print("  [static] generating data...")
                n_seg = max(1, static_n_seg)
                n_per_seg = max(1, T // n_seg)
                T_static = n_per_seg * n_seg
                if T_static != T:
                    print(f"    [static] note: using T_static={T_static} (=n_per_seg*n_seg), from requested T={T}")

                gen_static = IVAESynthetic(
                    n_per_seg=n_per_seg,
                    n_seg=n_seg,
                    latent_dim=r,
                    data_dim=static_data_dim or N,
                    n_layers=static_n_layers,
                    activation=static_activation,
                    noise_std=static_noise_std,
                    missing_rate=portion_missings,
                    seed=data_seed,
                )
                stat = gen_static.generate()
                x_stat = stat["x"]  # (T_static, data_dim)
                z_true = stat["latent"]  # (T_static, r)
                u_stat = stat["u"]  # (T_static, n_seg) - auxiliary variables for iVAE
                df_stat = pd.DataFrame(x_stat, columns=[f"series_{i}" for i in range(x_stat.shape[1])])
                df_stat.index = pd.RangeIndex(start=0, stop=x_stat.shape[0])

                model_names_static = _models_for_suite("static")
                model_configs_static = _build_model_configs_for_suite(
                    exp_cfg, "static", model_names_static, r, config_root
                )
                model_configs_static = _apply_tuned_overrides("static", model_configs_static)
                factors_stat = _eval_suite(
                    suite="static",
                    model_configs=model_configs_static,
                    df=df_stat,
                    f_true=z_true,
                    r_true=r,
                    u_train=u_stat,  # Pass auxiliary variables for iVAE
                    results=results,
                    device=self.device,
                    model_names=_models_for_suite("static"),
                )
                # Keep factors from the run with best MCC so far (for visualization)
                for k, v in factors_stat.items():
                    current_mcc = results["static"][k]["mcc"][-1] if results["static"][k]["mcc"] else np.nan
                    if np.isfinite(current_mcc) and current_mcc > best_mcc_for_viz["static"].get(k, -np.inf):
                        best_mcc_for_viz["static"][k] = float(current_mcc)
                        if "static" not in factors_for_viz:
                            factors_for_viz["static"] = {}
                        factors_for_viz["static"][k] = (np.copy(v[0]), np.copy(v[1]))
    
        # Aggregate and report results
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)

        # Base output directory
        base_output_dir = Path(exp_cfg.get("output_dir", "./results/factor_recovery"))
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results per suite/model directory
        for suite in suites:
            print(f"\n[{suite}]")
            model_names = list(results[suite].keys())
            
            for model_name in model_names:
                print(f"  {model_name.upper()}:")
                mcc_vals = np.array(results[suite][model_name]["mcc"])
                sub_vals = np.array(results[suite][model_name]["subspace_dist"])
                smooth_vals = np.array(results[suite][model_name]["smoothness"])
                trace_r2_vals = np.array(results[suite][model_name]["trace_r2"])

                valid_mask = ~np.isnan(mcc_vals)
                n_valid = int(valid_mask.sum())

                # Create suite/model-specific output directory
                model_output_dir = base_output_dir / suite / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)

                if n_valid > 0:
                    mcc_mean = float(np.mean(mcc_vals[valid_mask]))
                    mcc_std = float(np.std(mcc_vals[valid_mask]))
                    sub_mean = float(np.mean(sub_vals[valid_mask]))
                    sub_std = float(np.std(sub_vals[valid_mask]))
                    smooth_mean = float(np.mean(smooth_vals[valid_mask]))
                    smooth_std = float(np.std(smooth_vals[valid_mask]))
                    trace_r2_mean = float(np.mean(trace_r2_vals[valid_mask]))
                    trace_r2_std = float(np.std(trace_r2_vals[valid_mask]))
                    
                    print(f"    Valid runs: {n_valid}/{n_simulations}")
                    print(f"    MCC:         {mcc_mean:.4f} ± {mcc_std:.4f}")
                    print(f"    Subspace:    {sub_mean:.4f} ± {sub_std:.4f}")
                    print(f"    Smoothness:  {smooth_mean:.4f} ± {smooth_std:.4f}")
                    print(f"    Trace R²:    {trace_r2_mean:.4f} ± {trace_r2_std:.4f}")
                    
                    # Save per-model CSV
                    try:
                        model_csv = model_output_dir / "factor_recovery_summary.csv"
                        df_model = pd.DataFrame([{
                            "suite": suite,
                            "model": model_name,
                            "n_valid": n_valid,
                            "n_simulations": n_simulations,
                            "mcc_mean": mcc_mean,
                            "mcc_std": mcc_std,
                            "subspace_mean": sub_mean,
                            "subspace_std": sub_std,
                            "smoothness_mean": smooth_mean,
                            "smoothness_std": smooth_std,
                            "trace_r2_mean": trace_r2_mean,
                            "trace_r2_std": trace_r2_std,
                        }])
                        df_model.to_csv(model_csv, index=False)
                    except Exception as e:
                        print(f"      Warning: Failed to save model CSV: {e}")
                    
                    # Save per-model detailed JSON
                    try:
                        import json
                        model_json = model_output_dir / "factor_recovery_detailed.json"
                        # For DDFM, average MC samples to single value (align with other models)
                        if model_name == "ddfm" and n_valid > 1:
                            model_data = {
                                "mcc": [mcc_mean],
                                "subspace_dist": [sub_mean],
                                "smoothness": [smooth_mean],
                                "trace_r2": [trace_r2_mean],
                            }
                        else:
                            model_data = {
                                "mcc": [float(x) if not np.isnan(x) else None for x in results[suite][model_name]["mcc"]],
                                "subspace_dist": [float(x) if not np.isnan(x) else None for x in results[suite][model_name]["subspace_dist"]],
                                "smoothness": [float(x) if not np.isnan(x) else None for x in results[suite][model_name]["smoothness"]],
                                "trace_r2": [float(x) if not np.isnan(x) else None for x in results[suite][model_name]["trace_r2"]],
                            }
                        with open(model_json, "w") as f:
                            json.dump(model_data, f, indent=2)
                    except Exception as e:
                        print(f"      Warning: Failed to save model JSON: {e}")
                    
                    # Save per-model text summary
                    try:
                        model_summary = model_output_dir / "factor_recovery_summary.txt"
                        with open(model_summary, "w") as f:
                            f.write("Factor Recovery Experiment Results\n")
                            f.write("=" * 60 + "\n\n")
                            f.write(f"Configuration:\n")
                            f.write(f"  Suite: {suite}\n")
                            f.write(f"  Model: {model_name.upper()}\n")
                            f.write(f"  T={T}, N={N}, r={r}\n")
                            f.write(f"  n_simulations={n_simulations}\n")
                            f.write(f"\n")
                            f.write(f"Results:\n")
                            f.write(f"  Valid runs: {n_valid}/{n_simulations}\n")
                            f.write(f"  MCC:         {mcc_mean:.4f} ± {mcc_std:.4f}\n")
                            f.write(f"  Subspace:    {sub_mean:.4f} ± {sub_std:.4f}\n")
                            f.write(f"  Smoothness:  {smooth_mean:.4f} ± {smooth_std:.4f}\n")
                            f.write(f"  Trace R²:    {trace_r2_mean:.4f} ± {trace_r2_std:.4f}\n")
                    except Exception as e:
                        print(f"      Warning: Failed to save model text summary: {e}")

                    # Save true and learned factors (from last simulation) for plotting
                    suite_factors = factors_for_viz.get(suite)
                    if suite_factors is not None and model_name in suite_factors:
                        try:
                            f_true_save, f_hat_save = suite_factors[model_name]
                            np.save(model_output_dir / "factors_true.npy", np.asarray(f_true_save))
                            np.save(model_output_dir / "factors_hat.npy", np.asarray(f_hat_save))
                        except Exception as e:
                            print(f"      Warning: Failed to save factors for {model_name}: {e}")
                else:
                    print("    No valid runs")

        # Optionally write LaTeX table to paper tables dir
        tables_dir = exp_cfg.get("factor_recovery_tables_dir")
        if tables_dir:
            table_path = Path(tables_dir) / "factor_recovery.tex"
            if write_latex_table_to_file(base_output_dir, table_path):
                print(f"Wrote factor recovery table to {table_path}")
            else:
                print("No factor_recovery_summary.csv found; skipping LaTeX table write.")


def _eval_suite(
    *,
    suite: str,
    model_configs: Dict[str, DictConfig],
    df: pd.DataFrame,
    f_true: np.ndarray,
    r_true: int,
    results: dict,
    u_train: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    model_names: Optional[list[str]] = None,
) -> Dict[str, tuple]:
    """
    Evaluate models on a suite and return factors for visualization.
    Uses per-model config from model_configs[model_name] (loaded from config/model/*.yaml + experiment *_overrides).
    
    Returns
    -------
    Dict[str, tuple]
        Dictionary mapping model_name to (f_true, f_hat) tuple for visualization
    """
    if model_names is None:
        model_names = ["dfm", "ddfm"] if suite == "dynamic" else ["dfm", "ddfm", "vae", "ivae"]
    
    factors_dict = {}

    # Timeout limits per model (in seconds). Models that exceed this are skipped.
    # DDFM is a placeholder and may hang, so give it a shorter timeout.
    timeout_limits = {
        "dfm": 300,      # 5 minutes max for DFM EM
        "ddfm": 60,      # 1 minute max for DDFM (placeholder, may hang)
        "vae": 600,      # 10 minutes max for VAE training
        "ivae": 600,     # 10 minutes max for iVAE training
        "ivdfm": 600,    # 10 minutes max for iVDFM training
    }
    
    # Post-fit timeout: factors() or metric computation can hang (e.g. DFM with NaN state, or DDFM).
    post_fit_timeout_sec = 30

    for model_name in model_names:
        try:
            start_time = time.time()
            timeout_sec = timeout_limits.get(model_name, 300)
            print(f"    [{suite}] fitting {model_name.upper()}... (timeout: {timeout_sec}s)", flush=True)

            try:
                cfg_for_model = model_configs.get(model_name)
                if cfg_for_model is None:
                    raise ValueError(f"No config for model {model_name} (missing in model_configs)")
                model = _fit_model_with_timeout(
                    model_name=model_name,
                    df_train=df,
                    cfg=cfg_for_model,
                    r=r_true,
                        u_train=u_train if model_name == "ivae" else None,
                    device=device,
                    timeout_seconds=timeout_sec,
                )
            except TimeoutError:
                print(f"      [{model_name.upper()}] TIMEOUT after {timeout_sec}s - skipping", flush=True)
                results[suite][model_name]["mcc"].append(np.nan)
                results[suite][model_name]["subspace_dist"].append(np.nan)
                results[suite][model_name]["smoothness"].append(np.nan)
                continue

            fit_time = time.time() - start_time
            print(f"      [{model_name.upper()}] fit completed in {fit_time:.1f}s", flush=True)

            # factors() or metric computation can hang (e.g. DFM with NaN state, or slow LA). Use a short timeout.
            old_handler = None
            try:
                if hasattr(signal, "SIGALRM") and hasattr(signal, "alarm"):
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(post_fit_timeout_sec)
                try:
                    f_hat = model.factors()
                    # Align factor dimension (number of latent factors)
                    if f_hat.shape[1] != f_true.shape[1]:
                        min_dim = min(f_hat.shape[1], f_true.shape[1])
                        f_hat = f_hat[:, :min_dim]
                        f_true_eval = f_true[:, :min_dim]
                    else:
                        f_true_eval = f_true
                    # Align time dimension
                    if f_hat.shape[0] != f_true_eval.shape[0]:
                        T_common = min(f_true_eval.shape[0], f_hat.shape[0])
                        f_true_eval = f_true_eval[-T_common:, :]
                        f_hat = f_hat[-T_common:, :]
                    f_true_t = torch.from_numpy(f_true_eval).float()
                    f_hat_t = torch.from_numpy(f_hat).float()
                    mcc = mean_corr_coef(f_true_t, f_hat_t).item()
                    sub_dist = subspace_distance(f_true_t, f_hat_t).item()
                    smooth = temporal_smoothness(f_hat_t).item()
                    trace_r2_score = trace_r2(f_true_t, f_hat_t).item()
                finally:
                    if hasattr(signal, "alarm"):
                        signal.alarm(0)
                    if old_handler is not None:
                        signal.signal(signal.SIGALRM, old_handler)
            except TimeoutError:
                print(f"      [{model_name.upper()}] post-fit timeout ({post_fit_timeout_sec}s) - skipping", flush=True)
                results[suite][model_name]["mcc"].append(np.nan)
                results[suite][model_name]["subspace_dist"].append(np.nan)
                results[suite][model_name]["smoothness"].append(np.nan)
                results[suite][model_name]["trace_r2"].append(np.nan)
                continue

            # Guard against invalid metrics (NaN/inf)
            if not (np.isfinite(mcc) and np.isfinite(sub_dist) and np.isfinite(smooth) and np.isfinite(trace_r2_score)):
                print(
                    f"      Warning: invalid metrics for {model_name.upper()} "
                    f"(MCC={mcc}, Subspace={sub_dist}, Smoothness={smooth}, TraceR2={trace_r2_score}) - skipping this run",
                    flush=True,
                )
                results[suite][model_name]["mcc"].append(np.nan)
                results[suite][model_name]["subspace_dist"].append(np.nan)
                results[suite][model_name]["smoothness"].append(np.nan)
                results[suite][model_name]["trace_r2"].append(np.nan)
                continue

            results[suite][model_name]["mcc"].append(mcc)
            results[suite][model_name]["subspace_dist"].append(sub_dist)
            results[suite][model_name]["smoothness"].append(smooth)
            results[suite][model_name]["trace_r2"].append(trace_r2_score)
            print(f"      MCC: {mcc:.4f}, Subspace: {sub_dist:.4f}, Smoothness: {smooth:.4f}, TraceR2: {trace_r2_score:.4f}", flush=True)
            factors_dict[model_name] = (f_true_eval.copy(), f_hat.copy())
        except KeyboardInterrupt:
            print(f"      [{model_name.upper()}] interrupted by user")
            raise
        except Exception as e:
            elapsed = time.time() - start_time if 'start_time' in locals() else 0.0
            print(f"      Error fitting {model_name} (after {elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()
            results[suite][model_name]["mcc"].append(np.nan)
            results[suite][model_name]["subspace_dist"].append(np.nan)
            results[suite][model_name]["smoothness"].append(np.nan)
            results[suite][model_name]["trace_r2"].append(np.nan)
    
    return factors_dict


def _fit_model(
    model_name: str,
    df_train: pd.DataFrame,
    cfg: DictConfig,
    r: int = None,
    u_train: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
):
    """
    Fit a model on training data.
    
    For DDFM in factor recovery experiments, we override activation to "tanh"
    to allow factors to take both positive and negative values, matching
    the signed ground-truth factors.
    
    Parameters
    ----------
    model_name : str
        One of "dfm", "ddfm", "vae", "ivae", "ivdfm"
    df_train : pd.DataFrame
        Training data (T x N)
    cfg : DictConfig
        Configuration containing model settings
    r : int, optional
        Number of factors (for DFM block structure, iVDFM). If None, uses config default.
    u_train : np.ndarray, optional
        Auxiliary variables (T x aux_dim). Required for iVAE.
    device : torch.device, optional
        Device to run on. If None, auto-detects.
        
    Returns
    -------
    Model
        Fitted model with factors() method
    """
    # Extract encoder_layers for DDFM if available
    encoder_layers = None
    if isinstance(cfg, DictConfig):
        encoder_layers = cfg.model.get("encoder_layers", None)
    
    # For DDFM in factor recovery, override activation to "tanh" for signed factors
    activation_override = "tanh" if model_name == "ddfm" else None
    
    return fit_model(
        model_name=model_name,
        df_train=df_train,
        cfg=cfg,
        r=r,
        u_train=u_train,
        device=device,
        scaler=None,  # Will be created inside fit_model
        encoder_layers=encoder_layers,
        activation_override=activation_override,
    )


def aggregate_results_to_table(results_dir: Path) -> pd.DataFrame:
    """
    Read all factor_recovery_summary.csv under results_dir/{dynamic,static}/{model}/
    and return a DataFrame with one row per model (order: ivdfm, ddfm, ivae, vae, dfm).
    """
    model_order = ["ivdfm", "ddfm", "ivae", "vae", "dfm"]
    rows = []
    for suite in ["dynamic", "static"]:
        suite_dir = results_dir / suite
        if not suite_dir.exists():
            continue
        for model_name in model_order:
            csv_path = suite_dir / model_name / "factor_recovery_summary.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "suite": suite,
                    "model": model_name,
                    "mcc_mean": r.get("mcc_mean", np.nan),
                    "mcc_std": r.get("mcc_std", np.nan),
                    "subspace_mean": r.get("subspace_mean", np.nan),
                    "subspace_std": r.get("subspace_std", np.nan),
                    "smoothness_mean": r.get("smoothness_mean", np.nan),
                    "smoothness_std": r.get("smoothness_std", np.nan),
                    "trace_r2_mean": r.get("trace_r2_mean", np.nan),
                    "trace_r2_std": r.get("trace_r2_std", np.nan),
                })
    if not rows:
        return pd.DataFrame()
    full = pd.DataFrame(rows)
    # Pivot: one row per model, columns (metric_suite) e.g. mcc_mean_dynamic, mcc_mean_static
    wide = full.pivot(
        index="model",
        columns="suite",
        values=[
            "mcc_mean",
            "mcc_std",
            "subspace_mean",
            "subspace_std",
            "smoothness_mean",
            "smoothness_std",
            "trace_r2_mean",
            "trace_r2_std",
        ],
    )
    if wide.empty:
        return wide
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    models_found = [m for m in model_order if m in wide.index]
    return wide.reindex(models_found)


def _format_latex_table_rows(agg: pd.DataFrame) -> list[str]:
    """Build LaTeX table body rows (with \\textbf/\\underline). Returns list of lines."""
    model_order = ["ivdfm", "ddfm", "ivae", "vae", "dfm"]
    mean_keys = [
        "mcc_mean_dynamic", "mcc_mean_static",
        "subspace_mean_dynamic", "subspace_mean_static",
        "smoothness_mean_dynamic", "smoothness_mean_static",
        "trace_r2_mean_dynamic", "trace_r2_mean_static",
    ]
    std_for_mean = {
        "mcc_mean_dynamic": "mcc_std_dynamic",
        "mcc_mean_static": "mcc_std_static",
        "subspace_mean_dynamic": "subspace_std_dynamic",
        "subspace_mean_static": "subspace_std_static",
        "smoothness_mean_dynamic": "smoothness_std_dynamic",
        "smoothness_mean_static": "smoothness_std_static",
        "trace_r2_mean_dynamic": "trace_r2_std_dynamic",
        "trace_r2_mean_static": "trace_r2_std_static",
    }
    higher_better = {"mcc_mean_dynamic", "mcc_mean_static", "trace_r2_mean_dynamic", "trace_r2_mean_static"}
    best_per_col = {}
    second_per_col = {}
    for key in mean_keys:
        if key not in agg.columns:
            continue
        vals = agg[key].values
        is_higher = key in higher_better
        if is_higher:
            best_idx = int(np.nanargmax(vals))
            second_vals = np.where(np.arange(len(vals)) == best_idx, -np.inf, np.where(np.isnan(vals), -np.inf, vals))
            second_idx = int(np.nanargmax(second_vals))
        else:
            best_idx = int(np.nanargmin(np.where(np.isnan(vals), np.inf, vals)))
            second_vals = np.where(np.arange(len(vals)) == best_idx, np.inf, np.where(np.isnan(vals), np.inf, vals))
            second_idx = int(np.nanargmin(second_vals))
        best_per_col[key] = agg.index[best_idx]
        second_per_col[key] = agg.index[second_idx]
    model_display = {"ivdfm": "iVDFM", "ddfm": "DDFM", "ivae": "iVAE", "vae": "VAE", "dfm": "DFM"}
    lines = []

    def _fmt_cell(mean_val: float, std_val: float) -> str:
        if np.isnan(mean_val):
            return "---"
        if np.isnan(std_val):
            return f"{mean_val:.4f}"
        return f"{mean_val:.4f} $\\pm$ {std_val:.4f}"

    for model in model_order:
        if model not in agg.index:
            continue
        row = agg.loc[model]
        parts = [model_display.get(model, model.upper())]
        for key in mean_keys:
            mean_val = float(row.get(key, np.nan))
            std_key = std_for_mean.get(key)
            std_val = float(row.get(std_key, np.nan)) if std_key is not None else np.nan
            sval = _fmt_cell(mean_val, std_val)
            if key in best_per_col and best_per_col[key] == model:
                parts.append(f"\\textbf{{{sval}}}")
            elif key in second_per_col and second_per_col[key] == model:
                parts.append(f"\\underline{{{sval}}}")
            else:
                parts.append(sval)
        lines.append(" & ".join(parts) + " \\\\")
    return lines


def print_latex_table_from_aggregate(results_dir: Path) -> None:
    """Aggregate factor recovery results and print LaTeX table rows to stdout."""
    agg = aggregate_results_to_table(results_dir)
    if agg.empty:
        print("No factor_recovery_summary.csv found; run factor recovery first.")
        return
    lines = _format_latex_table_rows(agg)
    print("% Paste into tables/factor_recovery.tex (replace \\midrule ... \\bottomrule):")
    print("\\midrule")
    for line in lines:
        print(line)
    print("\\bottomrule")


def write_latex_table_to_file(results_dir: Path, output_path: Path) -> bool:
    """
    Aggregate factor recovery results and write the full table to output_path
    (e.g. tables/factor_recovery.tex). Returns True if written.
    """
    agg = aggregate_results_to_table(results_dir)
    if agg.empty:
        return False
    lines = _format_latex_table_rows(agg)
    header = r"""\begin{table}[H]
\centering
\caption{Synthetic factor recovery: \textbf{dynamic} (T=200, N=20, r=5) and \textbf{static} (T=200, N=20, r=5). Metrics after MCC-based matching where applicable. Higher is better for MCC and trace $R^2$; lower for subspace distance and smoothness. Values are mean $\pm$ std over simulations.}
\label{tab:synthetic_factor_recovery}
\begin{tabular}{lcccccccc}
\toprule
Model & \multicolumn{2}{c}{MCC $\uparrow$} & \multicolumn{2}{c}{Subspace $\downarrow$} & \multicolumn{2}{c}{Smoothness $\downarrow$} & \multicolumn{2}{c}{Trace $R^2$ $\uparrow$} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
 & Dynamic & Static & Dynamic & Static & Dynamic & Static & Dynamic & Static \\
\midrule
"""
    footer = r"""\bottomrule
\end{tabular}
\end{table}
"""
    body = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(header + body + footer, encoding="utf-8")
    return True

