import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from experiments.exp_basic import Exp_Basic
from synthetic.causal import CausalSynthetic, CausalRegimeSynthetic, CausalChainSynthetic
from utils.model_fitting import fit_model


class Exp_Causal_Intervention(Exp_Basic):
    """
    Causality / intervention experiment.

    Evaluates whether learned representations support *correct do-interventions*
    by comparing model-implied IRFs against ground-truth IRFs from a known SCM.
    """

    # --------------------------------------------------
    # Boilerplate (required by Exp_Basic)
    # --------------------------------------------------

    def _build_model(self):
        # Causal intervention is only defined for iVDFM (intervenable innovations).
        # No single persistent model is stored here.
        return None

    def train(self):
        # No separate training loop.
        return None

    def test(self):
        # Not used; everything runs in `run()`.
        return None

    # --------------------------------------------------
    # Main entrypoint
    # --------------------------------------------------

    def run(self) -> None:
        """
        Data: synthetic only, from CausalSynthetic (SCM ε_t -> f_t -> y_t).
        No real dataset is loaded; dataset config is ignored.
        """
        exp_cfg = self.cfg.experiment

        # -------------------------
        # Configuration (all from experiment yaml; no dataset)
        # -------------------------
        n_simulations = int(exp_cfg.get("n_simulations", 5))
        T_default = int(exp_cfg.get("T", 200))
        r = int(exp_cfg.get("r", 3))
        r_true = int(exp_cfg.get("r_true", r))
        T_list_cfg = exp_cfg.get("T_list")
        r_fit_list_cfg = exp_cfg.get("r_fit_list")
        T_list = [int(x) for x in T_list_cfg] if T_list_cfg else [T_default]
        r_fit_list = [int(x) for x in r_fit_list_cfg] if r_fit_list_cfg else [r_true]
        stress_mode = len(T_list) > 1 or len(r_fit_list) > 1

        rho = float(exp_cfg.get("rho", 0.8))
        horizon = int(exp_cfg.get("irf_horizon", 20))
        do_value = float(exp_cfg.get("do_value", 3.0))
        scm_type = str(exp_cfg.get("scm_type", "base")).lower()
        scm_types: List[str]
        if scm_type in {"all", "compare", "comparison"}:
            scm_types = ["base", "regime", "chain"]
        else:
            scm_types = [scm_type]
        num_regimes = int(exp_cfg.get("num_regimes", 2))
        chain_strength = float(exp_cfg.get("chain_strength", 0.2))

        models = list(exp_cfg.get("models", ["ivdfm"]))

        save_synthetic_data = bool(exp_cfg.get("save_synthetic_data", False))
        optuna_cfg = exp_cfg.get("optuna") or {}
        n_trials_opt = int(optuna_cfg.get("n_trials", 0) or 0)
        run_optuna = "ivdfm" in models and n_trials_opt > 0 and not stress_mode
        # Only used when Optuna is enabled (n_trials > 0) and stress_mode is off.
        # We reset this per SCM type so "best params" files don't mix settings.
        best_params_per_sim: List[Dict[str, Any]] = []

        print("=" * 60)
        print("Causal / Intervention Experiment")
        print("=" * 60)
        print(f"scm_types={scm_types}, T_list={T_list}, r_true={r_true}, r_fit_list={r_fit_list}, stress_mode={stress_mode}")
        print(f"models={models}, n_simulations={n_simulations}, save_synthetic_data={save_synthetic_data}, optuna={run_optuna}")
        print("=" * 60)

        # Run each SCM type separately (keeps outputs organized and matches paper tables).
        for scm_type_i in scm_types:
            best_params_per_sim = []
            output_dir = Path(exp_cfg.get("output_dir", "./results/causal")) / scm_type_i
            output_dir.mkdir(parents=True, exist_ok=True)

            stress_rows: List[Dict[str, Any]] = []
            results: Dict[str, Dict[str, list]] = {
                m: {"irf_error": [], "sign_accuracy": [], "irf_mae": [], "irf_corr": []}
                for m in models
            }

            print("\n" + "-" * 60)
            print(f"SCM: {scm_type_i}")
            print("-" * 60)

            for T in T_list:
                do_time = max(0, min(int(exp_cfg.get("do_time", T // 2)), T - 1))
                for sim in range(n_simulations):
                    seed = int(self.cfg.get("seed", 2026)) + sim
                    print(f"\n--- scm={scm_type_i} T={T} Simulation {sim + 1}/{n_simulations} (seed={seed}) ---")

                    scm = self._build_scm(scm_type_i, T, r_true, rho, num_regimes, chain_strength, seed)

                    base = scm.generate()
                    do_spec = {"time": do_time, "dim": 0, "value": do_value}
                    _ = scm.generate(do=do_spec)

                    x_base = base["x"]

                    if save_synthetic_data and not stress_mode:
                        self._save_synthetic_data(output_dir, sim, base, scm)

                    irf_true = (
                        do_value
                        * scm.ground_truth_irf(
                            dim=do_spec["dim"],
                            horizon=horizon,
                            do_time=do_time,
                        )
                    )

                    df_base = pd.DataFrame(
                        x_base,
                        columns=[f"series_{i}" for i in range(x_base.shape[1])],
                    )

                    for r_fit in r_fit_list:
                        for model_name in models:
                            print(f"  [T={T} r_fit={r_fit} {model_name.upper()}] fitting...")
                            cfg_fit = self.cfg
                            best_params_sim: Optional[Dict[str, Any]] = None

                            if model_name == "ivdfm" and run_optuna and r_fit == r_true and T == T_default:
                                best_params_sim, cfg_fit = self._run_optuna_causal(
                                    df_base=df_base,
                                    scm=scm,
                                    do_spec=do_spec,
                                    horizon=horizon,
                                    irf_true=irf_true,
                                    r=r_fit,
                                    exp_cfg=exp_cfg,
                                    optuna_cfg=optuna_cfg,
                                )
                                best_params_per_sim.append({"sim": sim, "best_params": best_params_sim})

                            try:
                                start = time.time()
                                # For causal intervention we need factors/innovations for the full timeline.
                                cfg_for_fit = cfg_fit
                                if isinstance(cfg_fit, DictConfig):
                                    cfg_for_fit = OmegaConf.create(OmegaConf.to_container(cfg_fit, resolve=True))
                                    if model_name == "ivdfm":
                                        try:
                                            cfg_for_fit.model.window = int(T)
                                            cfg_for_fit.model.window_size = int(T)
                                            cfg_for_fit.model.stride = 1
                                        except Exception:
                                            pass

                                model = fit_model(
                                    model_name=model_name,
                                    df_train=df_base,
                                    cfg=cfg_for_fit,
                                    r=r_fit,
                                    device=self.device,
                                )
                                fit_time = time.time() - start
                                print(f"    fit completed in {fit_time:.1f}s")

                                irf_hat = self._compute_model_irf(
                                    model=model,
                                    scm=scm,
                                    do_spec=do_spec,
                                    horizon=horizon,
                                )

                                err = float(np.mean((irf_hat - irf_true) ** 2))
                                sign_acc = float(np.mean(np.sign(irf_hat) == np.sign(irf_true)))
                                irf_mae = float(np.mean(np.abs(irf_hat - irf_true)))
                                a, b = irf_hat.ravel(), irf_true.ravel()
                                irf_corr = float(np.corrcoef(a, b)[0, 1]) if (np.std(a) > 1e-12 and np.std(b) > 1e-12) else np.nan

                                if stress_mode:
                                    stress_rows.append({
                                        "T": T,
                                        "r_fit": r_fit,
                                        "model": model_name,
                                        "scm_type": scm_type_i,
                                        "irf_error": err,
                                        "sign_accuracy": sign_acc,
                                        "irf_mae": irf_mae,
                                        "irf_corr": irf_corr,
                                    })
                                else:
                                    results[model_name]["irf_error"].append(err)
                                    results[model_name]["sign_accuracy"].append(sign_acc)
                                    results[model_name]["irf_mae"].append(irf_mae)
                                    results[model_name]["irf_corr"].append(irf_corr)

                                print(f"    IRF-MSE={err:.4f}, SignAcc={sign_acc:.3f}, IRF-MAE={irf_mae:.4f}, IRF-Corr={irf_corr:.3f}")

                            except Exception as e:
                                print(f"    ERROR for {model_name}: {e}")
                                if stress_mode:
                                    stress_rows.append({
                                        "T": T,
                                        "r_fit": r_fit,
                                        "model": model_name,
                                        "scm_type": scm_type_i,
                                        "irf_error": np.nan,
                                        "sign_accuracy": np.nan,
                                        "irf_mae": np.nan,
                                        "irf_corr": np.nan,
                                    })
                                else:
                                    results[model_name]["irf_error"].append(np.nan)
                                    results[model_name]["sign_accuracy"].append(np.nan)
                                    results[model_name]["irf_mae"].append(np.nan)
                                    results[model_name]["irf_corr"].append(np.nan)

            if stress_mode and stress_rows:
                self._save_stress_results(stress_rows, output_dir, exp_cfg)
            else:
                self._save_results(results, output_dir, scm_type=scm_type_i)
            if run_optuna and best_params_per_sim:
                self._save_best_params(output_dir, best_params_per_sim, results, exp_cfg, scm_type=scm_type_i)

    # --------------------------------------------------
    # SCM factory
    # --------------------------------------------------

    def _build_scm(
        self,
        scm_type: str,
        T: int,
        r: int,
        rho: float,
        num_regimes: int,
        chain_strength: float,
        seed: int,
    ):
        """Build SCM instance: base (linear), regime (piecewise A), or chain (upper-tri A)."""
        if scm_type == "regime":
            return CausalRegimeSynthetic(
                n_obs=T,
                state_dim=r,
                num_regimes=num_regimes,
                seed=seed,
            )
        if scm_type == "chain":
            return CausalChainSynthetic(
                n_obs=T,
                state_dim=r,
                rho=rho,
                chain_strength=chain_strength,
                seed=seed,
            )
        return CausalSynthetic(
            n_obs=T,
            state_dim=r,
            rho=rho,
            seed=seed,
        )

    # --------------------------------------------------
    # Synthetic data save
    # --------------------------------------------------

    def _save_synthetic_data(
        self,
        output_dir: Path,
        sim: int,
        base: Dict[str, np.ndarray],
        scm: Any,
    ) -> None:
        """Save SCM-generated data as CSV (observations, state, shock)."""
        prefix = f"sim_{sim:03d}"
        obs_cols = [f"series_{i}" for i in range(base["x"].shape[1])]
        pd.DataFrame(base["x"], columns=obs_cols).to_csv(
            output_dir / f"{prefix}_observations.csv", index=False
        )
        state_cols = [f"state_{i}" for i in range(base["state"].shape[1])]
        pd.DataFrame(base["state"], columns=state_cols).to_csv(
            output_dir / f"{prefix}_state.csv", index=False
        )
        shock_cols = [f"shock_{i}" for i in range(base["shock"].shape[1])]
        pd.DataFrame(base["shock"], columns=shock_cols).to_csv(
            output_dir / f"{prefix}_shock.csv", index=False
        )
        print(f"    Saved synthetic data: {prefix}_observations.csv, {prefix}_state.csv, {prefix}_shock.csv")

    # --------------------------------------------------
    # Optuna causal objective
    # --------------------------------------------------

    def _suggest_causal_ivdfm(self, trial: Any) -> Dict[str, Any]:
        """Suggest only non-structural iVDFM hyperparameters (optimization + capacity). Structural params fixed from config."""
        from dfm_python.models.ivdfm.optuna import suggest_ivdfm_hyperparameters_causal
        return suggest_ivdfm_hyperparameters_causal(trial, fixed=None)

    def _run_optuna_causal(
        self,
        df_base: pd.DataFrame,
        scm: CausalSynthetic,
        do_spec: Dict[str, Any],
        horizon: int,
        irf_true: np.ndarray,
        r: int,
        exp_cfg: DictConfig,
        optuna_cfg: Dict[str, Any],
    ) -> tuple:
        """Run Optuna study minimizing IRF MSE; return (best_params, cfg_with_best_model)."""
        import optuna

        n_trials = int(optuna_cfg.get("n_trials", 10))
        timeout = optuna_cfg.get("timeout")
        if timeout is not None:
            timeout = float(timeout)
        max_epochs_per_trial = optuna_cfg.get("max_epochs_per_trial")
        max_epochs_per_trial = int(max_epochs_per_trial) if max_epochs_per_trial not in (None, "null") else None

        base_model = OmegaConf.to_container(self.cfg.model, resolve=True) or {}

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_causal_ivdfm(trial)
            if max_epochs_per_trial is not None:
                params["max_epochs"] = max_epochs_per_trial
            model_config = {**base_model, **params}
            cfg_trial = OmegaConf.create({
                **OmegaConf.to_container(self.cfg, resolve=True),
                "model": model_config,
            })
            try:
                model = fit_model(
                    model_name="ivdfm",
                    df_train=df_base,
                    cfg=cfg_trial,
                    r=r,
                    device=self.device,
                )
                irf_hat = self._compute_model_irf(model=model, scm=scm, do_spec=do_spec, horizon=horizon)
                irf_hat = np.asarray(irf_hat, dtype=np.float64)
                if irf_hat.shape[1] != irf_true.shape[1]:
                    irf_hat = irf_hat[:, : irf_true.shape[1]]
                mse = float(np.mean((irf_hat - irf_true) ** 2))
                if not np.isfinite(mse):
                    raise optuna.TrialPruned()
                return mse
            except Exception as e:
                raise optuna.TrialPruned() from e

        study = optuna.create_study(direction="minimize", study_name="causal_ivdfm")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best_params = dict(study.best_params) if study.best_trials else {}
        if max_epochs_per_trial is not None:
            best_params["max_epochs"] = max_epochs_per_trial
        model_with_best = {**base_model, **best_params}
        cfg_fit = OmegaConf.create({
            **OmegaConf.to_container(self.cfg, resolve=True),
            "model": model_with_best,
        })
        print(f"    Optuna: best IRF-MSE={study.best_value:.4f} (n_trials={n_trials})")
        return best_params, cfg_fit

    # --------------------------------------------------
    # Best params save
    # --------------------------------------------------

    def _save_best_params(
        self,
        output_dir: Path,
        best_params_per_sim: List[Dict[str, Any]],
        results: Dict[str, Dict[str, list]],
        exp_cfg: DictConfig,
        scm_type: str = "base",
    ) -> None:
        """Save best Optuna params per sim and overall to output_dir (JSON)."""
        save_best = (exp_cfg.get("optuna") or {}).get("save_best_params", True)
        if not save_best or not best_params_per_sim:
            return
        for item in best_params_per_sim:
            sim = item["sim"]
            params = item["best_params"]
            path = output_dir / f"best_params_sim_{sim:03d}.json"
            with open(path, "w") as f:
                json.dump({"sim": sim, "best_params": params}, f, indent=2)
        irf_errors = np.array(results.get("ivdfm", {}).get("irf_error", []))
        if len(irf_errors) and np.any(np.isfinite(irf_errors)):
            best_sim_idx = int(np.nanargmin(irf_errors))
            overall = best_params_per_sim[best_sim_idx]["best_params"]
            with open(output_dir / "best_params.json", "w") as f:
                json.dump({
                    "best_sim": best_sim_idx,
                    "scm_type": scm_type,
                    "best_params": overall,
                    "irf_error_mean": float(np.nanmean(irf_errors)),
                }, f, indent=2)
            print(f"\nSaved best params: best_params.json (scm_type={scm_type}, from sim {best_sim_idx}), best_params_sim_*.json")

    # --------------------------------------------------
    # IRF evaluation utilities
    # --------------------------------------------------

    def _compute_model_irf(
        self,
        model: Any,
        scm: CausalSynthetic,
        do_spec: Dict[str, Any],
        horizon: int,
    ) -> np.ndarray:
        """
        Compute model-implied IRF: apply do(η_{t0,k} = c) at representation level
        and return observation-level difference vs baseline.
        Uses model.predict_irf() when available (e.g. iVDFM); otherwise returns NaN-filled array.
        """
        do_time = do_spec["time"]
        do_dim = do_spec["dim"]
        do_value = do_spec["value"]
        obs_dim = scm.obs_dim

        if hasattr(model, "predict_irf"):
            try:
                irf_hat = model.predict_irf(
                    do_time=do_time,
                    do_dim=do_dim,
                    do_value=do_value,
                    horizon=horizon,
                )
                irf_hat = np.asarray(irf_hat, dtype=np.float64)
                if np.any(~np.isfinite(irf_hat)):
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"predict_irf returned non-finite values: min={np.nanmin(irf_hat)}, max={np.nanmax(irf_hat)}")
                if irf_hat.shape[1] != obs_dim:
                    irf_hat = irf_hat[:, :obs_dim]
                return irf_hat
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
        return np.full((horizon, obs_dim), np.nan)

    # --------------------------------------------------
    # Stress results (varying T, misspecified r)
    # --------------------------------------------------

    def _save_stress_results(
        self,
        stress_rows: List[Dict[str, Any]],
        output_dir: Path,
        exp_cfg: DictConfig,
    ) -> None:
        """Aggregate stress_rows by (T, r_fit, model, scm_type); save CSV and optionally full table LaTeX."""
        df = pd.DataFrame(stress_rows)
        agg = df.groupby(["T", "r_fit", "model", "scm_type"], dropna=False).agg(
            irf_error_mean=("irf_error", "mean"),
            irf_error_std=("irf_error", "std"),
            sign_accuracy_mean=("sign_accuracy", "mean"),
            sign_accuracy_std=("sign_accuracy", "std"),
            irf_mae_mean=("irf_mae", "mean"),
            irf_mae_std=("irf_mae", "std"),
            irf_corr_mean=("irf_corr", "mean"),
            irf_corr_std=("irf_corr", "std"),
        ).reset_index()
        csv_path = output_dir / "causal_stress_summary.csv"
        agg.to_csv(csv_path, index=False)
        print("\n" + "=" * 60)
        print("STRESS RESULTS (varying T / misspecified r)")
        print("=" * 60)
        print(agg.to_string(index=False))
        print(f"\nSaved: {csv_path}")

        full_table_path = exp_cfg.get("full_table_path")
        if full_table_path:
            self._write_full_table(agg, Path(full_table_path), exp_cfg)

    def _write_full_table(self, df_stress: pd.DataFrame, path: Path, exp_cfg: DictConfig) -> None:
        """Write causal_intervention_full.tex: SCM comparison + varying T and misspecified r for base, regime, chain (read all stress CSVs)."""
        causal_root = Path(exp_cfg.get("output_dir", "./results/causal"))
        stress_dfs = []
        for name in ["base", "regime", "chain"]:
            csv_path = causal_root / name / "causal_stress_summary.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "scm_type" not in df.columns:
                        df["scm_type"] = name
                    stress_dfs.append(df)
                except Exception:
                    pass
        df_all = pd.concat(stress_dfs, ignore_index=True) if stress_dfs else df_stress

        models_present = [m for m in ["ivdfm"] if m in df_all["model"].values]
        has_extra = "irf_mae_mean" in df_all.columns and "irf_corr_mean" in df_all.columns
        ncol = 5
        tabular_spec = "lcccc"
        header = "    Setting & IRF-MSE $\\downarrow$ & IRF-MAE $\\downarrow$ & Sign Acc $\\uparrow$ & IRF Corr $\\uparrow$ \\\\"
        lines = [
            r"\begin{table}[p]",
            "    \\centering",
            r"    \caption{iVDFM causal intervention stress tests: varying $T$ and misspecified $r$ (base, regime, chain SCMs). IRF-MSE $\downarrow$, IRF-MAE $\downarrow$: mean squared / absolute error; Sign Acc $\uparrow$: fraction of (time step, variable) pairs with correct sign; IRF Corr $\uparrow$: Pearson correlation with ground truth. Mean $\pm$ std over five simulations.}",
            r"    \label{tab:causal_intervention_full}",
            f"    \\begin{{tabular}}{{{tabular_spec}}}",
            "    \\toprule",
            header,
            "    \\midrule",
        ]

        def _fmt(m, s):
            return f"${float(m):.2f} \\pm {float(s):.2f}$" if np.isfinite(m) else "---"

        # SCM comparison block: prefer causal_summary.csv; fallback to stress (T=200, r_fit=3)
        scm_rows = []
        T_ref_scm, r_ref_scm = 200, 3
        for name, label in [("base", "Base (linear)"), ("regime", "Regime"), ("chain", "Chain")]:
            row = None
            csv_path = causal_root / name / "causal_summary.csv"
            if csv_path.exists():
                try:
                    df_scm = pd.read_csv(csv_path)
                    rw = df_scm[df_scm["model"] == "ivdfm"]
                    if len(rw) and has_extra and "irf_mae_mean" in rw.columns:
                        row = rw.iloc[0]
                except Exception:
                    pass
            if row is None and len(df_all) and "scm_type" in df_all.columns:
                rw = df_all[(df_all["scm_type"] == name) & (df_all["T"] == T_ref_scm) & (df_all["r_fit"] == r_ref_scm)]
                if len(rw):
                    row = rw.iloc[0]
            if row is not None and has_extra:
                cells = f"{_fmt(row['irf_error_mean'], row['irf_error_std'])} & {_fmt(row['irf_mae_mean'], row['irf_mae_std'])} & {_fmt(row['sign_accuracy_mean'], row['sign_accuracy_std'])} & {_fmt(row['irf_corr_mean'], row['irf_corr_std'])}"
                scm_rows.append(f"    {label} & {cells} \\\\")
        if scm_rows:
            lines.append(f"    \\multicolumn{{{ncol}}}{{l}}{{\\textit{{SCM comparison ($T=200$, $r=3$)}}}} \\\\")
            lines.extend(scm_rows)
            lines.append("    \\addlinespace")

        scm_labels = [("base", "Base (linear)"), ("regime", "Regime"), ("chain", "Chain")]
        scm_short = {"base": "base", "regime": "Regime", "chain": "Chain"}
        T_vals = sorted(df_all["T"].unique()) if len(df_all) else []
        r_vals = sorted(df_all["r_fit"].unique()) if len(df_all) else []
        r_ref = 3 if 3 in r_vals else (r_vals[0] if r_vals else 3)
        T_ref = 200 if 200 in T_vals else (T_vals[0] if T_vals else 200)

        for scm_type, label in scm_labels:
            df_s = df_all[df_all["scm_type"] == scm_type] if "scm_type" in df_all.columns and len(df_all) else pd.DataFrame()
            if len(df_s) == 0:
                continue
            sub_label = scm_short.get(scm_type, label)
            if len(T_vals) > 1:
                lines.append(f"    \\multicolumn{{{ncol}}}{{l}}{{\\textit{{Varying $T$ ({sub_label} SCM, $r=3$)}}}} \\\\")
                for T in T_vals:
                    row = df_s[(df_s["T"] == T) & (df_s["r_fit"] == r_ref)]
                    if len(row) == 0:
                        cells = " & ".join(["---"] * 4)
                    else:
                        r = row.iloc[0]
                        cells = f"{_fmt(r['irf_error_mean'], r['irf_error_std'])} & {_fmt(r['irf_mae_mean'], r['irf_mae_std'])} & {_fmt(r['sign_accuracy_mean'], r['sign_accuracy_std'])} & {_fmt(r['irf_corr_mean'], r['irf_corr_std'])}"
                    lines.append(f"    $T={int(T)}$ & {cells} \\\\")
                lines.append("    \\addlinespace")
            if len(r_vals) > 1:
                lines.append(f"    \\multicolumn{{{ncol}}}{{l}}{{\\textit{{Misspecified $r$ ({sub_label} SCM, $T=200$, true $r=3$)}}}} \\\\")
                for r_fit in r_vals:
                    row = df_s[(df_s["T"] == T_ref) & (df_s["r_fit"] == r_fit)]
                    if len(row) == 0:
                        cells = " & ".join(["---"] * 4)
                    else:
                        r = row.iloc[0]
                        cells = f"{_fmt(r['irf_error_mean'], r['irf_error_std'])} & {_fmt(r['irf_mae_mean'], r['irf_mae_std'])} & {_fmt(r['sign_accuracy_mean'], r['sign_accuracy_std'])} & {_fmt(r['irf_corr_mean'], r['irf_corr_std'])}"
                    lines.append(f"    $r={int(r_fit)}$ (fit) & {cells} \\\\")
                lines.append("    \\addlinespace")

        lines.extend([
            "    \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote full table: {path}")

        # Short table (causal_intervention.tex): SCM comparison only (T=200, r=3)
        short_path = path.parent / "causal_intervention.tex"
        scm_short_rows = []
        for name, label in [("base", "Base (linear)"), ("regime", "Regime"), ("chain", "Chain")]:
            row = None
            if len(df_all) and "scm_type" in df_all.columns:
                rw = df_all[(df_all["scm_type"] == name) & (df_all["T"] == T_ref) & (df_all["r_fit"] == r_ref)]
                if len(rw):
                    row = rw.iloc[0]
            if row is not None:
                scm_short_rows.append(f"iVDFM & {label} & {_fmt(row['irf_error_mean'], row['irf_error_std'])} & {_fmt(row['sign_accuracy_mean'], row['sign_accuracy_std'])} \\\\")
        if scm_short_rows:
            short_lines = [
                r"\begin{table}[t]",
                "    \\centering",
                r"    \caption{Causal intervention accuracy on synthetic SCMs. IRF-MSE $\downarrow$: mean squared error between ground-truth and model-implied impulse response; Sign Accuracy $\uparrow$: fraction of (time step, variable) pairs with correct sign. Mean $\pm$ std over five simulations.}",
                r"    \label{tab:causal_intervention}",
                r"    \begin{tabular}{llcc}",
                "    \\toprule",
                "    Model & SCM & IRF-MSE $\\downarrow$ & Sign Accuracy $\\uparrow$ \\\\",
                "    \\midrule",
            ] + [f"    {r}" for r in scm_short_rows] + [
                "    \\bottomrule",
                "    \\end{tabular}",
                "\\end{table}",
            ]
            short_path.write_text("\n".join(short_lines), encoding="utf-8")
            print(f"Wrote short table: {short_path}")

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    def _save_results(self, results: dict, output_dir: Path, scm_type: str = "base") -> None:
        rows = []
        for model, vals in results.items():
            irf_err = np.array(vals["irf_error"], dtype=np.float64)
            sign_acc = np.array(vals["sign_accuracy"], dtype=np.float64)
            irf_mae_arr = np.array(vals.get("irf_mae", [np.nan] * len(irf_err)), dtype=np.float64)
            irf_corr_arr = np.array(vals.get("irf_corr", [np.nan] * len(irf_err)), dtype=np.float64)
            n = len(irf_err)
            irf_mean = float(np.nanmean(irf_err)) if n > 0 else np.nan
            irf_std = float(np.nanstd(irf_err)) if n > 0 else np.nan
            sign_mean = float(np.nanmean(sign_acc)) if n > 0 else np.nan
            sign_std = float(np.nanstd(sign_acc)) if n > 0 else np.nan
            irf_mae_mean = float(np.nanmean(irf_mae_arr)) if n > 0 else np.nan
            irf_mae_std = float(np.nanstd(irf_mae_arr)) if n > 0 else np.nan
            irf_corr_mean = float(np.nanmean(irf_corr_arr)) if n > 0 else np.nan
            irf_corr_std = float(np.nanstd(irf_corr_arr)) if n > 0 else np.nan
            rows.append({
                "model": model,
                "scm_type": scm_type,
                "irf_error_mean": irf_mean,
                "irf_error_std": irf_std,
                "sign_accuracy_mean": sign_mean,
                "sign_accuracy_std": sign_std,
                "irf_mae_mean": irf_mae_mean,
                "irf_mae_std": irf_mae_std,
                "irf_corr_mean": irf_corr_mean,
                "irf_corr_std": irf_corr_std,
            })

        df = pd.DataFrame(rows)
        csv_path = output_dir / "causal_summary.csv"
        df.to_csv(csv_path, index=False)

        print("\n" + "=" * 60)
        print("AGGREGATED CAUSAL RESULTS")
        print("=" * 60)
        print(df.to_string(index=False))
        print(f"\nSaved: {csv_path}")
