#!/usr/bin/env python3
"""Plot the factor-recovery comparison grids (Appendix figures) from saved results.

Reads factors_true.npy / factors_hat.npy under results/factor_recovery/{dynamic,static}/<model>/
and writes dynamic_factor_comparison.png and static_factor_comparison.png.

Run from the project root:
    uv run python scripts/plot_factor_recovery.py [--results results/factor_recovery] [--out results/figures]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _rescale_recovered_to_true_scale(f_true: np.ndarray, f_hat: np.ndarray) -> np.ndarray:
    """Rescale f_hat (per column) to match f_true mean and std for visualization. Correlation unchanged."""
    f_hat = f_hat.copy()
    for j in range(f_hat.shape[1]):
        mt, st = np.mean(f_true[:, j]), np.std(f_true[:, j])
        mh, sh = np.mean(f_hat[:, j]), np.std(f_hat[:, j])
        if sh > 1e-12:
            f_hat[:, j] = (f_hat[:, j] - mh) * (st / sh) + mt
        else:
            f_hat[:, j] = mt
    return f_hat


def _match_factors_by_correlation(f_true: np.ndarray, f_hat: np.ndarray) -> tuple:
    """Match recovered factors to true factors by correlation (Hungarian). Returns (f_true, f_hat, correlations)."""
    from scipy.optimize import linear_sum_assignment
    r = min(f_true.shape[1], f_hat.shape[1])
    f_true = f_true[:, :r].copy()
    f_hat = f_hat[:, :r].copy()
    if r == 0:
        return f_true, f_hat, np.zeros(0)
    f_true_std = np.std(f_true, axis=0)
    f_hat_std = np.std(f_hat, axis=0)
    if np.any(f_true_std < 1e-10):
        m = f_true_std < 1e-10
        f_true[:, m] += np.random.randn(f_true.shape[0], m.sum()) * 1e-8
    if np.any(f_hat_std < 1e-10):
        m = f_hat_std < 1e-10
        f_hat[:, m] += np.random.randn(f_hat.shape[0], m.sum()) * 1e-8
    try:
        corr_matrix = np.corrcoef(f_true.T, f_hat.T)
        if np.any(~np.isfinite(corr_matrix)):
            return f_true, f_hat, np.ones(r) * 0.5
        cross_corr = np.abs(np.where(np.isfinite(corr_matrix[:r, r:]), corr_matrix[:r, r:], 0.0))
        cost = 1.0 - cross_corr
        if np.any(~np.isfinite(cost)):
            return f_true, f_hat, np.ones(r) * 0.5
        row_ind, col_ind = linear_sum_assignment(cost)
        correlations = np.array([cross_corr[row_ind[i], col_ind[i]] for i in range(r)])
        return f_true[:, row_ind], f_hat[:, col_ind], np.where(np.isfinite(correlations), correlations, 0.0)
    except Exception:
        return f_true, f_hat, np.ones(r) * 0.5


def _align_recovered_to_true_order(f_true: np.ndarray, f_hat: np.ndarray) -> tuple:
    """
    Align recovered factors so that column j shows (true factor j, recovered factor that best matches true j).
    Returns (f_true unchanged, f_hat reordered, correlations per column).
    Use this for plots so that the same ground-truth curves appear in every row (column j = true factor j).
    """
    from scipy.optimize import linear_sum_assignment
    r = min(f_true.shape[1], f_hat.shape[1])
    f_true = f_true[:, :r].copy()
    f_hat = f_hat[:, :r].copy()
    if r == 0:
        return f_true, f_hat, np.zeros(0)
    f_true_std = np.std(f_true, axis=0)
    f_hat_std = np.std(f_hat, axis=0)
    if np.any(f_true_std < 1e-10):
        m = f_true_std < 1e-10
        f_true[:, m] += np.random.randn(f_true.shape[0], m.sum()) * 1e-8
    if np.any(f_hat_std < 1e-10):
        m = f_hat_std < 1e-10
        f_hat[:, m] += np.random.randn(f_hat.shape[0], m.sum()) * 1e-8
    try:
        corr_matrix = np.corrcoef(f_true.T, f_hat.T)
        if np.any(~np.isfinite(corr_matrix)):
            return f_true, f_hat, np.ones(r) * 0.5
        cross_corr = np.abs(np.where(np.isfinite(corr_matrix[:r, r:]), corr_matrix[:r, r:], 0.0))
        cost = 1.0 - cross_corr
        if np.any(~np.isfinite(cost)):
            return f_true, f_hat, np.ones(r) * 0.5
        # row_ind[k] = true index, col_ind[k] = hat index in the optimal matching
        row_ind, col_ind = linear_sum_assignment(cost)
        # Column j should show true j and the recovered factor that matched true j: perm_hat[j] = col_ind[k] where row_ind[k]==j
        perm_hat = np.zeros(r, dtype=int)
        for k in range(r):
            perm_hat[row_ind[k]] = col_ind[k]
        correlations = np.array([cross_corr[j, perm_hat[j]] for j in range(r)])
        return f_true, f_hat[:, perm_hat], np.where(np.isfinite(correlations), correlations, 0.0)
    except Exception:
        return f_true, f_hat, np.ones(r) * 0.5


def plot_dynamic_factor_comparison_from_results(
    results_dir: Path,
    output_path: Path,
    ordered_models: list[str] = None,
    n_cols: int = 5,
) -> None:
    """
    Load per-model factors from results_dir/dynamic/{model}/factors_*.npy and save a 5×5 comparison plot.
    Run factor recovery first so factors_true.npy and factors_hat.npy exist per model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if ordered_models is None:
        ordered_models = ["ivdfm", "ddfm", "ivae", "vae", "dfm"]
    true_color = "black"
    recovered_color = "#87CEEB"  # ice blue / sky blue
    dynamic_dir = results_dir / "dynamic"
    if not dynamic_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {dynamic_dir}")
    models_dict = {}
    for model_name in ordered_models:
        true_path = dynamic_dir / model_name / "factors_true.npy"
        hat_path = dynamic_dir / model_name / "factors_hat.npy"
        if not true_path.exists() or not hat_path.exists():
            continue
        models_dict[model_name] = (np.load(true_path), np.load(hat_path))
    if not models_dict:
        raise FileNotFoundError(
            f"No factors found under {dynamic_dir}; run factor recovery to generate factors_*.npy per model."
        )
    # One canonical ground truth for the whole grid so black (True) lines are identical in every row
    first_model = next((m for m in ordered_models if m in models_dict), None)
    f_true_canonical, _ = models_dict[first_model]
    if f_true_canonical.shape[1] < n_cols:
        f_true_canonical = np.hstack([f_true_canonical, np.zeros((f_true_canonical.shape[0], n_cols - f_true_canonical.shape[1]))])
    n_rows = len(ordered_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for row_idx, model_name in enumerate(ordered_models):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if model_name not in models_dict:
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(f"{model_name.upper()}\n(N/A)", fontsize=12, fontweight="bold")
                continue
        if model_name not in models_dict:
            continue
        _, f_hat_raw = models_dict[model_name]
        # Align without padding f_hat so Hungarian uses only real factors; then pad for display
        _, f_hat_matched, correlations = _align_recovered_to_true_order(f_true_canonical, f_hat_raw)
        if f_hat_matched.shape[1] < n_cols:
            f_hat_matched = np.hstack([f_hat_matched, np.zeros((f_hat_matched.shape[0], n_cols - f_hat_matched.shape[1]))])
        f_hat_display = _rescale_recovered_to_true_scale(f_true_canonical, f_hat_matched)
        for factor_idx in range(n_cols):
            ax = axes[row_idx, factor_idx]
            ax.plot(f_true_canonical[:, factor_idx], color=true_color, linestyle="--", linewidth=1.5, label="True", alpha=0.9)
            ax.plot(f_hat_display[:, factor_idx], color=recovered_color, linewidth=1.5, label="Recovered", alpha=0.9)
            corr = float(correlations[factor_idx]) if factor_idx < len(correlations) else 0.0
            ax.text(0.02, 0.98, f"corr: {corr:.2f}", transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            if row_idx == 0:
                ax.set_title(f"Factor {factor_idx + 1}")
            if factor_idx == 0:
                ax.set_ylabel(model_name.upper(), fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_static_factor_comparison_from_results(
    results_dir: Path,
    output_path: Path,
    ordered_models: list[str] = None,
    n_cols: int = 5,
) -> None:
    """
    Load per-model factors from results_dir/static/{model}/factors_*.npy and save a 5×5 comparison plot.
    Run factor recovery first so factors_true.npy and factors_hat.npy exist per model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if ordered_models is None:
        ordered_models = ["ivdfm", "ddfm", "ivae", "vae", "dfm"]
    true_color = "black"
    recovered_color = "#87CEEB"  # ice blue / sky blue
    static_dir = results_dir / "static"
    if not static_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {static_dir}")
    models_dict = {}
    for model_name in ordered_models:
        true_path = static_dir / model_name / "factors_true.npy"
        hat_path = static_dir / model_name / "factors_hat.npy"
        if not true_path.exists() or not hat_path.exists():
            continue
        models_dict[model_name] = (np.load(true_path), np.load(hat_path))
    if not models_dict:
        raise FileNotFoundError(
            f"No factors found under {static_dir}; run factor recovery to generate factors_*.npy per model."
        )
    # One canonical ground truth for the whole grid so black (True) lines are identical in every row
    first_model = next((m for m in ordered_models if m in models_dict), None)
    f_true_canonical, _ = models_dict[first_model]
    if f_true_canonical.shape[1] < n_cols:
        f_true_canonical = np.hstack([f_true_canonical, np.zeros((f_true_canonical.shape[0], n_cols - f_true_canonical.shape[1]))])
    n_rows = len(ordered_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for row_idx, model_name in enumerate(ordered_models):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if model_name not in models_dict:
                ax.axis("off")
                if col_idx == 0:
                    ax.set_ylabel(f"{model_name.upper()}\n(N/A)", fontsize=12, fontweight="bold")
                continue
        if model_name not in models_dict:
            continue
        _, f_hat_raw = models_dict[model_name]
        _, f_hat_matched, correlations = _align_recovered_to_true_order(f_true_canonical, f_hat_raw)
        if f_hat_matched.shape[1] < n_cols:
            f_hat_matched = np.hstack([f_hat_matched, np.zeros((f_hat_matched.shape[0], n_cols - f_hat_matched.shape[1]))])
        f_hat_display = _rescale_recovered_to_true_scale(f_true_canonical, f_hat_matched)
        for factor_idx in range(n_cols):
            ax = axes[row_idx, factor_idx]
            ax.plot(f_true_canonical[:, factor_idx], color=true_color, linestyle="--", linewidth=1.5, label="True", alpha=0.9)
            ax.plot(f_hat_display[:, factor_idx], color=recovered_color, linewidth=1.5, label="Recovered", alpha=0.9)
            corr = float(correlations[factor_idx]) if factor_idx < len(correlations) else 0.0
            ax.text(0.02, 0.98, f"corr: {corr:.2f}", transform=ax.transAxes, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
            if row_idx == 0:
                ax.set_title(f"Factor {factor_idx + 1}")
            if factor_idx == 0:
                ax.set_ylabel(model_name.upper(), fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(description="Plot factor-recovery comparison grids")
    p.add_argument("--results", type=Path, default=Path("results/factor_recovery"))
    p.add_argument("--out", type=Path, default=Path("results/figures"))
    args = p.parse_args()
    plot_dynamic_factor_comparison_from_results(args.results, args.out / "dynamic_factor_comparison.png")
    plot_static_factor_comparison_from_results(args.results, args.out / "static_factor_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
