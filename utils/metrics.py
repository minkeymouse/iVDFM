import torch
import numpy as np
from typing import List


# ============================================================
# Probabilistic: CRPS from quantiles
# ============================================================

def crps_from_quantiles(
    y_true: np.ndarray,
    quantile_forecasts: dict,
    quantiles: List[float],
) -> float:
    """
    CRPS approximated from predicted quantiles using pinball loss.

    CRPS(y) ≈ (1/K) * sum_k rho_{q_k}(y_true - y_hat_{q_k})
    where rho_q(e) = max(q*e, (q-1)*e) is the pinball loss.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth, shape (horizon, n_series) or (n,).
    quantile_forecasts : dict
        Maps quantile q (float) to array of same shape as y_true.
    quantiles : List[float]
        Quantile levels used (e.g. [0.1, 0.5, 0.9]).

    Returns
    -------
    float
        Mean CRPS (≥ 0). Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    crps = 0.0
    for q in quantiles:
        if q not in quantile_forecasts:
            raise KeyError(f"Quantile {q} not in quantile_forecasts")
        y_q = np.asarray(quantile_forecasts[q], dtype=np.float64)
        if y_q.shape != y_true.shape:
            y_q = y_q.reshape(y_true.shape)
        diff = y_true - y_q  # (H, N)
        # Pinball: rho_q(e) = max(q*e, (q-1)*e)
        crps += np.mean(np.maximum(q * diff, (q - 1) * diff))
    return float(crps / len(quantiles))


def _standardize(x):
    """Standardize columns. x must be torch.Tensor (use _to_tensor upstream if needed)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


def pearson_corr(x, y):
    """
    x, y: (T, d) torch.Tensor or np.ndarray (converted to torch).
    returns: (d, d)
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    x = _standardize(x)
    y = _standardize(y)
    return x.T @ y / x.size(0)


def rankdata(x):
    """
    Column-wise ordinal ranking. x must be torch.Tensor (use conversion upstream if needed).
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    return torch.argsort(torch.argsort(x, dim=0), dim=0).float() + 1.0


def spearman_corr(x, y):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    return pearson_corr(rankdata(x), rankdata(y))


# ============================================================
# Auction assignment (permutation matching)
# ============================================================

def auction_assignment(x, eps=None, max_iter: int = 10000):
    """
    Solve max-sum assignment using auction algorithm.
    x: (d, d) torch.Tensor (or converted from np.ndarray).
    max_iter: hard cap to avoid infinite loop on ties/near-ties.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    d = x.size(0)
    eps = 1.0 / d if eps is None else eps

    price = torch.zeros((1, d), device=x.device)
    assignment = torch.full((d,), -1, device=x.device, dtype=torch.long)
    bids = torch.zeros_like(x)
    it = 0
    while (assignment == -1).any() and it < max_iter:
        it += 1
        unassigned = (assignment == -1).nonzero(as_tuple=False).squeeze(1)
        if unassigned.dim() == 0:
            break

        value = x[unassigned] - price
        top2_val, top2_idx = value.topk(2, dim=1)

        best_item = top2_idx[:, 0]
        bid_inc = top2_val[:, 0] - top2_val[:, 1] + eps

        bids.zero_()
        bids[unassigned, best_item] = bid_inc

        items = (bids > 0).any(dim=0).nonzero(as_tuple=False).squeeze(1)
        if items.numel() == 0:
            break
        max_bid, winner_idx = bids[:, items].max(dim=0)

        price[:, items] += max_bid

        assignment[torch.isin(assignment, items)] = -1
        assignment[winner_idx] = items

    return assignment


# ============================================================
# Permutation-invariant latent recovery metric (MCC)
# ============================================================

def mean_corr_coef(z_true, z_hat, method="pearson"):
    """
    MCC: Mean Correlation Coefficient. Valid only when z_true has identifiable
    ground-truth semantics and z_hat is recovering those factors (e.g. synthetic
    factor recovery, oracle experiments). Do not use for cross-model stability
    or representation-similarity comparison.
    z_true, z_hat: (T, d) torch.Tensor or np.ndarray (converted to torch).
    """
    if isinstance(z_true, np.ndarray):
        z_true = torch.from_numpy(np.asarray(z_true, dtype=np.float64))
    if isinstance(z_hat, np.ndarray):
        z_hat = torch.from_numpy(np.asarray(z_hat, dtype=np.float64))
    if method == "pearson":
        C = pearson_corr(z_true, z_hat)
    elif method == "spearman":
        C = spearman_corr(z_true, z_hat)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    C = C.abs()
    assignment = auction_assignment(C)

    return C[torch.arange(C.size(0)), assignment].mean()


# ============================================================
# Subspace metric (rotation-invariant, not permutation-based)
# ============================================================

def subspace_distance(z_true, z_hat):
    """
    Principal angle distance between column spaces. Valid for static factor
    recovery with the same sampling scheme. Not valid for temporal stability
    or windowed vs timestep representations; do not report alongside representation-similarity metrics.
    z_true, z_hat: (T, d) torch.Tensor or np.ndarray (converted to torch).
    """
    if isinstance(z_true, np.ndarray):
        z_true = torch.from_numpy(np.asarray(z_true, dtype=np.float64))
    if isinstance(z_hat, np.ndarray):
        z_hat = torch.from_numpy(np.asarray(z_hat, dtype=np.float64))
    T = min(z_true.size(0), z_hat.size(0))
    z_true = z_true[:T]
    z_hat = z_hat[:T]

    Qa, _ = torch.linalg.qr(z_true)
    Qb, _ = torch.linalg.qr(z_hat)

    s = torch.linalg.svdvals(Qa.T @ Qb)
    s = torch.clamp(s, 0.0, 1.0)

    return torch.mean(torch.acos(s))


def trace_r2(z_true, z_hat):
    """
    Trace R² (precision score) between true and recovered factors.
    
    This metric measures how well the recovered factors span the true factor space.
    Used in the original DDFM paper (Andreini et al., 2020).
    
    Formula:
        trace_R2 = trace(F_true^T @ F_hat @ pinv(F_hat^T @ F_hat) @ F_hat^T @ F_true) 
                   / trace(F_true^T @ F_true)
    
    Parameters
    ----------
    z_true : torch.Tensor
        True factors of shape (T, d)
    z_hat : torch.Tensor
        Recovered factors of shape (T, d)
        
    Returns
    -------
    torch.Tensor
        Trace R² score (higher is better, range: 0 to 1, where 1 = perfect recovery)
    """
    T = min(z_true.size(0), z_hat.size(0))
    z_true = z_true[:T]
    z_hat = z_hat[:T]
    
    # Compute trace R²
    # numerator: trace(F_true^T @ F_hat @ pinv(F_hat^T @ F_hat) @ F_hat^T @ F_true)
    # denominator: trace(F_true^T @ F_true)
    
    # F_hat^T @ F_hat
    f_hat_t_f_hat = z_hat.T @ z_hat
    
    # pinv(F_hat^T @ F_hat) - use pseudo-inverse for numerical stability
    pinv_f_hat_t_f_hat = torch.linalg.pinv(f_hat_t_f_hat)
    
    # F_hat^T @ F_true
    f_hat_t_f_true = z_hat.T @ z_true
    
    # F_true^T @ F_hat
    f_true_t_f_hat = z_true.T @ z_hat
    
    # numerator: trace(F_true^T @ F_hat @ pinv(F_hat^T @ F_hat) @ F_hat^T @ F_true)
    numerator = torch.trace(f_true_t_f_hat @ pinv_f_hat_t_f_hat @ f_hat_t_f_true)
    
    # denominator: trace(F_true^T @ F_true)
    denominator = torch.trace(z_true.T @ z_true)
    
    # Avoid division by zero
    if denominator < 1e-8:
        return torch.tensor(0.0, device=z_true.device, dtype=z_true.dtype)
    
    return numerator / denominator


# ============================================================
# Temporal smoothness metric
# ============================================================

def temporal_smoothness(z):
    """
    Average L2 step size of concept trajectories. z is normalized per column
    before computing step sizes so the metric is comparable across models
    (scale-invariant).
    Smoothness = (1/(T-1)) * sum_{t=2}^T ||z_t - z_{t-1}||_2 on normalized z.
    z: (T, d) torch.Tensor or np.ndarray (converted to torch).
    """
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(np.asarray(z, dtype=np.float64))
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device)
    z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-8)
    diffs = z[1:] - z[:-1]  # (T-1, d)
    step_sizes = torch.norm(diffs, p=2, dim=1)  # (T-1,)
    return torch.mean(step_sizes)


