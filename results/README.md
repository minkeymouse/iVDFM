# Results shipped with the repository

Only the runs that back the published tables and figures are kept here.
Every directory below was produced by the corresponding `run.py` experiment
(see the top-level README for the exact commands). Forecasting results
(Table 3 and the full forecasting table in the appendix) are not shipped
because each run writes several hundred megabytes of per-origin forecasts;
re-run `run_forecasting.sh` to regenerate them.

| Directory | Paper reference | Experiment |
| --- | --- | --- |
| `factor_recovery/dynamic/` | Table 1 (dynamic DGP, T=200, N=20, r=5) | `experiment=factor_recovery`, suite `dynamic` |
| `factor_recovery/static/` | Table 2 (static DGP, T=200, N=20, r=5) | `experiment=factor_recovery`, suite `static` |
| `factor_recovery/optuna_best_params_*.json` | Appendix C, hyperparameter protocol | Optuna selection used for the two suites |
| `causal_intervention/{base,regime,chain}/` | Table 4 (rows T=200, r=3) and the full stress table in Appendix D | `experiment=causal_intervention` with `T_list` / `r_fit_list` |
| `ivae_static_diagnostic/` | Appendix D, iVAE static diagnostic table | `experiment=factor_recovery`, static suite, T=2000, n_seg=20, linear mixing, 400 epochs, 5 simulations |
| `application/exchange/` | Section 5, daily exchange-rate application (r=2) and its figure | `experiment=application` (AUD, JPY, CAD, CNY, NZD vs USD, 2007-10-01 to 2009-03-31) |
| `application/illness/` | Section 5, weekly influenza-like illness application (r=2) | `experiment=application` on `national_illness.csv` |

Per model directory, `factor_recovery_summary.csv` holds the mean and standard
deviation of MCC, subspace distance, smoothness and Trace R^2 over simulations,
`factor_recovery_detailed.json` the per-simulation values, and `factors_true.npy`
/ `factors_hat.npy` the ground-truth and recovered factors of the last simulation.
The application directories contain the recovered factors (`factors.npy`),
factor-to-series correlations (`factor_series_corr.csv`) and the standardized
input window (`window_scaled.csv`).
