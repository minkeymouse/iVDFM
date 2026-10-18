# iVDFM: Identifiable Variational Dynamic Factor Model

Experiment code for

> Minkey Chang and Jae-Young Kim. **Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics.** *Proceedings of The 1st Symposium on Probabilistic Machine Learning (ProbML 2026)*, PMLR 327:189–213, 2026.
> [PMLR](https://proceedings.mlr.press/v327/chang26a.html) · [arXiv:2603.22886](https://arxiv.org/abs/2603.22886)

The model itself (iVDFM, together with the DFM and DDFM baselines) is implemented in the
[`dfm-python`](https://github.com/minkeymouse/dfm-python) package, which this repository installs
from PyPI. What lives here is everything needed to reproduce the paper's experiments: synthetic
data generators, model wrappers with a common `fit` / `factors` / `predict` interface, evaluation
metrics, Hydra configs, run scripts, and the result files behind the published tables.

## Repository layout

| Path | Contents |
| --- | --- |
| `run.py` | Hydra entry point; dispatches to one of the experiments below |
| `run_*.sh` | Full paper runs: `run_factor_recovery.sh`, `run_causal_intervention.sh`, `run_forecasting.sh`, `run_hyperparameter.sh` |
| `experiments/` | `exp_factor_recovery.py`, `exp_causal_intervention.py`, `exp_forecasting.py`, `exp_application.py` |
| `model/` | Thin wrappers with a common `fit` / `factors` / `predict` interface: `ivdfm.py`, `dfm.py`, `ddfm.py` (dfm-python), `ivae.py` (VAE and iVAE baselines), `neuralforecast.py` (iTransformer, TimeMixer, TimeXer, MLPMultivariate) |
| `synthetic/` | Dynamic, static (iVAE-style) and SCM data-generating processes |
| `utils/` | Factor-recovery metrics (MCC, subspace distance, smoothness, Trace R²), rolling-origin forecast evaluation (CRPS, sMSE), model fitting |
| `data_provider/` | Loaders for the real benchmark datasets |
| `config/` | Hydra configs: `experiment/`, `dataset/`, `model/` |
| `scripts/` | Optuna searches (`tune_forecasting.py`, `tune_factor_recovery.py`) and the appendix figure script (`plot_factor_recovery.py`) |
| `results/` | Result files backing the paper's tables; see [`results/README.md`](results/README.md) |

## Setup

Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/minkeymouse/iVDFM.git
cd iVDFM
uv sync
```

`uv sync` installs `dfm-python`, PyTorch, and `neuralforecast` (only needed for the forecasting
baselines). A GPU is optional; the synthetic experiments run on CPU.

### Datasets

Synthetic experiments (factor recovery, causal intervention) generate their own data. The
forecasting and application experiments expect the standard long-term-forecasting benchmarks under
`data/` (not tracked in git):

```
data/ETT-small/ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
data/weather/weather.csv
data/illness/national_illness.csv
data/exchange/exchange_rate.csv
```

These are the files distributed with [Autoformer](https://github.com/thuml/Autoformer) /
[Time-Series-Library](https://github.com/thuml/Time-Series-Library) (ETT, Weather, Illness,
Exchange). Paths can be changed in `config/dataset/*.yaml` and `config/experiment/application.yaml`.

## Reproducing the paper

| Paper | Command | Output |
| --- | --- | --- |
| Tables 1–2 (factor recovery, dynamic and static DGPs) | `bash run_factor_recovery.sh` | `results/factor_recovery/{dynamic,static}/<model>/` |
| Table 4 and the appendix stress table (causal intervention) | `bash run_causal_intervention.sh` | `results/causal/{base,regime,chain}/` |
| Table 3 (forecasting, CRPS and sMSE) | `bash run_forecasting.sh` | `results/forecasting/<dataset>/<model>/h<horizon>.json` |
| Section 5 application, exchange rates | `uv run python run.py experiment=application` | `results/application/exchange/` |
| Section 5 application, influenza-like illness | same, with the overrides listed at the top of `config/experiment/application.yaml` | `results/application/illness/` |
| Appendix factor-recovery figures | `uv run python scripts/plot_factor_recovery.py` | `results/figures/` |
| iVDFM hyperparameter search for forecasting | `bash run_hyperparameter.sh` | `results/optuna/<dataset>_ivdfm_best.json` |

`run_causal_intervention.sh`, `run_forecasting.sh` and `run_hyperparameter.sh` accept `--test` for a
quick smoke run; trailing arguments are passed through as Hydra overrides, for example

```bash
uv run python run.py experiment=factor_recovery experiment.n_simulations=2 experiment.models='[ivdfm,ivae]'
uv run python run.py experiment=forecasting experiment.datasets='[etth1]' experiment.models='[ivdfm]' experiment.horizons='[96]'
```

The hyperparameters used in the paper are fixed in `config/experiment/*.yaml`
(`suite_model_overrides` for factor recovery, `ivdfm_dataset_params` for forecasting). The Optuna
selections behind them are recorded in `results/factor_recovery/optuna_best_params_*.json`.

## Citation

```bibtex
@InProceedings{pmlr-v327-chang26a,
  title     = {Conditionally Identifiable Latent Representation for Multivariate Time Series with Structural Dynamics},
  author    = {Chang, Minkey and Kim, Jae-Young},
  booktitle = {Proceedings of The 1st Symposium on Probabilistic Machine Learning},
  pages     = {189--213},
  year      = {2026},
  editor    = {Swaroop, Siddharth and R{\"u}gamer, David and Kristiadi, Agustinus},
  volume    = {327},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v327/chang26a.html}
}
```

## License

MIT. See [LICENSE](LICENSE).
