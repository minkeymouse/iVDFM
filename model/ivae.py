"""
VAE/iVAE implementation for factor recovery experiments.

Provides a unified Model class interface similar to DFM/DDFM wrappers, with training loops
for VAE and iVAE models.
"""

from __future__ import annotations

from numbers import Number
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler


def _check_inputs(size, mu, v, device=None):
    """helper function to ensure inputs are compatible"""
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
            if device is not None:
                mu = mu.to(device)
        if v is None:
            v = torch.Tensor([1])
            if device is not None:
                v = v.to(device)
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
            if device is not None:
                v = v.to(device)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v, device=x.device if hasattr(x, 'device') else None)
    else:
        mu, v = _check_inputs(x.size(), mu, v, device=x.device if hasattr(x, 'device') else None)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h


class cleanIVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
        return elbo, z


class cleanVAE(nn.Module):

    def __init__(self, data_dim, latent_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1)
        self.prior_var = torch.ones(1)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def forward(self, x):
        g, v = self.encoder(x)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z = self.forward(x)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps = log_normal(z, None, None, broadcast_size=True).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps)).mean()
        return elbo, z


class Model:
    """
    VAE/iVAE model wrapper with unified interface.
    
    Supports both VAE (unconditional prior) and iVAE (conditional prior with auxiliary variables).
    Follows the same API pattern as model/dfm.py and model/ddfm.py.
    """

    def __init__(
        self,
        *,
        model_type: str = "vae",
        config_dict: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize VAE/iVAE model.
        
        Parameters
        ----------
        model_type : str
            Either "vae" or "ivae"
        config_dict : dict, optional
            Model configuration dictionary
        device : torch.device, optional
            Device to run on. If None, uses CPU or auto-detects GPU.
        """
        self.model_type = model_type.lower()
        if self.model_type not in ["vae", "ivae"]:
            raise ValueError(f"model_type must be 'vae' or 'ivae', got '{model_type}'")
        
        self.config_dict = config_dict or {}
        # Force GPU usage for VAE/iVAE when available, regardless of device parameter
        if torch.cuda.is_available():
            # Always prefer GPU for VAE/iVAE when available
            if device is not None and device.type == "cuda":
                self.device = device  # Use provided GPU device
            else:
                self.device = torch.device("cuda:0")  # Force GPU even if CPU was passed
            print(f"  [{self.model_type.upper()}] Using GPU: {self.device}")
        else:
            # Fallback to CPU only if GPU is not available
            self.device = device or torch.device("cpu")
            print(f"  [{self.model_type.upper()}] Using CPU (GPU not available)")
        self._model = None
        self._u_train = None  # Store auxiliary variables for factor extraction
        self._x_full = None  # Store full dataset for factor extraction

    def fit(
        self,
        df_train: pd.DataFrame,
        *,
        u_train: Optional[np.ndarray] = None,
        target_scaler: Optional[StandardScaler | RobustScaler] = None,
    ) -> "Model":
        """
        Fit (train) VAE or iVAE model on training data.
        
        Parameters
        ----------
        df_train : pd.DataFrame
            Training data (T x N)
        u_train : np.ndarray, optional
            Auxiliary variables (T x aux_dim). Required for iVAE, ignored for VAE.
        target_scaler : StandardScaler | RobustScaler, optional
            Scaler (not used for VAE/iVAE, kept for API compatibility)
            
        Returns
        -------
        Model
            Self for method chaining
        """
        # Extract configuration
        data_dim = df_train.shape[1]
        latent_dim = int(self.config_dict.get("latent_dim", 2))
        n_layers = int(self.config_dict.get("n_layers", 3))
        hidden_dim = int(self.config_dict.get("hidden_dim", 100))
        activation = str(self.config_dict.get("activation", "xtanh"))
        slope = float(self.config_dict.get("slope", 0.1))
        
        learning_rate = float(self.config_dict.get("learning_rate", 0.01))
        batch_size = int(self.config_dict.get("batch_size", 64))
        epochs = int(self.config_dict.get("epochs", 20))
        a = float(self.config_dict.get("a", 100.0))
        b = float(self.config_dict.get("b", 1.0))
        c = float(self.config_dict.get("c", 1.0))
        d = float(self.config_dict.get("d", 1.0))
        anneal = bool(self.config_dict.get("anneal", False))
        no_scheduler = bool(self.config_dict.get("no_scheduler", False))
        
        # Convert DataFrame to numpy and then to tensor
        x_np = df_train.values.astype(np.float32)
        x_tensor = torch.from_numpy(x_np).to(self.device)
        self._x_full = x_tensor  # Store for factor extraction
        
        # Handle auxiliary variables for iVAE
        if self.model_type == "ivae":
            if u_train is None:
                raise ValueError("u_train is required for iVAE model")
            aux_dim = u_train.shape[1]
            u_tensor = torch.from_numpy(u_train.astype(np.float32)).to(self.device)
            self._u_train = u_tensor
        else:
            aux_dim = 0  # Not used for VAE
            u_tensor = None
        
        # Create model
        if self.model_type == "ivae":
            self._model = cleanIVAE(
                data_dim=data_dim,
                latent_dim=latent_dim,
                aux_dim=aux_dim,
                n_layers=n_layers,
                activation=activation,
                hidden_dim=hidden_dim,
                slope=slope,
            ).to(self.device)
        else:  # VAE
            self._model = cleanVAE(
                data_dim=data_dim,
                latent_dim=latent_dim,
                n_layers=n_layers,
                activation=activation,
                hidden_dim=hidden_dim,
                slope=slope,
            ).to(self.device)
        
        # Create DataLoader
        # NOTE:
        # - We deliberately set drop_last=False here. With the very small synthetic
        #   smoke-test setups (e.g. T=60 and batch_size=64), using drop_last=True
        #   would yield zero batches, which in turn causes a ZeroDivisionError
        #   when computing epoch averages and prevents any training from
        #   happening. For our use-cases, keeping the final (possibly smaller)
        #   batch is perfectly fine.
        if self.model_type == "ivae":
            dataset = TensorDataset(x_tensor, u_tensor)
        else:
            dataset = TensorDataset(x_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        # Setup optimizer and scheduler
        optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)
        scheduler = None
        if not no_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0)
        
        # Training loop
        N = len(df_train)  # Total number of samples for ELBO calculation
        self._model.train()
        
        for epoch in range(1, epochs + 1):
            # Handle annealing schedule
            if anneal:
                a_epoch = a
                b_epoch = b
                c_epoch = c
                d_epoch = d
                if epoch > epochs / 1.6:
                    b_epoch = 1.0
                    c_epoch = 1.0
                    d_epoch = 1.0
                    a_epoch = 2.0 * a
            else:
                a_epoch = a
                b_epoch = b
                c_epoch = c
                d_epoch = d
            
            epoch_loss = 0.0
            for batch_data in data_loader:
                if self.model_type == "ivae":
                    x_batch, u_batch = batch_data
                    x_batch = x_batch.to(self.device)
                    u_batch = u_batch.to(self.device)
                else:
                    x_batch = batch_data[0].to(self.device)
                    u_batch = None
                
                optimizer.zero_grad()
                
                # Compute ELBO loss
                if self.model_type == "ivae":
                    loss, _ = self._model.elbo(x_batch, u_batch, N, a=a_epoch, b=b_epoch, c=c_epoch, d=d_epoch)
                else:
                    loss, _ = self._model.elbo(x_batch, u_batch, N, a=a_epoch, b=b_epoch, c=c_epoch, d=d_epoch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Update scheduler (only if we saw at least one batch)
            if scheduler is not None and len(data_loader) > 0:
                avg_loss = epoch_loss / len(data_loader)
                scheduler.step(avg_loss)
        
        return self

    def factors(self) -> np.ndarray:
        """
        Extract latent factors from the trained model.
        
        Returns
        -------
        np.ndarray
            Extracted factors (T x latent_dim)
        """
        if self._model is None:
            raise ValueError("Call fit() before factors().")
        
        self._model.eval()
        
        with torch.no_grad():
            # Extract factors using encoder on full dataset
            if self.model_type == "ivae":
                if self._u_train is None:
                    raise ValueError("Auxiliary variables u_train not stored. Cannot extract factors.")
                # Extract factors using encoder
                g, v = self._model.encoder(self._x_full, self._u_train)
                # Use mean of posterior (g) as factors
                z = g
            else:
                # For VAE
                g, v = self._model.encoder(self._x_full)
                # Use mean of posterior (g) as factors
                z = g
        
        return z.cpu().numpy()
    
    def sample_posterior(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the posterior distribution p(z|x) or p(z|x,u).
        
        For iVAE: samples from p(z|x,u) (conditional posterior)
        For VAE: samples from p(z|x) (posterior)
        
        Parameters
        ----------
        n_samples : int
            Number of samples per time step (default: 1)
            
        Returns
        -------
        np.ndarray
            Sampled factors (T x latent_dim) if n_samples=1, or (T x n_samples x latent_dim) if n_samples>1
        """
        if self._model is None:
            raise ValueError("Call fit() before sample_posterior().")
        
        self._model.eval()
        
        with torch.no_grad():
            if self.model_type == "ivae":
                if self._u_train is None:
                    raise ValueError("Auxiliary variables u_train not stored. Cannot sample from posterior.")
                g, v = self._model.encoder(self._x_full, self._u_train)
                # Sample from p(z|x,u): mean=g, variance=v
                z = self._model.reparameterize(g, v)
            else:
                # For VAE: sample from p(z|x)
                g, v = self._model.encoder(self._x_full)
                z = self._model.reparameterize(g, v)
        
        return z.cpu().numpy()
    
    def sample_prior(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the prior distribution.
        
        For iVAE: samples from p(z|u) (conditional prior) - this shows identification benefit!
        For VAE: samples from p(z) (unconditional prior)
        
        Parameters
        ----------
        n_samples : int
            Number of samples per time step (default: 1)
            
        Returns
        -------
        np.ndarray
            Sampled factors (T x latent_dim) if n_samples=1, or (T x n_samples x latent_dim) if n_samples>1
        """
        if self._model is None:
            raise ValueError("Call fit() before sample_prior().")
        
        self._model.eval()
        
        with torch.no_grad():
            if self.model_type == "ivae":
                if self._u_train is None:
                    raise ValueError("Auxiliary variables u_train not stored. Cannot sample from conditional prior.")
                # Sample from p(z|u): conditional prior
                # Prior mean is zero, variance is l = prior(u)
                l = self._model.prior(self._u_train)  # (T, latent_dim) - variance
                prior_mean = torch.zeros_like(l)  # Mean is zero
                z = self._model.reparameterize(prior_mean, l)
            else:
                # For VAE: sample from p(z) = N(0, I)
                T = self._x_full.shape[0]
                latent_dim = self._model.latent_dim
                z = torch.randn(T, latent_dim, device=self._x_full.device)
        
        return z.cpu().numpy()
