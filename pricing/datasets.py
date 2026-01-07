
import torch
import torch.nn as nn

from archive.utils import safe_one_minus_exp_over_r
from archive.utils import payoff_K1_T1

SEED = 42
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32



def sample_uniform(n, low, high, device=DEVICE):
   return (low + (high - low) * torch.rand(n, 1, device=device, dtype=DTYPE))


def sample_domain(Np, S_max, I_max, r_max, sigma_max):
   S = sample_uniform(Np, 0.0, S_max)
   I = sample_uniform(Np, 0.0, I_max)
   t = sample_uniform(Np, 0.0, 1.0)
   r = sample_uniform(Np, 0.0, r_max)
   s = sample_uniform(Np, 0.0, sigma_max)
   return torch.cat([S, I, t, r, s], dim=1)


def sample_TC(n_tc, S_max, I_max, r_max, sigma_max):
   S = sample_uniform(n_tc, 0.0, S_max)
   I = sample_uniform(n_tc, 0.0, I_max)
   t = torch.ones(n_tc, 1, device=DEVICE, dtype=DTYPE)  # t = T = 1
   r = sample_uniform(n_tc, 0.0, r_max)
   s = sample_uniform(n_tc, 0.0, sigma_max)
   X = torch.cat([S, I, t, r, s], dim=1)
   Y = payoff_K1_T1(I)  # (n,1)
   return X, Y


def sample_BC_S0(n_bc, S_max, I_max, r_max, sigma_max):
   S = torch.zeros(n_bc, 1, device=DEVICE, dtype=DTYPE)  # S=0
   I = sample_uniform(n_bc, 0.0, I_max)
   t = sample_uniform(n_bc, 0.0, 1.0)
   r = sample_uniform(n_bc, 0.0, r_max)
   s = sample_uniform(n_bc, 0.0, sigma_max)
   X = torch.cat([S, I, t, r, s], dim=1)


   tau = 1.0 - t
   disc = torch.exp(-r * tau)
   Y = disc * payoff_K1_T1(I)  # approx BC (paper eq6)
   return X, Y


def sample_BC_Smax(n_bc, S_max, I_max, r_max, sigma_max):
   S = torch.full((n_bc, 1), S_max, device=DEVICE, dtype=DTYPE)  # S=Smax
   I = sample_uniform(n_bc, 0.0, I_max)
   t = sample_uniform(n_bc, 0.0, 1.0)
   r = sample_uniform(n_bc, 0.0, r_max)
   s = sample_uniform(n_bc, 0.0, sigma_max)
   X = torch.cat([S, I, t, r, s], dim=1)


   tau = 1.0 - t
   disc = torch.exp(-r * tau)
   term = disc * (I - 1.0)
   carry = S * safe_one_minus_exp_over_r(r, tau)
   Y = torch.clamp(term + carry, min=0.0)  # approx upper bound (paper eq7 form)
   return X, Y


def sample_BC_Imax(n_bc, S_max, I_max, r_max, sigma_max):
   S = sample_uniform(n_bc, 0.0, S_max)
   I = torch.full((n_bc, 1), I_max, device=DEVICE, dtype=DTYPE)  # I=Imax
   t = sample_uniform(n_bc, 0.0, 1.0)
   r = sample_uniform(n_bc, 0.0, r_max)
   s = sample_uniform(n_bc, 0.0, sigma_max)
   X = torch.cat([S, I, t, r, s], dim=1)


   tau = 1.0 - t
   disc = torch.exp(-r * tau)
   term = disc * (I_max - 1.0)
   carry = S * safe_one_minus_exp_over_r(r, tau)
   Y = term + carry  # (paper eq8)
   return X, Y
