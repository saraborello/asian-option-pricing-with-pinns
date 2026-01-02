
import math
import time
import numpy as np
import pandas as pd
import yfinance as yf


import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def safe_one_minus_exp_over_r(r, tau, eps=1e-6):
   """
   Computes (1 - exp(-r*tau))/r safely.
   If r ~ 0 => limit is tau.
   """
   return torch.where(
       torch.abs(r) > eps,
       (1.0 - torch.exp(-r * tau)) / r,
       tau
   )


def payoff_K1_T1(I_tilde):
   # Terminal payoff for fixed-strike call, K=1, T=1: max(I - 1, 0)
   return torch.clamp(I_tilde - 1.0, min=0.0)
