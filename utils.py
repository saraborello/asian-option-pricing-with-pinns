
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
# dataset
N_LIBRARY        = 3000
N_COLLOCATION    = 200_000
N_TERMINAL       = 25_000
N_BOUNDARY       = 20_000
N_VALIDATION_MC  = 1_000

F0_RANGE         = (40.0, 110.0)
MONEY_RANGE      = (0.5, 1.5)     # K/F0
SIGMA_RANGE      = (0.15, 0.60)
R_RANGE          = (0.00, 0.05)

AVG_BUSINESS_DAYS = 20
def make_business_calendar(n_days=AVG_BUSINESS_DAYS):
    dt_years = 1.0 / 252.0
    return n_days, dt_years

def simulate_future_gbm(F0, sigma, n_steps, dt_years):
    W = np.random.normal(size=n_steps)
    Fs = np.empty(n_steps + 1, dtype=float)
    Fs[0] = F0
    for i in range(1, n_steps + 1):
        Fs[i] = Fs[i-1] * math.exp(
            -0.5 * sigma**2 * dt_years + sigma * math.sqrt(dt_years) * W[i-1]
        )
    return Fs

def arithmetic_average(arr):
    return float(np.mean(arr))


def build_library_balanced(n_paths=N_LIBRARY,
                           F0_range=F0_RANGE,
                           moneyness_range=MONEY_RANGE,
                           sigma_range=SIGMA_RANGE,
                           r_range=R_RANGE):
    """
    Libreria bilanciata di stati (tau, F, A, K, r, sigma, T_years, F0_ref).
    """
    n_days, dt_years = make_business_calendar()
    T_years = n_days * dt_years

    rows = []
    for _ in range(n_paths):
        F0       = np.random.uniform(*F0_range)
        sigma    = np.random.uniform(*sigma_range)
        r        = np.random.uniform(*r_range)
        m        = np.random.uniform(*moneyness_range)
        K        = F0 * m

        F_path = simulate_future_gbm(F0, sigma, n_days, dt_years)

        for j in range(1, n_days + 1):
            tau = 1.0 - j / n_days
            A_j = arithmetic_average(F_path[:j+1])
            rows.append({
                "tau": tau,
                "F": F_path[j],
                "A": A_j,
                "K": K,
                "r": r,
                "sigma": sigma,
                "T_years": T_years,
                "F0_ref": F0
            })
    return pd.DataFrame(rows)


def rnd_series(s, dec):
    s = pd.to_numeric(s, errors="coerce")
    return np.round(s.astype(float), dec)

def make_gid_from_available(df, cols_specs):
    """
    Build a group id (gid) using available columns only.
    cols_specs: list of (column, decimals)
    """
    parts = []
    for c, d in cols_specs:
        if c in df.columns:
            parts.append(rnd_series(df[c], d).astype(str))
    if not parts:
        return pd.Series(["gid0"] * len(df), index=df.index)
    return parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep="|")