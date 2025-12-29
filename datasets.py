
from datetime import datetime, timedelta
import os
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import make_business_calendar
from utils import simulate_future_gbm
from utils import arithmetic_average
from utils import build_library_balanced

AVG_BUSINESS_DAYS = 20
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

def sample_collocation_balanced(df_lib, n=N_COLLOCATION):
    df = df_lib.copy()
    # oversampling near ATM e near-maturity
    df["atm_score"] = 1.0 / (1e-6 + np.abs(df["A"] - df["K"]))
    df["tau_score"] = 1.0 / (1e-3 + df["tau"])
    score = 0.5 * df["atm_score"] + 0.5 * df["tau_score"]
    prob = score / score.sum()
    idx = np.random.choice(len(df), size=min(n, len(df)), replace=False, p=prob)
    return df.iloc[idx][["tau","F","A","K","r","sigma","T_years","F0_ref"]].reset_index(drop=True)

def sample_terminal_balanced(df_lib, n=N_TERMINAL):
    df = df_lib[df_lib["tau"] <= (1.0/AVG_BUSINESS_DAYS)].copy()
    if df.empty:
        df = df_lib.copy()
    df = df.sample(n=min(n, len(df)), replace=False)
    df["target_V"] = np.maximum(df["A"] - df["K"], 0.0)
    return df[["tau","F","A","K","r","sigma","T_years","F0_ref","target_V"]].reset_index(drop=True)

def sample_boundary_balanced(df_lib, n=N_BOUNDARY):
    df = df_lib.sample(n=min(n, len(df_lib)), replace=False).copy()

    Fmin = 0.3 * df["F0_ref"]
    Fmax = 3.0 * df["F0_ref"]
    Amin = 0.3 * df["F0_ref"]
    Amax = 3.0 * df["F0_ref"]

    half = len(df) // 2
    dfF = df.iloc[:half].copy()
    dfA = df.iloc[half:].copy()

    choiceF = np.random.rand(half) < 0.5
    dfF.loc[choiceF, "F"]  = Fmin.iloc[:half][choiceF].values
    dfF.loc[~choiceF, "F"] = Fmax.iloc[:half][~choiceF].values
    dfF["target_V"] = np.clip(dfF["A"] - dfF["K"], 0.0, dfF["A"])

    choiceA = np.random.rand(len(dfA)) < 0.5
    dfA.loc[choiceA, "A"]  = Amin.iloc[half:][choiceA].values
    dfA.loc[~choiceA, "A"] = Amax.iloc[half:][~choiceA].values
    dfA["target_V"] = np.clip(dfA["A"] - dfA["K"], 0.0, dfA["A"])

    out = pd.concat([dfF, dfA], ignore_index=True)
    return out[["tau","F","A","K","r","sigma","T_years","F0_ref","target_V"]].reset_index(drop=True)


def price_asian_arith_mc(F0, K, r, sigma, n_steps, dt_years, n_paths=50_000):
    disc = math.exp(-r * (n_steps * dt_years))
    pay = []
    for _ in range(n_paths):
        F = simulate_future_gbm(F0, sigma, n_steps, dt_years)
        A = arithmetic_average(F)
        pay.append(max(A - K, 0.0))
    return disc * (sum(pay) / len(pay))

def build_validation_mc_balanced(n=N_VALIDATION_MC,
                                 F0_range=F0_RANGE,
                                 moneyness_range=MONEY_RANGE,
                                 sigma_range=SIGMA_RANGE,
                                 r_range=R_RANGE):
    n_days, dt_years = make_business_calendar()
    rows = []

    F0s       = np.random.uniform(*F0_range, size=n)
    moneyness = np.random.uniform(*moneyness_range, size=n)
    sigmas    = np.random.uniform(*sigma_range, size=n)
    rs        = np.random.uniform(*r_range, size=n)
    Ks        = F0s * moneyness
    T_years   = np.full(n, n_days * dt_years)

    print("Calcolo prezzi Monte Carlo bilanciati...")
    for F0, K, sigma, r, T in zip(F0s, Ks, sigmas, rs, T_years):
        price_mc = price_asian_arith_mc(F0, K, r, sigma, n_days, dt_years, n_paths=8000)
        rows.append({
            "F0": F0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T_years": T,
            "price_mc": price_mc
        })
    return pd.DataFrame(rows)
