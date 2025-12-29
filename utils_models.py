import torch
import numpy as np
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

np.random.seed(42)

def normalize_df(df):
    df = df.copy()
    s = df["F0_ref"].values
    df["F_n"]     = df["F"] / s
    df["A_n"]     = df["A"] / s
    df["K_n"]     = df["K"] / s
    df["r_n"]     = df["r"]
    df["sigma_n"] = df["sigma"]
    df["tau_n"]   = df["tau"]
    df["Tn"]      = df["T_years"]
    return df

def pack_inputs(df_n):
    X = np.stack([
        df_n["tau_n"].values,
        df_n["F_n"].values,
        df_n["A_n"].values,
        df_n["K_n"].values,
        df_n["r_n"].values,
        df_n["sigma_n"].values,
        df_n["Tn"].values
    ], axis=1)
    return torch.tensor(X, dtype=torch.float32, device=DEVICE)