
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

HIDDEN = 256
LAYERS = 5
LR = 1e-3
EPOCHS = 3000
BATCH_C = 2048
BATCH_T = 512
BATCH_B = 512
ALPHA, BETA, GAMMA = 1.0, 2.0, 1.0
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_dim=7, out_dim=1, hidden=HIDDEN, layers=LAYERS, act='tanh'):
        super().__init__()
        acts = {'tanh': nn.Tanh(), 'gelu': nn.GELU(), 'relu': nn.ReLU()}
        activation = acts.get(act, nn.Tanh())
        net = []
        net.append(nn.Linear(in_dim, hidden))
        net.append(activation)
        for _ in range(layers - 2):
            net.append(nn.Linear(hidden, hidden))
            net.append(activation)
        net.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*net)
    def forward(self, x):
        return self.net(x)

def pde_residual(model, X):
    X.requires_grad_(True)
    V = model(X)
    ones = torch.ones_like(V)

    tau   = X[:, 0:1]
    F     = X[:, 1:2]
    A     = X[:, 2:3]
    K     = X[:, 3:4]
    r     = X[:, 4:5]
    sigma = X[:, 5:6]
    T     = X[:, 6:7]

    dV = torch.autograd.grad(V, X, grad_outputs=ones,
                             create_graph=True, retain_graph=True)[0]
    V_tau = dV[:, 0:1]
    V_F   = dV[:, 1:2]
    V_A   = dV[:, 2:3]

    V_F_grad = torch.autograd.grad(V_F, X, grad_outputs=torch.ones_like(V_F),
                                   create_graph=True, retain_graph=True)[0]
    V_FF = V_F_grad[:, 1:2]

    eps = 1e-6
    c_tau = 1.0 / (T * (1.0 - torch.clamp(tau, max=1.0 - eps)))
    c_tau = torch.clamp(c_tau, max=200.0)

    R = V_tau + 0.5 * (sigma**2) * (F**2) * V_FF + r * F * V_F \
        + c_tau * (F - A) * V_A - r * V
    return R

def minibatch(X, batch):
    n = X.shape[0]
    if batch >= n:
        idx = torch.arange(n, device=DEVICE)
    else:
        idx = torch.randint(0, n, (batch,), device=DEVICE)
    return X[idx], idx




