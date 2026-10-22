# -*- coding: utf-8 -*-
"""Chocs de risque et ACP.

Facteurs = log-chocs x ~ N(0, dt * D R D), D = diag(sigma). Scenario de spot
S = S0 * exp(x). L'ACP demandee (Gamma ACP et Sliding ACP) porte sur les
SPOTS : on diagonalise cov(S) empirique. Les scores d'un scenario sont les
projections z_j = W_j^T (S - S0) le long des vecteurs propres.
"""
import numpy as np


def log_shocks(sigma, R, dt, N, seed):
    """N log-chocs (N, n) ~ N(0, dt * diag(sigma) R diag(sigma))."""
    D = np.diag(np.asarray(sigma, dtype=float))
    Sigma = dt * D @ np.asarray(R, dtype=float) @ D
    L = np.linalg.cholesky(Sigma)
    z = np.random.default_rng(seed).standard_normal((N, len(sigma)))
    return z @ L.T


def spot_scenarios(S0, x):
    """(N, n) scenarios de spot S = S0 * exp(x)."""
    return np.asarray(S0, dtype=float) * np.exp(x)


def pca_spots(S):
    """ACP empirique des scenarios de spot : cov(S) = W diag(lam) W^T, desc.

    Renvoie (lam, W) tries par variance decroissante. W colonnes = axes propres
    (orthonormes) ; lam = variances le long de ces axes.
    """
    C = np.cov(S, rowvar=False)
    lam, W = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1]
    return lam[o], W[:, o]
