# -*- coding: utf-8 -*-
"""Marché + book EXACT de config.py, avec choix de la corrélation.

corr="factoriel" : modèle niveau/pente/courbure calé sur un profil ~60/26/10
                   (spectre rapide, >95 % de variance sur 3 PC).
corr="blocs"     : corrélation sectorielle 0.9 / 0.2 (historique).

Le portefeuille (strikes, positions) est inchangé ; seule la corrélation change.
Pricing en Monte-Carlo noCRN (jamais de formule fermée).
"""
from dataclasses import replace

import numpy as np

from config import ExperimentConfig
from portfolio import build_market, build_portfolio
from pricing import GroupPricer

SECTOR_SIZE = 5


def block_correlation(n, bs=SECTOR_SIZE, rho_in=0.9, rho_out=0.2):
    blk = np.arange(n) // bs
    R = np.where(blk[:, None] == blk[None, :], rho_in, rho_out).astype(float)
    np.fill_diagonal(R, 1.0)
    return R


def _poly_shapes(n, K):
    """K formes orthonormées : niveau (plat), pente (linéaire), courbure..."""
    x = np.linspace(-1.0, 1.0, n)
    Q, _ = np.linalg.qr(np.vstack([x ** k for k in range(K)]).T)
    return Q


def factor_correlation(n, strengths=(8.75, 4.88, 3.75), idio=0.03):
    """Corrélation factorielle niveau/pente/courbure. Forces calées -> ~60/26/10.
    R = sum_k a_k phi_k phi_k^T + idio*I, renormalisée à diagonale 1 (donc DP)."""
    Phi = _poly_shapes(n, len(strengths))
    R = sum(a * np.outer(Phi[:, k], Phi[:, k]) for k, a in enumerate(strengths))
    R = R + idio * np.eye(n)
    d = np.sqrt(np.diag(R))
    return R / np.outer(d, d)


def build(corr="factoriel", cfg=None):
    """(cfg, market, portfolio). corr in {"factoriel", "blocs"}."""
    cfg = cfg or ExperimentConfig().validate()
    market = build_market(cfg.market)
    n = cfg.market.n_underlyings
    market.corr_matrix = (factor_correlation(n) if corr == "factoriel"
                          else block_correlation(n))
    pf = build_portfolio(cfg.portfolio, market)
    return cfg, market, pf


def group_pricers(pf, market, d, dt_age, M, seed):
    """(V0, Vh) callables spot(d)->prix, Monte-Carlo noCRN (paquets frais).
    V0 = valeur aujourd'hui ; Vh = book VIEILLI (maturités T_j - dt_age)."""
    grp = pf.group(d)
    aged = [replace(dv, maturity=dv.maturity - dt_age) for dv in grp]
    p0 = GroupPricer(grp,  market, M, "noCRN", seed)
    ph = GroupPricer(aged, market, M, "noCRN", seed + 1)
    return p0, ph
