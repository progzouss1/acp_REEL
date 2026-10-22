# -*- coding: utf-8 -*-
"""Experience 3, d=25 : proxys ACP vs full repricing, book NON vieilli.

Correlation `corr` : "factoriel" (niveau / pente / courbure) ou "blocs"
(5 secteurs de 5, 0,9 / 0,2) -- matrices construites dans market.py ; la meme
matrice sert au pricer (Cholesky) et a la generation des chocs.

Book non vieilli : les maturites restent T_j a l'horizon, il n'y a pas de
theta et P&L = V(S) - V(S0) ; seuls les chocs dependent de l'horizon (1 j, 10 j).

Proxys : pricer unitaire a M=100 000 trajectoires, generateur reinitialise a la
MEME graine a chaque appel (regime R1 de l'experience 2). Grecs par differences
finies (methods.py), sliders de Tchebychev.

Reference : full repricing (fullreval.py), une valorisation independante par
scenario, M=100 000, V(S0) a 10^7 trajectoires, mise en cache.

VaR / ES exprimees en PERTE (positives) ; dVaR_pct > 0 : perte surestimee.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from market import build
from shocks import log_shocks, spot_scenarios, pca_spots
from pricing import GroupPricer
from fullreval import full_repricing
import methods as M
from concurrent.futures import ProcessPoolExecutor

D, NCB, SEED = 25, 7, 0
SEED_PRICER = SEED + D                # graine unique du pricer des proxys
HORIZONS = {"1j": 1 / 252, "10j": 10 / 252}
CORRELATIONS = ("blocs", "factoriel")
K_EFF = {"factoriel": 3, "blocs": 5}  # dimension effective attendue
KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
Q_VAR, Q_ES = 0.01, 0.025

# Parametres de l'experience du rapport ; les valeurs reduites servent au test rapide
PARAMS = dict(N=10_000, M_proxy=100_000, M_fr=100_000, M_v0=10_000_000)
PARAMS_RAPIDE = dict(N=300, M_proxy=5_000, M_fr=5_000, M_v0=50_000)


class SameSeedPricer(object):
    """Callable x -> V(x) : generateur reinitialise a la meme graine a chaque appel."""

    def __init__(self, derivs, market, M_paths, seed):
        self.p = GroupPricer(derivs, market, M_paths, "noCRN", seed)
        self.seed = seed
        self.n_calls = 0
        self.t = 0.0

    def __call__(self, x, additional_data=None):
        t0 = time.perf_counter()
        self.p._rng = np.random.default_rng(self.seed)
        v = self.p(x)
        self.t += time.perf_counter() - t0
        self.n_calls += 1
        return v


def var_loss(p, q=Q_VAR):
    """VaR en PERTE (positive) : -quantile_q du P&L."""
    return -float(np.quantile(p, q))


def es_loss(p, q=Q_ES):
    thr = np.quantile(p, q)
    return -float(p[p <= thr].mean())


def cache_tag(hname, corr):
    return "%s_%s" % (corr, hname)


def scenarios(hname, N, corr):
    """(market, pf, S0, S, dS, lam, W) pour un horizon : memes chocs partout."""
    if corr not in CORRELATIONS:
        raise ValueError("corr doit etre dans %r" % (CORRELATIONS,))
    cfg, market, pf = build(corr=corr)
    S0, SIG = market.spots[:D], market.vols[:D]
    R = market.corr_matrix[:D, :D]
    x = log_shocks(SIG, R, HORIZONS[hname], N, SEED)
    S = spot_scenarios(S0, x)
    lam, W = pca_spots(S)
    return market, pf, S0, S, S - S0, lam, W


def reference(hname, corr, N, M_fr, M_v0, cache_dir="cache_fr", n_jobs=None):
    """Full repricing de l'horizon (calcule une fois, puis relu du cache)."""
    market, pf, S0, S, _, _, _ = scenarios(hname, N, corr)
    return full_repricing(pf, market, D, S, tag=cache_tag(hname, corr),
                          M_fr=M_fr, M_v0=M_v0, n_jobs=n_jobs,
                          cache_dir=cache_dir)


def run_horizon(hname, corr, N, M_proxy, M_fr, M_v0, cache_dir="cache_fr"):
    """Toutes les methodes pour un horizon -> (DataFrame, dict des P&L)."""
    market, pf, S0, S, dS, lam, W = scenarios(hname, N, corr)
    p = SameSeedPricer(pf.group(D), market, M_proxy, SEED_PRICER)
    V0 = p(S0)
    theta = 0.0                           # book non vieilli

    fr = full_repricing(pf, market, D, S, tag=cache_tag(hname, corr), M_fr=M_fr,
                        M_v0=M_v0, n_jobs=1, cache_dir=cache_dir)
    ref = fr["pnl"]
    var_ref, es_ref = var_loss(ref), es_loss(ref)

    rows, pnls = [], {}

    def add(meth, r, k=None, conf="-"):
        v, e = var_loss(r["pnl"]), es_loss(r["pnl"])
        rows.append(dict(correlation=corr, horizon=hname, methode=meth, config=conf,
                         k=(k if k is not None else np.nan),
                         VaR99=v, ES975=e,
                         dVaR_pct=100 * (v - var_ref) / var_ref,
                         dES_pct=100 * (e - es_ref) / es_ref,
                         RMSE=float(np.sqrt(np.mean((r["pnl"] - ref) ** 2))),
                         appels=r["n_calls"], t_build=r["t_build"],
                         t_onl=r["t_online"], t_tot=r["t_build"] + r["t_online"]))
        pnls["%s|%s" % (meth, "" if k is None else k)] = r["pnl"]

    add("Full repricing", dict(pnl=ref, n_calls=N, t_build=0.0, t_online=fr["t_online"]))
    add("Delta", M.delta(p, S0, theta, dS))
    add("Delta-Gamma", M.delta_gamma(p, S0, theta, dS))
    for r in M.gamma_acp(p, S0, theta, dS, lam, W, KS):
        add("Gamma-ACP", r, k=r["k"])
    for m_head, name in [(1, "{1..}"), (2, "{2,1..}"), (3, "{3,1..}")]:
        for r in M.slider(p, S0, theta, V0, dS, W, KS, m_head, NCB):
            add("Slider " + name, r, k=r["k"], conf=name)

    t_call = p.t / p.n_calls                       # cout moyen d'une valorisation
    df = pd.DataFrame(rows)
    df["t_call"] = t_call
    df["t_FR_seq"] = N * t_call                    # full repricing equivalent 1 coeur
    df["speedup"] = df["t_FR_seq"] / df["t_tot"]
    df.loc[df.methode == "Full repricing", "speedup"] = 1.0
    return df, pnls


# ---------------------------------------------------------------- cache des proxys
def _proxy_key(hname, corr, params):
    """Empreinte de tout ce qui determine les resultats des proxys."""
    conf = dict(hname=hname, corr=corr, params=params, D=D, NCB=NCB, SEED=SEED,
                SEED_PRICER=SEED_PRICER, KS=KS, EPS=M.EPS, Q=(Q_VAR, Q_ES))
    h = hashlib.md5(json.dumps(conf, sort_keys=True).encode()).hexdigest()[:10]
    return "proxys_%s_%s_N%d_%s" % (corr, hname, params["N"], h)


def load_horizon(hname, corr, params, cache_dir="cache_proxys"):
    """(DataFrame, dict des P&L) relus du cache, ou None s'ils n'y sont pas."""
    base = os.path.join(cache_dir, _proxy_key(hname, corr, params))
    if not (os.path.exists(base + ".csv") and os.path.exists(base + ".npz")):
        return None
    z = np.load(base + ".npz")
    return pd.read_csv(base + ".csv"), {k: z[k] for k in z.files}


def save_horizon(hname, corr, params, df, pnls, cache_dir="cache_proxys"):
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, _proxy_key(hname, corr, params))
    df.to_csv(base + ".csv", index=False)
    np.savez_compressed(base + ".npz", **pnls)


def run_horizon_job(args):
    """Enveloppe pour ProcessPoolExecutor (doit etre au niveau module) :
    calcule un horizon et l'enregistre dans le cache des proxys."""
    hname, corr, params, cache_dir, proxy_cache = args
    df, pnls = run_horizon(hname, corr, cache_dir=cache_dir, **params)
    save_horizon(hname, corr, params, df, pnls, proxy_cache)
    return df, pnls


def run_all_horizons(corr, params, cache_dir="cache_fr", proxy_cache="cache_proxys"):
    """Proxys des deux horizons : relus du cache, sinon calcules en parallele.

    Renvoie (DataFrame, {horizon: dict des P&L}, liste des horizons calcules)."""
    res, todo = {}, []
    for h in HORIZONS:
        r = load_horizon(h, corr, params, proxy_cache)
        if r is None:
            todo.append(h)
        else:
            res[h] = r
    if todo:
        jobs = [(h, corr, params, cache_dir, proxy_cache) for h in todo]
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            for h, r in zip(todo, ex.map(run_horizon_job, jobs)):
                res[h] = r
    df = pd.concat([res[h][0] for h in HORIZONS], ignore_index=True)
    return df, {h: res[h][1] for h in HORIZONS}, todo


# --------------------------------------------------- cache du bruit de reference
def reference_noise_cached(corr, M_fr, n_rep=20, cache_dir="cache_fr"):
    """Ecart-type d'une valorisation de reference en S0 (relu du cache)."""
    from fullreval import reference_noise
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "sigma_ref_%s_M%d_n%d.json" % (corr, M_fr, n_rep))
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["sigma_ref"], True
    cfg, market, pf = build(corr=corr)
    s = reference_noise(pf, market, D, market.spots[:D], M_fr=M_fr, n_rep=n_rep)
    with open(path, "w") as f:
        json.dump({"sigma_ref": s}, f)
    return s, False
