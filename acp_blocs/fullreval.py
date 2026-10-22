# -*- coding: utf-8 -*-
"""Full repricing de reference (book NON vieilli).

P&L_FR(s) = V(S_s) - V(S0), avec
  - V(S_s) : une valorisation Monte-Carlo par scenario, M_fr trajectoires,
             graine differente pour chaque scenario et independante de celle
             des proxys ;
  - V(S0)  : valorise UNE fois avec M_v0 trajectoires (gros budget : V(S0)
             decale tous les P&L d'un meme montant, son bruit biaiserait
             directement la VaR). Calcule par paquets de 1 M trajectoires.

Les N scenarios sont decoupes en blocs FIXES (taille `chunk`), chacun avec sa
graine (SeedSequence.spawn) : le resultat ne depend pas du nombre de processus.
Les fonctions de travail sont au niveau module : ProcessPoolExecutor
fonctionne aussi depuis un notebook sous Windows.

Cache disque : un .npz par (d, etiquette, M_fr, M_v0, graine, N).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pricing import GroupPricer

V0_BLOCK = 1_000_000


def _price_chunk(args):
    derivs, market, M_paths, seed_seq, S_chunk = args
    p = GroupPricer(derivs, market, M_paths, "noCRN", seed_seq)
    return np.array([p(s) for s in S_chunk])


def _price_v0(args):
    derivs, market, M_paths, seed_seq, d = args
    return GroupPricer(derivs, market, M_paths, "noCRN", seed_seq)(market.spots[:d])


def _cache_path(cache_dir, d, tag, M_fr, M_v0, seed, N):
    return os.path.join(cache_dir, "fr_d%d_%s_M%d_V%d_s%d_N%d.npz"
                        % (d, tag, M_fr, M_v0, seed, N))


def full_repricing(pf, market, d, S, tag, M_fr=100_000, M_v0=10_000_000,
                   seed=777, chunk=200, n_jobs=None, cache_dir="cache_fr"):
    """Dict : pnl (N,), V0, n_calls, t_build=0, t_online (temps mur), n_jobs, cached."""
    S = np.asarray(S, dtype=float)
    N = len(S)
    n_jobs = n_jobs or os.cpu_count() or 1
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, d, tag, M_fr, M_v0, seed, N)
        if os.path.exists(path):
            z = np.load(path)
            if np.allclose(z["S"], S):              # memes scenarios ?
                return dict(pnl=z["Vh"] - float(z["V0"]), V0=float(z["V0"]),
                            n_calls=N, t_build=0.0, t_online=float(z["t"]),
                            n_jobs=int(z["n_jobs"]), cached=True)

    grp = pf.group(d)
    ss_v0, ss_chunks = np.random.SeedSequence(seed).spawn(2)
    blocks = [S[i:i + chunk] for i in range(0, N, chunk)]
    seeds = ss_chunks.spawn(len(blocks))
    n_v0 = max(1, -(-M_v0 // V0_BLOCK))
    v0_jobs = [(grp, market, M_v0 // n_v0, sd, d) for sd in ss_v0.spawn(n_v0)]
    jobs = [(grp, market, M_fr, sd, b) for sd, b in zip(seeds, blocks)]

    ex = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    try:
        run = (lambda f, js: list(ex.map(f, js))) if ex else \
              (lambda f, js: [f(j) for j in js])
        t0 = time.perf_counter()
        V0 = float(np.mean(run(_price_v0, v0_jobs)))
        Vh = np.concatenate(run(_price_chunk, jobs))
        t = time.perf_counter() - t0
    finally:
        if ex:
            ex.shutdown()

    if path:
        np.savez_compressed(path, S=S, Vh=Vh, V0=V0, t=t, n_jobs=n_jobs)
    return dict(pnl=Vh - V0, V0=V0, n_calls=N, t_build=0.0, t_online=t,
                n_jobs=n_jobs, cached=False)


def reference_noise(pf, market, d, point, M_fr=100_000, n_rep=20, seed=999):
    """Ecart-type Monte-Carlo d'UNE valorisation de reference en `point`.

    C'est le plancher du RMSE proxy - full repricing : meme un proxy parfait ne
    peut pas descendre sous ce bruit.
    """
    p = GroupPricer(pf.group(d), market, M_fr, "noCRN", seed)
    v = np.array([p(point) for _ in range(n_rep)])
    return float(v.std(ddof=1))
