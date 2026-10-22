# -*- coding: utf-8 -*-
"""Les methodes d'approximation du P&L, chacune autonome.

Convention commune, au pivot spot S0 (book vieilli Vh) :
  P&L(scenario) = theta + [approximation de Vh(S) - Vh(S0)]
  theta = Vh(S0) - V0          (thetadeterministe, 2 appels partages)
  dS    = S - S0 = S0*(exp(x)-1)   (choc de spot EXACT, jamais linearise)

Chaque methode renvoie un dict : pnl (N,), n_calls (build), t_build, t_online.
Le deroule sur les N scenarios est de l'algebre pure : 0 appel au pricer.
"""
import numpy as np

from monitoring import chrono
from cheb import cheb_nodes, fit_tensor, eval_tensor

EPS = 0.10   # pas de differences finies (bump relatif ~10 %)


# ------------------------------------------------------------------ DF spot
def _grad(V, x0, dirs, steps):
    """Gradient par differences centrees le long de `dirs` (pas `steps`)."""
    g = np.empty(len(dirs))
    for j, (v, h) in enumerate(zip(dirs, steps)):
        g[j] = (V(x0 + h * v) - V(x0 - h * v)) / (2.0 * h)
    return g, 2 * len(dirs)


def _hess(V, x0, dirs, steps):
    """Hessienne par double difference croisee (triangle sup. + diagonale)."""
    m = len(dirs)
    H = np.zeros((m, m))
    calls = 0
    for j in range(m):
        vj = steps[j] * dirs[j]
        for k in range(j, m):
            vk = steps[k] * dirs[k]
            H[j, k] = H[k, j] = (
                V(x0 + vj + vk) - V(x0 + vj - vk)
                - V(x0 - vj + vk) + V(x0 - vj - vk)
            ) / (4.0 * steps[j] * steps[k])
            calls += 4
    return H, calls


# ---------------------------------------------------------------- 1. Delta
def delta(Vh, S0, theta, dS, eps=EPS):
    """VaR-Delta, base normale : P&L = theta + delta . dS."""
    dirs = list(np.eye(len(S0)))
    steps = eps * S0
    t = {}
    with chrono(t, "b"):
        g, nc = _grad(Vh, S0, dirs, steps)
    with chrono(t, "o"):
        pnl = theta + dS @ g
    return dict(pnl=pnl, n_calls=nc, t_build=t["b"], t_online=t["o"])


# ---------------------------------------------------- 2. Delta-Gamma normale
def delta_gamma(Vh, S0, theta, dS, eps=EPS):
    """VaR-Delta-Gamma, base normale (hessienne pleine) :
    P&L = theta + delta . dS + 1/2 dS^T Gamma dS."""
    dirs = list(np.eye(len(S0)))
    steps = eps * S0
    t = {}
    with chrono(t, "b"):
        g, c1 = _grad(Vh, S0, dirs, steps)
        H, c2 = _hess(Vh, S0, dirs, steps)
    with chrono(t, "o"):
        pnl = theta + dS @ g + 0.5 * np.einsum("si,ij,sj->s", dS, H, dS)
    return dict(pnl=pnl, n_calls=c1 + c2, t_build=t["b"], t_online=t["o"])


# ------------------------------------------------------- 3. Delta-Gamma ACP
def gamma_acp(Vh, S0, theta, dS, lam, W, ks, eps=EPS):
    """VaR-Delta-Gamma en base ACP (spots), tronquee, une ligne par k.

    On bump les spots LE LONG des axes propres (pas eps*sqrt(lam_j)) : grecs
    par composante principale g_j, H_jl. Scores z_j = W_j^T dS. Le bloc K x K
    (K = max(ks)) est construit UNE fois ; chaque k<=K est une troncature
    gratuite. n_calls affiche = cout STANDALONE du bloc k : 2k + 2k(k+1).
    A k = d, coincide avec le Delta-Gamma normal (rotation orthonormale).
    """
    K = max(ks)
    steps = eps * np.sqrt(lam[:K])
    Z = dS @ W[:, :K]                     # scores (N, K)
    g = np.zeros(K)
    H = np.zeros((K, K))
    ksset = set(ks)
    rows, t_build, tb = [], 0.0, {}
    for j in range(K):                    # ajoute une PC -> bloc (j+1)x(j+1)
        vj = steps[j] * W[:, j]
        with chrono(tb, "inc"):
            g[j] = (Vh(S0 + vj) - Vh(S0 - vj)) / (2.0 * steps[j])
            for l in range(j + 1):        # nouvelle ligne/colonne du bloc
                vl = steps[l] * W[:, l]
                H[j, l] = H[l, j] = (
                    Vh(S0 + vj + vl) - Vh(S0 + vj - vl)
                    - Vh(S0 - vj + vl) + Vh(S0 - vj - vl)
                ) / (4.0 * steps[j] * steps[l])
        t_build += tb["inc"]
        k = j + 1
        if k in ksset:
            to = {}
            with chrono(to, "o"):
                z = Z[:, :k]
                pnl = theta + z @ g[:k] + 0.5 * np.einsum("si,ij,sj->s", z, H[:k, :k], z)
            rows.append(dict(k=k, pnl=pnl, n_calls=2 * k + 2 * k * (k + 1),
                             t_build=t_build, t_online=to["o"]))
    return rows


# ---------------------------------------------------------- 4. Sliding ACP
def _slide_1d(Vh, S0, W, j, a, b, ncb):
    nodes = cheb_nodes(a, b, ncb)
    pts = S0[None, :] + np.outer(nodes, W[:, j])
    pts = np.maximum(pts, 1e-8)           # garde-fou : jamais de spot <= 0
    vals = np.array([Vh(p) for p in pts])
    return fit_tensor(vals, [nodes], [(a, b)]), ncb


def _slide_tensor(Vh, S0, W, dims, bounds, ncb):
    nodes = [cheb_nodes(*bounds[i], ncb) for i in dims]
    grids = np.meshgrid(*nodes, indexing="ij")
    flat = np.stack([gg.ravel() for gg in grids], axis=1)     # (ncb^m, m)
    pts = np.maximum(S0[None, :] + flat @ W[:, dims].T, 1e-8)  # (ncb^m, d)
    vals = np.array([Vh(p) for p in pts]).reshape(grids[0].shape)
    return fit_tensor(vals, nodes, [bounds[i] for i in dims]), flat.shape[0]


def slider(Vh, S0, theta, Vh0, dS, W, ks, m_head, ncb=7):
    """Sliding ACP (spots), config {m_head, 1, ..., 1}, une ligne par k.

    Bloc de tete = tenseur de Chebyshev m_head-D sur PC1..PC_{m_head} (leurs
    cross-gammas exacts) ; queue = slides additives 1-D sur PC_{m_head+1..k}.
    m_head = 1 -> slider additif {1,...,1}. Base = ACP des SPOTS ; scores raw
    z_j = W_j^T dS, bornes = [min, max] des scores (par PC).

    Balayage k en UNE passe : la tete est batie une fois, la queue s'allonge
    d'une PC quand k croit (aucune reconstruction). n_calls = cout standalone
    de la config a ce k : 1 + ncb^m_head + (k - m_head) * ncb.
    """
    Z = dS @ W
    bounds = [(Z[:, j].min(), Z[:, j].max()) for j in range(W.shape[1])]
    ks = [k for k in ks if k >= m_head]
    if not ks:
        return []

    tb = {}
    with chrono(tb, "head"):
        head, _ = _slide_tensor(Vh, S0, W, list(range(m_head)), bounds, ncb)
    t_build = tb["head"]

    rows = []
    tail_fits = []
    j_next = m_head
    for k in ks:
        with chrono(tb, "tail"):
            while j_next < k:                          # allonge la queue
                cf, _ = _slide_1d(Vh, S0, W, j_next, *bounds[j_next], ncb)
                tail_fits.append((j_next, cf))
                j_next += 1
        t_build += tb["tail"]
        to = {}
        with chrono(to, "o"):
            approx = eval_tensor(head, Z[:, :m_head], bounds[:m_head])
            for j, cf in tail_fits:
                approx = approx + eval_tensor(cf, Z[:, [j]], [bounds[j]]) - Vh0
            pnl = theta + (approx - Vh0)
        rows.append(dict(k=k, pnl=pnl,
                         n_calls=1 + ncb ** m_head + (k - m_head) * ncb,
                         t_build=t_build, t_online=to["o"]))
    return rows
