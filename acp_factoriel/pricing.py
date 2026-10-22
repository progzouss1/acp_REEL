# -*- coding: utf-8 -*-
"""Les 5 fonctions cibles V_d et leur evaluation Monte-Carlo.

La cible du proxy TT du groupe d est la fonction DETERMINISTE

    V_d(x) = somme_j alpha_j * Pi_j(x),   x dans R^d,  j parcourt les 20 produits,

evaluee par Monte-Carlo : les 20 payoffs sont pris sur LES MEMES trajectoires
(le cout d'un noeud = UNE simulation, pas vingt), chaque terme portant son
propre exp(-r*T_j). Tous les produits etant europeens, seule compte la loi
marginale de S(T) : le meme paquet de normales sert a toutes les maturites.

Regimes de tirage au build (les prix aux noeuds de la sous-grille) :

  CRN, normals=None   "CRN-groupe" : un paquet G_d (M, d) tire a l'init du
                      pricer, reutilise a TOUS les appels. Bruits independants
                      d'un groupe a l'autre.
  CRN, normals=G      "CRN-global" : G est un paquet (M, 25) de normales DEJA
                      correlees N(0, C_25), tire UNE fois dans le notebook ;
                      le groupe d consomme G[:, :d]. Licite car le Cholesky   <-- J'AURAI JAMAIS REFLECHIS A CA
                      d'un bloc principal est le bloc principal du Cholesky     
                      (groupes emboites, equicorrelation) : G[:, :d] est
                      exactement N(0, C_d). Bruits COHERENTS entre groupes.
  noCRN               un paquet G_d frais A CHAQUE APPEL (le generateur avance
                      de noeud en noeud). Au sein d'un appel, le meme paquet
                      sert a toutes les maturites (comme dans les deux CRN).

La reference des erreurs (verite-terrain) est TOUJOURS un pricer noCRN a seed
dedie, independant du build, quel que soit le regime de build.
"""

import math
from typing import Dict # forcer l'import de Dict pour les annotations de type price_break_down, sinon erreur mypy

import numpy as np


def correlated_normals(rng, corr, n):
    """n tirages ~ N(0, corr), shape (n, d). Repli eigen si corr non-PSD."""
    d = corr.shape[0]
    z = rng.standard_normal((n, d))
    
    try: # ce repli n'est pas du tout nécessaire
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(corr)
        L = V @ np.diag(np.sqrt(np.clip(w, 0.0, None)))
    
    
    return z @ L.T


class GroupPricer(object):
    # Callable : C'est une fonction, une classe, une fermeture ou un objet qui peut être exécuté ou appelé
    """Callable x -> V_d(x), signature MoCaX (point, additional_data=None). 

    Parameters
    ----------
    derivatives : liste de Derivative, tous portant les memes underlying_ids.
    market      : MarketData complet ; le sous-marche est extrait via .sub.
    M_paths     : trajectoires par evaluation.
    regime      : "CRN" ou "noCRN".
    seed        : graine du generateur (ignoree si `normals` est fourni).
    normals     : ndarray (M_paths, >= d) de normales deja correlees N(0, C),
                  ou None. Fourni => variante CRN-global (exige regime="CRN").
    use_nominal : si True, chaque prix est pondere par derivative.nominal.
    """

    __slots__ = ("derivs", "sub", "r", "vol", "d", "M", "regime", "use_nominal",
                 "_G", "_rng", "_maturities", "_by_maturity", "_drift",
                 "_sqrt_t", "_disc", "n_calls")

    def __init__(self, derivatives, market, M_paths, regime, seed,
                 normals=None, use_nominal=True):
        self.derivs = list(derivatives)
        if not self.derivs: # si self.derivs est vide
            raise ValueError("Aucun derive fourni.")
        ids = self.derivs[0].underlying_ids
        for dv in self.derivs:
            if dv.underlying_ids != ids:
                raise ValueError("Groupe non homogene : produit %d porte %r "
                                 "au lieu de %r."
                                 % (dv.derivative_id, dv.underlying_ids, ids))

        self.sub = market.sub(ids) 
        self.r = float(self.sub.rate)
        self.vol = np.asarray(self.sub.vols, dtype=float)
        self.d = len(self.vol)
        self.M = int(M_paths)
        self.regime = regime
        self.use_nominal = bool(use_nominal)
        self.n_calls = 0

        # une exponentielle et une actualisation par maturite DISTINCTE
        self._maturities = sorted({float(dv.maturity) for dv in self.derivs})
        self._by_maturity = {T: [dv for dv in self.derivs
                                 if abs(float(dv.maturity) - T) < 1e-12]
                             for T in self._maturities} # on regroupe les produits par maturités communes
        self._drift = {T: (self.r - 0.5 * self.vol ** 2) * T
                       for T in self._maturities} # les drifts correspondent à chaque maturité T, pour chaque actif ayant la maturité correspondante, sous la mesure risque neutre
        self._sqrt_t = {T: math.sqrt(T) for T in self._maturities} # racines carrées des maturités 
        self._disc = {T: math.exp(-self.r * T) for T in self._maturities}

        if normals is not None:
            if regime != "CRN": # S'il y a des normales fournies, le régime doit être "CRN" pour que cela ait du sens
                raise ValueError("`normals` fourni exige regime='CRN' "
                                 "(variante CRN-global).")
            G = np.asarray(normals, dtype=float)
            if G.shape[0] != self.M or G.shape[1] < self.d:
                raise ValueError("normals attendu (%d, >=%d), recu %s."
                                 % (self.M, self.d, (G.shape,))) # String Formatting in C-Style
            self._G = G[:, :self.d]           # troncature de colonnes, licite
            self._rng = None
        elif regime == "CRN":                 # CRN-groupe : paquet propre fige
            rng = np.random.default_rng(seed)
            self._G = correlated_normals(rng, self.sub.corr_matrix, self.M)
            self._rng = None
        elif regime == "noCRN":               # paquet frais a chaque appel
            self._G = None
            self._rng = np.random.default_rng(seed)
        else:
            raise ValueError("regime inconnu : %r" % regime)

    # ------------------------------------------------------------- interne
    def _draw(self):
        if self._G is not None:
            return self._G
        return correlated_normals(self._rng, self.sub.corr_matrix, self.M)

    def _spot_vector(self, point):
        s0 = np.asarray([float(point[i]) for i in range(self.d)], dtype=float)
        if np.any(s0 <= 0):
            raise ValueError("Spot negatif ou nul dans le point evalue.")
        return s0

    # ------------------------------------------------------ interface MoCaX
    def __call__(self, point, additional_data=None):
        self.n_calls += 1
        s0 = self._spot_vector(point)
        G = self._draw()                      # (M, d), partage entre maturites
        total = 0.0
        for T in self._maturities:
            terminal = s0 * np.exp(self._drift[T] + self.vol * self._sqrt_t[T] * G)
            disc = self._disc[T]
            for dv in self._by_maturity[T]:
                w = dv.nominal if self.use_nominal else 1.0
                total += w * disc * float(dv.payoff(terminal).mean())
        return float(total)

    def price_many(self, points):
        """(n, d) -> (n,). Respecte le regime (paquet frais par point si noCRN)."""
        return np.array([self(p) for p in np.asarray(points, dtype=float)])

    def price_breakdown(self, point) -> Dict[int, float]:
        """{derivative_id: contribution ponderee}. Somme = self(point)."""
        s0 = self._spot_vector(point)
        G = self._draw()
        out = {}
        for T in self._maturities:
            terminal = s0 * np.exp(self._drift[T] + self.vol * self._sqrt_t[T] * G)
            disc = self._disc[T]
            for dv in self._by_maturity[T]:
                w = dv.nominal if self.use_nominal else 1.0
                out[dv.derivative_id] = w * disc * float(dv.payoff(terminal).mean())
        return out

    def __repr__(self):
        return ("GroupPricer(d=%d, n_products=%d, maturites=%s, M=%d, "
                "regime=%s, global_normals=%s)"
                % (self.d, len(self.derivs), self._maturities, self.M,
                   self.regime, self._rng is None and self.regime == "CRN"))


# =========================================================================
#  Valeur du portefeuille complet
# =========================================================================
def portfolio_value_exact(pf, market, M_paths, seed, dimensions=None) -> float:
    """V(s0) = somme des V_d aux spots initiaux, par MC direct noCRN.

    Reference pour valider l'assemblage des proxies (independante du build).
    """
    dims = sorted({dv.dimension for dv in pf}) if dimensions is None \
        else list(dimensions)
    total = 0.0
    for d in dims:
        pricer = GroupPricer(pf.group(d), market, M_paths, "noCRN", seed + d)
        total += pricer(market.spots[:d])
    return float(total)


def assemble_portfolio_value(evaluators, scenario) -> float:
    """V(s) ~ somme_d eval_d(s[:d]) : somme des EVALUATIONS des proxies.

    On n'additionne JAMAIS les tensor-trains eux-memes : chaque proxy est
    evalue sur son prefixe, puis les scalaires sont sommes.

    Le decoupage s[:d] est licite parce que la boite vol-scaled est un produit
    d'intervalles par actif : l'intervalle de l'actif i ne depend que de
    (S_i(0), sigma_i), jamais de d.
    """
    s = np.asarray(scenario, dtype=float)
    total = 0.0
    for d, ev in evaluators.items():
        v = ev(s[:d].reshape(1, -1))
        total += float(np.asarray(v).ravel()[0])
    return float(total)
