# -*- coding: utf-8 -*-
"""Interpolation de Chebyshev TENSORIELLE (m dimensions), pour le slider.

fit_tensor : coefficients d'un tenseur de Chebyshev interpolant des valeurs
prises sur une grille de noeuds de Chebyshev (interpolation exacte : systeme
de Vandermonde carre resolu axe par axe).
eval_tensor : evaluation vectorisee du tenseur en des points quelconques.

Sert aux blocs {m,1,...} du slider : m = 1 (additif), 2 ou 3.
"""
import numpy as np
from numpy.polynomial.chebyshev import chebvander


def cheb_nodes(a, b, m):
    """m noeuds de Chebyshev sur [a, b]."""
    k = np.arange(m)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * k / (m - 1))


def _u(t, a, b):
    """Mappe [a, b] -> [-1, 1]."""
    return (2.0 * np.asarray(t, dtype=float) - (a + b)) / (b - a)


def fit_tensor(values, nodes_per_dim, bounds):
    """Coefficients du tenseur de Chebyshev interpolant `values` sur la grille.

    values         : array de forme (m_0, ..., m_{D-1}) (valeurs aux noeuds)
    nodes_per_dim  : liste des noeuds par dimension
    bounds         : liste de (a, b) par dimension
    """
    C = np.asarray(values, dtype=float)
    for ax, (nodes, (a, b)) in enumerate(zip(nodes_per_dim, bounds)):
        Vinv = np.linalg.inv(chebvander(_u(nodes, a, b), len(nodes) - 1))
        C = np.moveaxis(np.tensordot(Vinv, C, axes=([1], [ax])), 0, ax)
    return C


def eval_tensor(coef, pts, bounds):
    """Evalue le tenseur `coef` aux points `pts` (n, D). Renvoie (n,)."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    D = pts.shape[1]
    Ts = [chebvander(_u(pts[:, ax], *bounds[ax]), coef.shape[ax] - 1)
          for ax in range(D)]
    letters = "abcdefg"[:D]
    sub = letters + "," + ",".join("n" + l for l in letters) + "->n"
    return np.einsum(sub, coef, *Ts)
