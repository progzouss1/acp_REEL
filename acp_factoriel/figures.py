# -*- coding: utf-8 -*-
"""Figures de l'experience 3 (rapport).

  acp_blocs_correlation.png      matrice par blocs, spectre, vecteurs propres
  acp_factoriel_correlation.png  idem pour la correlation factorielle (market.py)
  m_acp_blocs_d25_fr.png         ecarts VaR/ES au full repricing en fonction de k
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from market import build, block_correlation
from shocks import log_shocks, spot_scenarios, pca_spots

BLUE, LIGHT = "#1f4e79", "#9bb7d4"
PC_COLORS = ["#1f4e79", "#c0504d", "#4f9a4f", "#e08a1e", "#7a4f9a"]


def _spectre_panels(fig, ax, R, lam, title_a):
    eR = np.sort(np.linalg.eigvalsh(R))[::-1]
    pR, pS = 100 * eR / eR.sum(), 100 * lam / lam.sum()
    im = ax[0].imshow(R, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[0].set_title(title_a); ax[0].set_xlabel("actif"); ax[0].set_ylabel("actif")
    x = np.arange(1, 11)
    ax[1].bar(x - 0.2, pR[:10], 0.4, label="valeurs propres de R", color=LIGHT)
    ax[1].bar(x + 0.2, pS[:10], 0.4, label="ACP des spots simulés", color=BLUE)
    ax[1].set_xticks(x); ax[1].set_xlabel("composante")
    ax[1].set_ylabel("part de variance (%)")
    ax[1].set_title("(b) Spectre : R vs covariance des spots"); ax[1].legend(fontsize=9)


def fig_correlation_blocs(path="acp_blocs_correlation.png"):
    cfg, market, pf = build()
    S0, sig = market.spots, market.vols
    R = block_correlation(25)
    S = spot_scenarios(S0, log_shocks(sig, R, 1 / 252, 10_000, 0))
    lam, W = pca_spots(S)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                           gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    _spectre_panels(fig, ax, R, lam, "(a) Matrice R : 5 secteurs de 5")
    for j, c in enumerate(PC_COLORS):
        ax[2].plot(np.arange(25), W[:, j], marker="o", ms=3, color=c, label="PC%d" % (j + 1))
    for b in range(1, 5):
        ax[2].axvline(5 * b - 0.5, color="grey", lw=0.6, ls=":")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("actif")
    ax[2].set_title("(c) Vecteurs propres des spots, PC1–PC5", pad=22)
    ax[2].legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_correlation_factorielle(path="acp_factoriel_correlation.png"):
    from market import factor_correlation, _poly_shapes
    cfg, market, pf = build()
    S0, sig = market.spots, market.vols
    R = factor_correlation(25)
    Phi = _poly_shapes(25, 3)
    for k in range(3):           # orientation : niveau > 0, pente croissante, bords > 0
        s = (np.sign(Phi[:, k].sum()) if k == 0 else
             np.sign(Phi[-1, k] - Phi[0, k]) if k == 1 else np.sign(Phi[0, k]))
        Phi[:, k] *= s
    S = spot_scenarios(S0, log_shocks(sig, R, 1 / 252, 10_000, 0))
    lam, W = pca_spots(S)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                           gridspec_kw=dict(width_ratios=[1, 1.3, 1.3]))
    _spectre_panels(fig, ax, R, lam, "(a) Matrice R factorielle")
    for j in range(3):
        w = W[:, j] * np.sign(W[:, j] @ Phi[:, j])
        ax[2].plot(np.arange(25), Phi[:, j], ls="--", color=PC_COLORS[j], lw=1.2, alpha=0.8)
        ax[2].plot(np.arange(25), w, marker="o", ms=3, color=PC_COLORS[j], label="PC%d" % (j + 1))
    ax[2].plot([], [], ls="--", color="grey", label="forme $\\varphi_k$")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("actif")
    ax[2].set_title("(c) Vecteurs propres des spots vs formes, PC1–PC3", pad=22)
    ax[2].legend(fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_resultats(csv, path, k_eff):
    """Ecarts VaR / ES au full repricing en fonction de k (trait mixte : k_eff)."""
    df = pd.read_csv(csv)
    sty = {"Gamma-ACP": ("^", "#1f4e79"), "Slider {1..}": ("D", "#4f9a4f"),
           "Slider {2,1..}": ("v", "#7a4f9a"), "Slider {3,1..}": ("P", "#a0522d")}
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for i, h in enumerate(["1j", "10j"]):
        s = df[df.horizon == h]
        for j, (col, lab) in enumerate([("dVaR_pct", "VaR 99 %"), ("dES_pct", "ES 97,5 %")]):
            a = ax[i, j]
            a.axhspan(-1, 1, color="0.9", zorder=0); a.axhline(0, color="k", lw=0.7)
            for m, (mk, c) in sty.items():
                t = s[s.methode == m].sort_values("k")
                a.plot(t.k, t[col], marker=mk, color=c, ms=5, label=m)
            for m, ls, c in [("Delta", ":", "#c0504d"), ("Delta-Gamma", "--", "#e08a1e")]:
                a.axhline(s[s.methode == m][col].iloc[0], ls=ls, color=c, lw=1.4,
                          label=m + " (base naturelle)")
            a.axvline(k_eff, color="grey", lw=0.8, ls="-.")
            a.set_title("%s — horizon %s" % (lab, "1 j" if h == "1j" else "10 j"))
            a.set_ylabel("écart au full repricing (%)"); a.grid(alpha=0.3)
    ax[1, 0].set_xlabel("nombre k de composantes"); ax[1, 1].set_xlabel("nombre k de composantes")
    h_, l_ = ax[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path, dpi=200); plt.close(fig)


def fig_qq(npz, path, k_eff):
    """QQ-plots P&L proxy vs full repricing, gamma-ACP et slider {1..} a k_eff."""
    z = np.load(npz)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, h in zip(ax, ["1j", "10j"]):
        ref = np.sort(z["%s|Full repricing|" % h])
        for meth, mk, c in [("Gamma-ACP", "^", "#1f4e79"), ("Slider {1..}", "D", "#4f9a4f")]:
            p = np.sort(z["%s|%s|%d" % (h, meth, k_eff)])
            a.plot(ref, p, mk, ms=2.5, color=c, alpha=0.6, label="%s, k=%d" % (meth, k_eff))
        a.plot([ref[0], ref[-1]], [ref[0], ref[-1]], "k--", lw=0.8)
        a.axvline(np.quantile(ref, 0.01), color="grey", ls=":", lw=0.8)
        a.set_title("QQ-plot P&L, horizon %s (pointillé : quantile 1 %%)" % ("1 j" if h == "1j" else "10 j"))
        a.set_xlabel("quantiles full repricing"); a.set_ylabel("quantiles proxy")
        a.grid(alpha=0.3); a.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
