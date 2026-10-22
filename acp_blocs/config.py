# -*- coding: utf-8 -*-
"""Configuration de l'experience (dataclasses). Un run = un ExperimentConfig.

Trois familles, volontairement separees :
  - DONNEES   : MarketConfig, PortfolioConfig      (le marche et les trades)
  - NUMERIQUE : MCConfig, TTConfig, BoxConfig      (comment on calcule)
  - ORCHESTR. : ExperimentConfig                   (ce qu'on fait tourner)

Le REGIME (CRN-groupe / CRN-global / noCRN) n'est PAS un champ de config :
c'est l'axe balaye par le notebook, une passe par variante.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple


# ------------------------------------------------------------------ marche
@dataclass
class MarketConfig:
    """Marche synthetique equicorrele.

    deterministic=True -> rampes regulieres (spot_i, vol_i croissants, sans seed).
    """
    n_underlyings: int = 25
    spot_range: Tuple[float, float] = (100.0, 120.0)
    vol_range: Tuple[float, float] = (0.30, 0.60)
    rho: float = 0.50
    rate: float = 0.01
    deterministic: bool = True
    seed: int = 20240101

    def validate(self):
        if self.n_underlyings < 2:
            raise ValueError("n_underlyings >= 2 requis.")
        if self.spot_range[0] <= 0 or self.spot_range[1] < self.spot_range[0]:
            raise ValueError("spot_range invalide : %r" % (self.spot_range,))
        if self.vol_range[0] <= 0 or self.vol_range[1] < self.vol_range[0]:
            raise ValueError("vol_range invalide : %r" % (self.vol_range,))
        lower = -1.0 / (self.n_underlyings - 1)
        if not (lower < self.rho < 1.0):
            raise ValueError("rho=%.4f hors de ]%.4f, 1[ (equicorrelation non PD)."
                             % (self.rho, lower))
        return self


# ------------------------------------------------------------- portefeuille
@dataclass
class ProductSpec:
    """kappa : moneyness (K = kappa * B_0) ; alpha : position (nominal)."""
    k: int
    kappa: float
    maturity: float
    callput_flag: int          # 0 = call, 1 = put
    alpha: float


DEFAULT_PRODUCT_GRID: List[ProductSpec] = [
    ProductSpec(k=1, kappa=0.90, maturity=1.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=2, kappa=0.95, maturity=2.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=3, kappa=1.00, maturity=3.0, callput_flag=0, alpha=+1.0),
    ProductSpec(k=4, kappa=1.05, maturity=4.0, callput_flag=1, alpha=-1.0),
    ProductSpec(k=5, kappa=1.10, maturity=5.0, callput_flag=1, alpha=-1.0),
]


@dataclass
class PortfolioConfig:
    """Groupes emboites G_d = {0..d-1} ; par groupe : types x grille = 4x5 = 20."""
    dimensions: Tuple[int, ...] = (5, 10, 15, 20, 25)
    option_types: Tuple[int, ...] = (0, 1, 2, 3)      # geo, basket, best-of, worst-of
    product_grid: List[ProductSpec] = field(
        default_factory=lambda: list(DEFAULT_PRODUCT_GRID))

    @property
    def n_products_per_group(self) -> int:
        return len(self.option_types) * len(self.product_grid)

    @property
    def n_products(self) -> int:
        return len(self.dimensions) * self.n_products_per_group

    def validate(self, n_underlyings: int):
        if not self.dimensions:
            raise ValueError("dimensions vide.")
        if max(self.dimensions) > n_underlyings:
            raise ValueError("dimension max %d > %d sous-jacents."
                             % (max(self.dimensions), n_underlyings))
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions dupliquees : %r" % (self.dimensions,))
        for s in self.product_grid:
            if s.kappa <= 0 or s.maturity <= 0 or s.callput_flag not in (0, 1):
                raise ValueError("ProductSpec k=%d invalide." % s.k)
        return self


# ------------------------------------------------------------- numerique
@dataclass
class BoxConfig:
    """Boite vol-scaled : [S0*exp(-z*sig*sqrt(h)), S0*exp(+z*sig*sqrt(h))].

    z_dom = 3.0 -> couverture marginale ~ 99.86% par actif (quantile 3 sigma).
    h_dom ne sert QU'A dimensionner la boite (le repricing reste a maturite).
    """
    h_dom: float = 1.0 / 252.0
    z_dom: float = 3.0

    def validate(self):
        if self.h_dom <= 0 or self.z_dom <= 0:
            raise ValueError("h_dom et z_dom doivent etre > 0.")
        return self


@dataclass
class MCConfig:
    """Budgets et graines Monte-Carlo (le regime est choisi par le notebook).

    m_gt    : trajectoires de la reference (verite-terrain des erreurs, noCRN).
              Ne sert QU'A mesurer les erreurs, jamais aux speed-ups.
    m_build : trajectoires aux noeuds d'entrainement du proxy (sous-grille)
    m_speed : trajectoires du MC "de production" concurrent du proxy ;
              c'est LUI que les speed-ups online/total comparent au proxy
    """
    m_gt: int = 1_000_000
    m_build: int = 200_000
    m_speed: int = 20_000
    seed_train: int = 12345    # noeuds (build) ; par groupe : seed_train + d
    seed_gt: int = 98765       # reference des erreurs
    seed_test: int = 2468      # points de test dans la boite
    seed_global: int = 424242  # paquet Z global (variante CRN-global)

    @property
    def budget_ratio(self) -> float:
        return float(self.m_gt) / float(self.m_build)

    def validate(self):
        if self.m_gt <= 0 or self.m_build <= 0 or self.m_speed <= 0:
            raise ValueError("budgets Monte-Carlo > 0 requis.")
        return self


DEFAULT_N_TRAIN: Dict[int, int] = {5: 300, 10: 1000, 15: 2000, 20: 3200, 25: 4500}


@dataclass
class TTConfig:
    """Proxy de Tchebychev au format tensor-train (MoCaXExtend)."""
    degree: int = 6                       # -> degree+1 points par dimension
    n_train: Dict[int, int] = field(default_factory=lambda: dict(DEFAULT_N_TRAIN))
    max_rank: int = 7
    tolerance: float = 1e-3
    rel_tolerance: float = 1e-8
    max_iters: int = 100
    max_rounds: int = 5
    train_frac: float = 0.8
    n_restarts: int = 3                   # relances du fit, selection sur val
    seed_fit: int = 0                     # np.random.seed avant chaque run ALS

    @property
    def points_per_dim(self) -> int:
        return self.degree + 1

    def full_grid_size(self, d: int) -> float:
        return float(self.points_per_dim) ** d

    def n_train_for(self, d: int) -> int:
        if d not in self.n_train:
            raise KeyError("n_train non defini pour d=%d." % d)
        return int(self.n_train[d])

    def algo_params(self, print_progress: bool = False) -> dict:
        return dict(tolerance=self.tolerance, rel_tolerance=self.rel_tolerance,
                    max_iters=self.max_iters, max_rank=self.max_rank,
                    max_rounds=self.max_rounds, print_progress=print_progress)

    def validate(self, dimensions):
        if self.degree < 1 or self.max_rank < 1 or self.n_restarts < 1:
            raise ValueError("degree, max_rank, n_restarts >= 1 requis.")
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError("train_frac dans ]0,1[ requis.")
        for d in dimensions:
            if self.n_train_for(d) < 1:
                raise ValueError("n_train[%d] >= 1 requis." % d)
        return self


# ---------------------------------------------------------- orchestration
@dataclass
class ExperimentConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    box: BoxConfig = field(default_factory=BoxConfig)
    mc: MCConfig = field(default_factory=MCConfig)
    tt: TTConfig = field(default_factory=TTConfig)

    n_test_points: int = 10_000       # points d'evaluation de l'erreur
    n_load: int = 100_000             # scenarios de la charge, pour spd_total

    def validate(self):
        self.market.validate()
        self.portfolio.validate(self.market.n_underlyings)
        self.box.validate()
        self.mc.validate()
        self.tt.validate(self.portfolio.dimensions)
        return self

    def summary(self) -> str:
        return "\n".join([
            "Marche       : %d sous-jacents, rho=%.2f, r=%.4f"
            % (self.market.n_underlyings, self.market.rho, self.market.rate),
            "Portefeuille : %d produits (%d groupes x %d types x %d moneyness)"
            % (self.portfolio.n_products, len(self.portfolio.dimensions),
               len(self.portfolio.option_types), len(self.portfolio.product_grid)),
            "Boite        : h=%.6f, z=%.1f sigma" % (self.box.h_dom, self.box.z_dom),
            "Monte-Carlo  : ref %s / build %s trajectoires"
            % ("{:,}".format(self.mc.m_gt), "{:,}".format(self.mc.m_build)),
            "Tensor-train : degre %d (%d pts/dim), max_rank=%d, tol=%.1e, "
            "restarts=%d" % (self.tt.degree, self.tt.points_per_dim,
                             self.tt.max_rank, self.tt.tolerance,
                             self.tt.n_restarts),
            "n_train      : %s" % {d: self.tt.n_train_for(d)
                                   for d in self.portfolio.dimensions},
        ])


# --------------------------------------------- helpers de variation
def replace_tt(cfg, **kw):
    return replace(cfg, tt=replace(cfg.tt, **kw))


def replace_mc(cfg, **kw):
    return replace(cfg, mc=replace(cfg.mc, **kw))


def replace_market(cfg, **kw):
    return replace(cfg, market=replace(cfg.market, **kw))