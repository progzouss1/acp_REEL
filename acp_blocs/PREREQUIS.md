# Prérequis pour exécuter le projet

## Logiciels

| élément | version | remarque |
|---|---|---|
| Python | **3.8 ou plus récent** (testé en 3.8.20) | |
| numpy | ≥ 1.17 (testé 1.24.4) | `default_rng`, `SeedSequence` |
| pandas | ≥ 1.1 (testé 2.0.3) | |
| matplotlib | ≥ 3.3 (testé 3.7.5) | |
| Jupyter | `notebook` ou `jupyterlab`, avec `ipykernel` | pour ouvrir le notebook |

Avec Python 3.8, les dernières versions disponibles sont numpy 1.24, pandas 2.0
et matplotlib 3.7 : `pip` choisit automatiquement les versions compatibles avec
la version de Python installée.

Aucune autre bibliothèque n'est nécessaire, et aucun accès réseau pendant le calcul.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate      Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook          # puis ouvrir le notebook du dossier
```

## Exécution

1. Ouvrir le notebook (`acp_factoriel.ipynb` ou `acp_blocs.ipynb`) **depuis ce
   dossier** : les modules `.py` doivent être dans le répertoire de travail.
2. Pour vérifier l'installation, mettre `RAPIDE = True` dans la cellule
   « Paramètres » et tout exécuter (~2 min, chiffres sans valeur).
3. Remettre `RAPIDE = False` et tout exécuter.

## Ressources

| étape | volume | durée indicative |
|---|---|---|
| full repricing (1 j + 10 j) | 20 000 valorisations à 10⁵ trajectoires | ~1,5 h CPU, divisée par le nombre de cœurs |
| proxys (2 horizons en parallèle) | ~3 600 valorisations par horizon | ~20 min |

- **Processeur** : le full repricing utilise tous les cœurs (`N_JOBS = None`),
  les proxys deux processus. Le résultat ne dépend pas du nombre de cœurs.
- **Mémoire** : ~1 Go par processus (paquets de 10⁵ × 25 trajectoires et de
  10⁶ × 25 pour V(S₀)).
- **Disque** : ~10 Mo (cache, CSV, P&L, figures).
- **Cache** : tout ce qui est long est enregistré, et une relance le relit au lieu
  de le recalculer :
  - `cache_fr/` : full repricing (`fr_d25_<corr>_<horizon>_….npz`) et bruit de
    référence (`sigma_ref_….json`) ;
  - `cache_proxys/` : résultats et P&L des proxys, un fichier par horizon.

  Une fois les caches remplis, le notebook s'exécute en quelques secondes.
  Supprimer un dossier force le recalcul de l'étape correspondante. Les fichiers
  de cache d'une version précédente du projet sont réutilisables : il suffit de
  les copier dans `cache_fr/`.

## Sorties (dans le dossier)

| fichier | contenu |
|---|---|
| `acp_<corr>_d25.csv` | toutes les configurations, 1 j et 10 j |
| `acp_<corr>_d25_pnl.npz` | P&L scénario par scénario de chaque méthode |
| `acp_<corr>_correlation.png` | matrice, spectre, vecteurs propres |
| `m_acp_<corr>_d25.png` | écarts VaR / ES au full repricing en fonction de k |
| `qq_acp_<corr>_d25.png` | QQ-plots proxy vs full repricing |

## Fichiers

- `acp_<corr>.ipynb` : **point d'entrée**, toute l'exécution se fait ici
- `acp.py` : scénarios, pricer à graine fixe, méthodes, métriques
- `fullreval.py` : full repricing de référence, parallélisé, avec cache
- `figures.py` : figures
- `market.py` : construction des deux matrices de corrélation et du book
- `methods.py`, `cheb.py`, `shocks.py`, `pricing.py`, `portfolio.py`, `model.py`,
  `config.py`, `monitoring.py` : modules du projet

## Sous Windows

Le full repricing et les proxys tournent dans des processus séparés
(`ProcessPoolExecutor`). Les fonctions de travail sont définies dans les modules
`.py`, ce qui est requis sous Windows ; il suffit de lancer le notebook depuis
ce dossier. En cas de problème, `N_JOBS = 1` force un calcul sur un seul cœur
pour le full repricing.
