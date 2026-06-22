# Modèle 5

Version du modèle 4 avec des échanges radiatifs infrarouges horizontaux entre les colonnes atmosphériques voisines.

## Installation

```bash
python -m pip install -r modele5/requirements.txt
```

## Commandes

Depuis la racine du dépôt :

```bash
# Simulation globale par défaut
python -m modele5.modele5

# Petite grille de développement
python -m modele5.modele5 --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele5/sorties/simulation_dev.npz

# Désactiver l’échange horizontal, pour comparer au modèle 4 rapide
python -m modele5.modele5 --facteur-horizontal 0 --output modele5/sorties/simulation_sans_horizontal.npz

# Tests
python modele5/tests/tester_modele5.py

# Visualiser une sortie
python modele5/planisphere.py --fichier modele5/sorties/simulation_modele5.npz
```

## Options principales

| Option | Défaut | Rôle |
| --- | ---: | --- |
| `--jours` | `1` | Durée simulée, en jours. |
| `--dt` | `1800` | Pas interne, en secondes. |
| `--sortie-heures` | `4` | Fréquence de sauvegarde. |
| `--facteur-horizontal` | `1` | Intensité de l’échange horizontal ; `0` le désactive. |
| `--max-latitudes`, `--max-longitudes` | — | Sous-grille de développement. |
| `--facteur-latent`, `--convection`, `--vent` | — | Termes de surface hérités du modèle 4. |

## Structure

| Élément | Rôle |
| --- | --- |
| `modele5.py` | Moteur et interface en ligne de commande. |
| `planisphere.py` | Cartes des températures et du flux horizontal. |
| `tests/tester_modele5.py` | Tests numériques. |
| `THEORIE.md` | Bilan de surface, échanges horizontaux et limites. |
| `requirements.txt` | Dépendances Python. |
