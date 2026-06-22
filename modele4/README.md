# Modèle 4

Grille de température de surface couplée aux flux radiatifs du modèle 3. Les cellules sont indépendantes : il n’y a pas de transport horizontal.

## Installation

```bash
python -m pip install -r modele4/requirements.txt
```

## Commandes

Depuis la racine du dépôt :

```bash
# Diagnostic mensuel global
python -m modele4.modele4

# Simulation temporelle courte sur une cellule
python -m modele4.modele4 --mode temporel --jours 0.020833333333333332 --max-latitudes 1 --max-longitudes 1 --frequence-sortie-pas 1 --output /tmp/modele4_test.npz

# Moteur rapide vectorisé
python -m modele4.rapide

# Lanceur interactif
python -m modele4.lancer

# Tests
python modele4/tests/tester_modele4.py
python modele4/tests/tester_rapide.py
```

## Structure

| Élément | Rôle |
| --- | --- |
| `modele4.py` | Moteur de référence et interface en ligne de commande. |
| `rapide.py` | Moteur vectorisé pour les simulations courantes. |
| `surface.py` | Capacité thermique, flux latent et convection. |
| `lancer.py` | Lanceur interactif. |
| `tests/` | Tests du moteur classique et rapide. |
| `THEORIE.md` | Bilan de surface, méthodes et limites. |
| `plan.md` | Notes de planification. |
