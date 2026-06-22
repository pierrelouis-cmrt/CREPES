# Modèle 3

Colonne radiative finale, fondée sur un paquet global précalculé. Elle fournit les flux radiatifs au modèle 4.

## Installation

```bash
python -m pip install -r modele3/requirements.txt
```

## Commandes

Depuis la racine du dépôt :

```bash
# Régénérer le paquet compact
python -m modele3.ressources.generer_donnees --overwrite

# Calculer une colonne
python -m modele3.modele3 --lat 0 --lon 0 --mois 7 --temperature-surface 293.0 --moyenne-journaliere-sw

# Lancer les tests
python modele3/tests/tester_modele3.py
```

## Structure

| Élément | Rôle |
| --- | --- |
| `modele3.py` | Calcul radiatif d’une colonne. |
| `physique.py` | Constantes et formules physiques. |
| `donnees.py` | Chargement du paquet compact. |
| `ressources/generer_donnees.py` | Génération du paquet global. |
| `ressources/donnees_precalculees/` | Paquet global versionné. |
| `tests/tester_modele3.py` | Tests numériques. |
| `THEORIE.md` | Méthodes radiatives, données et limites. |
