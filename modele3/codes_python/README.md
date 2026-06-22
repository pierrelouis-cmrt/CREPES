# Codes Python — modèle 3

Ce dossier fournit le calcul d'une colonne radiative locale et les modules qui
chargent le paquet compact ou regroupent les formules physiques.

## Lancer une colonne

Depuis la racine du dépôt :

```bash
python -m modele3.codes_python.modele3 --lat 48.5 --lon 2.3 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw
```

Utiliser `--jour-annee` à la place de `--mois` pour un jour précis,
`--heure-solaire` pour le court-onde instantané, `--co2` pour changer la
concentration et `--json` pour obtenir la sortie complète structurée.

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele3.py` | Interface en ligne de commande et calcul de colonne radiative. |
| `donnees.py` | Chargement du paquet et extraction de la colonne demandée. |
| `physique.py` | Constantes, bandes et formules radiatives partagées. |
| `calibrer_coefficients_co2.py` | Calibrage HITRAN/RADIS des coefficients CO₂ effectifs. |

Pour calibrer les coefficients CO₂ :

```bash
python -m modele3.codes_python.calibrer_coefficients_co2 --dry-run
```

La préparation des données est décrite dans le
[README des ressources](../ressources/README.md).
