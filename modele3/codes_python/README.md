# Codes Python — modèle 3

Ce dossier fournit le calcul d'une colonne radiative locale et les modules qui
chargent le paquet compact ou regroupent les formules physiques.

## Lancer une colonne

Depuis le dossier `modele3/` :

```bash
./modele3.py --lat 48.5 --lon 2.3 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw
```

Depuis la racine du dépôt :

```bash
python -m modele3 --lat 48.5 --lon 2.3 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw
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
| `calibrer_coefficients_h2o.py` | Calibrage HITRAN/RADIS des coefficients H₂O effectifs. |
| `Absorbance_H2O.py` | Visualisation qualitative des bandes et du continuum d'absorption H₂O. |

Pour calibrer les coefficients CO₂ :

```bash
python -m modele3.codes_python.calibrer_coefficients_co2 --dry-run
```

Pour calibrer les coefficients H₂O :

```bash
python -m modele3.codes_python.calibrer_coefficients_h2o --dry-run
```

## Visualiser l'absorption H₂O

Depuis la racine du dépôt :

```bash
python modele3/codes_python/Absorbance_H2O.py
```

Le script ouvre un graphique ; il n'accepte pas d'option et n'écrit pas de
fichier seul. Les bandes sont des enveloppes paramétrées complétées par un
continuum au-delà de `12 µm`, puis converties en pourcentage d'absorption avec
la loi de Beer-Lambert. Il sert à illustrer le spectre et n'est pas appelé par
`modele3.py`.

La préparation des données est décrite dans le
[README des ressources](../ressources/README.md).
