# Modèle 3 — colonne radiative locale

Le modèle 3 calcule le bilan radiatif d'une colonne atmosphérique locale sur la
grille mondiale de 5°. Il lit un paquet compact préparé à partir de données
ERA5, tient compte des opacités CO₂ et H₂O, et fournit les flux utilisés par
le modèle 4. Il ne fait pas évoluer lui-même la température de surface.

## Lancer

Depuis le dossier `modele3/` :

```
# Colonne locale avec les paramètres par défaut
./modele3.py

# Équivalent si le dossier courant est modele3/
python .
```

Les options restent disponibles si besoin :

```bash
./modele3.py --lat 0 --lon 0 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw
```

Les flux ERA5 affiches en validation sont mensuels. Si le court-onde principal
est lance en mode instantane, le bloc de validation compare donc un diagnostic
mensuel equivalent du modele, pas le flux instantane affiche dans `flux_W_m2`.
Pour rendre le flux court-onde principal directement comparable a ERA5, lancer
avec `--moyenne-journaliere-sw` et `--mois` sans `--jour-annee`.

Depuis la racine du dépôt :

```bash

# Colonne locale, moyenne journalière du court-onde
python -m modele3 --lat 0 --lon 0 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw

# Tests
python modele3/tests/tester_modele3.py
```

Pour régénérer le paquet compact lorsque les sources sont disponibles :

```bash
python -m modele3.ressources.generer_donnees --overwrite
```

## Visualisation de l'absorption H₂O

Le script suivant affiche un spectre indicatif de la vapeur d'eau entre `0,1`
et `30 µm` :

```bash
python modele3/codes_python/Absorbance_H2O.py
```

Il construit des bandes paramétrées et un continuum infrarouge, puis applique
Beer-Lambert. C'est une visualisation qualitative : elle ne lit pas RADIS,
ne calcule pas de raies HITRAN et ne modifie pas les coefficients utilisés par
la colonne radiative. Le calibrage effectif des coefficients H₂O est expliqué
dans [`documentation/CALIBRAGE_H2O.md`](documentation/CALIBRAGE_H2O.md).

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Chargement du paquet, calcul radiatif de colonne et calibrages CO₂/H₂O. |
| `ressources/` | Générateur du paquet compact et données générées. |
| `documentation/` | Théorie, provenance, calibrage CO₂ et notes de recherche. |
| `tests/` | Vérifications numériques automatisables. |
| `sorties/` | Images et autres sorties de diagnostic du modèle. |

Les hypothèses, limites et sources sont détaillées dans
[documentation/](documentation/README.md).
