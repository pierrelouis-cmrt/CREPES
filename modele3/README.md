# Modèle 3 — colonne radiative locale

Le modèle 3 calcule le bilan radiatif d'une colonne atmosphérique locale sur la
grille mondiale de 5°. Il lit un paquet compact préparé à partir de données
ERA5, tient compte des opacités CO₂ et H₂O, et fournit les flux utilisés par
le modèle 4. Il ne fait pas évoluer lui-même la température de surface.

## Lancer

Depuis le dossier `modele3/` :

```bash
python -m pip install -r requirements.txt

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
python -m pip install -r modele3/requirements.txt

# Colonne locale, moyenne journalière du court-onde
python -m modele3 --lat 0 --lon 0 --mois 7 --temperature-surface 293 --moyenne-journaliere-sw

# Tests
python modele3/tests/tester_modele3.py
```

Pour régénérer le paquet compact lorsque les sources sont disponibles :

```bash
python -m modele3.ressources.generer_donnees --overwrite
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Chargement du paquet, calcul radiatif de colonne et calibrages CO₂/H₂O. |
| `ressources/` | Générateur du paquet compact et données générées. |
| `documentation/` | Théorie, provenance, calibrage CO₂ et notes de recherche. |
| `tests/` | Vérifications numériques automatisables. |
| `codes_python/calibrer_coefficients_co2.py` | Outil de calibrage CO₂ dédié. |
| `codes_python/calibrer_coefficients_h2o.py` | Outil de calibrage H₂O dédié. |
| `documentation/CALIBRAGE_CO2.md` | Méthode de calibrage CO₂ actuelle. |
| `documentation/CALIBRAGE_H2O.md` | Méthode de calibrage H₂O proposée. |
| `ressources/calibrage_opacite_h2o/coefficients_h2o_modele3.json` | Coefficients H₂O chargés par le runtime. |
| `requirements.txt` | Dépendances du moteur. |
| `requirements-calibrage.txt` | Dépendances supplémentaires pour le calibrage. |

Les hypothèses, limites et sources sont détaillées dans
[documentation/](documentation/README.md).
