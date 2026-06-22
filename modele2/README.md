# Modèle 2

Colonne atmosphérique radiative CO₂ à six couches. Le modèle calcule les opacités, transmissions, émissivités et les flux infrarouges montant et descendant.

## Installation

```bash
python -m pip install -r modele2/requirements.txt
```

## Lancer le modèle

Depuis la racine du dépôt :

```bash
python modele2/modele2.py
```

Le script affiche les caractéristiques des couches, les opacités par bande, le flux infrarouge sortant au sommet et le flux infrarouge descendant à la surface.

## Générer le profil vertical

```bash
python modele2/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/ressources/profil_vertical_atmosphere_co2.png --csv modele2/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

Options principales :

| Option | Rôle |
| --- | --- |
| `--max-altitude-km` | Altitude maximale du profil. |
| `--step-m` | Pas vertical, en mètres. |
| `--surface-co2-ppm` | Concentration de CO₂ à la surface. |
| `--co2-gradient-ppm-per-km` | Gradient vertical de CO₂ ; `0` correspond à un gaz bien mélangé. |
| `--surface-pressure-pa` | Pression de surface, en pascals. |
| `--surface-temperature-k` | Température de surface, en kelvins. |
| `--output`, `--csv` | Chemins du graphique et du CSV générés. |
| `--no-plot` | Génère les fichiers sans ouvrir de fenêtre. |

## Structure

| Élément | Rôle |
| --- | --- |
| `modele2.py` | Code exécutable du noyau radiatif à six couches. |
| `ressources/profil_vertical_atmosphere_co2.py` | Génère le profil vertical de référence. |
| `ressources/profil_vertical_atmosphere_co2.csv` | Export numérique du profil. |
| `ressources/profil_vertical_atmosphere_co2.png` | Graphique de diagnostic du profil. |
| `requirements.txt` | Dépendances Python. |
| `THEORIE.md` | Hypothèses, équations, paramètres, validations et limites. |
