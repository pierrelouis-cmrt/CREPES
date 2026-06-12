# Modèle 2 - Spectre d'absorbance du CO2

Ce modèle calcule le spectre d'absorbance infrarouge du CO2 avec RADIS et les
raies spectroscopiques HITRAN. La concentration, la pression, la température et
la longueur du trajet optique sont paramétrables.

## Installation

```bash
pip install -r modele2/requirements.txt
```

## Exemple

```bash
python modele2/spectre_absorbance_co2.py --co2-ppm 800 --pressure-bar 0.8 --output modele2/spectre_800ppm.png --csv modele2/spectre_800ppm.csv --no-plot
```

Options principales :

- `--co2-ppm` : concentration volumique en ppm ;
- `--pressure-bar` : pression totale en bar ;
- `--temperature-k` : température en kelvins ;
- `--path-length-m` : trajet optique en mètres ;
- `--output` : fichier image produit ;
- `--csv` : export des valeurs numériques.

L'absorbance est définie par `A = -ln(T)`, où `T` est la transmittance.
