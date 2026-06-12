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

## Profil vertical de l'atmosphère

Le script `profil_atmosphere_co2.py` calcule, en fonction de l'altitude, la
pression atmosphérique, le rapport de mélange du CO2 en ppm, sa pression
partielle et sa concentration en molécules par mètre cube.

```bash
python modele2/profil_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/profil_atmosphere_co2.png --csv modele2/profil_atmosphere_co2.csv --no-plot
```

Par défaut, le rapport de mélange reste constant à 420 ppm. Une variation
linéaire peut être testée avec `--co2-gradient-ppm-per-km`. Par exemple,
`--co2-gradient-ppm-per-km -0.2` retire 0,2 ppm par kilomètre.
