# Absorption infrarouge du CO₂

Courbe de l'absorbance du CO₂ en fonction de la longueur d'onde, à 440 ppm sur 1 m.

---

# Absorbance du CO₂

## Source 

[Source bande absorbance CO2](https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2)

## Valeur en micrometre

| Bande (µm)    |
|---------------|
| 4,20 – 4,35   |
| 7,20 – 7,50   |
| 14,25 – 15,75 |


# Valeur pour le 1er modèle
Mesures prises à 440 ppm sur 1 m de colonne d'air, relevé manuellement sur le graphique :

| Bande (µm)        | Absorbance moyenne |
|-------------------|--------------------|
| 14,25 – 15,75     | 1                  |
| 4,20 – 4,35       | 3,25               |
| 7,20 – 7,50       | ≈ 0                |

# Utilité du code

Génère la courbe de l'absorbance du CO₂ en fonction de la longueur d'onde, à 440 ppm sur 1 m de colonne d'air, à partir des données spectroscopiques HITRAN.

# Fonctionnement du code

- Utilisation de l'API RADIS
    1. Paramètres : pression (1,013 atm), concentration (440 ppm), température (255 K)
    2. Calcul de l'absorbance sur 3 bandes spectrales (voir source ci-dessus)
- Récupération de l'absorbance aux niveaux des bandes d'absorption
- Interpolation et concaténation des bandes pour former une courbe continue
- Génération du graphique de l'absorbance en fonction de la longueur d'onde (4 – 17 µm)

# Installation

```bash
pip install radis numpy scipy matplotlib
```

# Utilisation

```bash
python co2_absorbance.py
```