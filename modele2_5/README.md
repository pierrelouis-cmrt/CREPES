# Modèle 2.5

Itération du modèle 2 : colonne radiative CO₂ à dix couches de pression, profil standard, bandes cœur/ailes et coefficients optiques calibrés.

## Installation

```bash
python -m pip install -r modele2_5/requirements.txt
```

## Commandes

Depuis la racine du dépôt :

```bash
python modele2_5/modele2_5.py
python modele2_5/ressources/tester_modele2_5.py
python modele2_5/ressources/calibrer_coefficients_optiques.py
```

Régénérer les profils :

```bash
python modele2_5/ressources/profil_temperature_standard.py --max-altitude-km 84 --step-m 100 --output modele2_5/ressources/profil_temperature_standard.png --csv modele2_5/ressources/profil_temperature_standard.csv --no-plot
python modele2_5/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 84 --step-m 100 --surface-co2-ppm 420 --output modele2_5/ressources/profil_vertical_atmosphere_co2.png --csv modele2_5/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

## Structure

| Élément | Rôle |
| --- | --- |
| `modele2_5.py` | Noyau radiatif CO₂ autonome. |
| `ressources/profil_temperature_standard.py` | Génère le profil de température standard. |
| `ressources/profil_vertical_atmosphere_co2.py` | Génère le profil pression-température-CO₂. |
| `ressources/calibrer_coefficients_optiques.py` | Calibre les opacités sur le doublement du CO₂. |
| `ressources/tester_modele2_5.py` | Tests numériques. |
| `THEORIE.md` | Hypothèses, équations, calibration, résultats et limites. |
