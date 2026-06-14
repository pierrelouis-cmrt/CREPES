# CREPES

Projet Climat, Groupe D, 2026

## Organisation

| Dossier                | Rôle                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `modele0_maintenance/` | Ancien modèle combiné, conservé comme référence stable.                                         |
| `modele1/`             | Colonne radiative CO2 simplifiée à 3 couches.                                                   |
| `modele2/`             | Colonne atmosphérique CO2 à 6 couches avec noyau radiatif infrarouge simplifié.                 |
| `modele2_5/`           | Itération du modèle 2 : 10 couches en pression, profil standard, bandes CO2 découpées et tests. |

## Résumé rapide des modèles

### Modèle 1

- Colonne atmosphérique moyenne à 3 couches.
- Flux solaire moyen et albédo.
- Émission infrarouge par Stefan-Boltzmann et Planck par bande.
- Bandes CO2 à 15 µm et 4,3 µm.
- Transmission, absorption et émissivité par Beer-Lambert.
- Flux infrarouge sortant au sommet et descendant à la surface.

### Modèle 2

Ajouts par rapport au modèle 1 :

- 6 couches verticales jusqu'à 80 km.
- Profil pression-température-CO2 issu de l'atmosphère standard.
- CO2 moyen par couche pondéré par la masse d'air.
- Opacité par couche via pression et concentration en CO2.
- Génération CSV/PNG du profil vertical.

### Modèle 2.5

Ajouts par rapport au modèle 2 :

- 10 couches en niveaux de pression.
- Profil de température standard 1976.
- Découpage CO2 en sous-bandes coeur/ailes.
- Facteur diffusif `D = 1,66`.
- Calibration sur le forçage `280 -> 560 ppm`.
- Tests numériques séparés.

## Modèle 0

Lancer une simulation courte depuis la racine :

```bash
python3 modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Inventorier les données du modèle 0 :

```bash
python3 modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

La documentation complète du modèle 0 est dans
`modele0_maintenance/README.md`.

## Modèle 2

Lancer le noyau radiatif du modèle 2 :

```bash
./.venv/bin/python modele2/modele2.py
```

Régénérer le profil vertical de pression et de CO2 :

```bash
./.venv/bin/python modele2/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/ressources/profil_vertical_atmosphere_co2.png --csv modele2/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

La documentation détaillée du modèle 2 est dans `modele2/README.md`.

## Modèle 2.5

Lancer le noyau radiatif du modèle 2.5 :

```bash
./.venv/bin/python modele2_5/modele2_5.py
```

Lancer les tests numériques séparés :

```bash
./.venv/bin/python modele2_5/ressources/tester_modele2_5.py
```

Régénérer les profils standard et CO2 :

```bash
./.venv/bin/python modele2_5/ressources/profil_temperature_standard.py --max-altitude-km 84 --step-m 100 --output modele2_5/ressources/profil_temperature_standard.png --csv modele2_5/ressources/profil_temperature_standard.csv --no-plot
./.venv/bin/python modele2_5/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 84 --step-m 100 --surface-co2-ppm 420 --output modele2_5/ressources/profil_vertical_atmosphere_co2.png --csv modele2_5/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

La documentation détaillée du modèle 2.5 est dans `modele2_5/README.md`.
