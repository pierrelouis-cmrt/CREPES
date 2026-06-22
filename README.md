# CREPES

Projet Climat, Groupe D, 2026

## Organisation

| Dossier                | Rôle                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `modele0_maintenance/` | Ancien modèle combiné, conservé comme référence stable.                                         |
| `modele1/`             | Colonne radiative CO2 simplifiée à 3 couches.                                                   |
| `modele2/`             | Colonne atmosphérique CO2 à 6 couches avec noyau radiatif infrarouge simplifié.                 |
| `modele2_5/`           | Itération du modèle 2 : 10 couches en pression, profil standard, bandes CO2 découpées et tests. |
| `modele3/`             | Colonne radiative finale pour le modèle 4, avec paquet `.npz` compact et provenances explicites. |
| `modele4/`             | Grille de température de surface couplée au modèle 3 et aux termes de surface du modèle 0.       |
| `modele5/`             | Grille modèle 4 rapide avec échanges radiatifs horizontaux entre colonnes voisines.              |
| `planisphere.py`       | Visualisation racine des sorties `.npz` des modèles 4 et 5.                                      |


## Lancement des codes 
Bien effectuer la commande suivante :
```bash 
python -m pip install -r modele5/requirements.txt
```

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
- Découpage CO2 en sous-bandes cœur/ailes.
- Facteur diffusif `D = 1,66`.
- Calibration sur le forçage `280 -> 560 ppm`.
- Tests numériques séparés.

### Modèle 3

Ajouts/corrections par rapport au modèle 2.5 :

- Paquet compact `modele3/ressources/donnees_precalculees/grille_5deg_2024/`.
- Grille globale `5 degrés` prête pour le modèle 4.
- Appel local par latitude/longitude et mois ou jour.
- Pression de surface locale au lieu de `1013.25 hPa` fixe.
- Profils ERA5 locaux `T(p)` et `q(p)` prétraités par couche.
- Opacité H2O effective additionnée à l'opacité CO2 avant transmission.
- Émissivité constante `0.98`.
- Albédo de surface lu depuis `ressources/albedo/albedo01.csv` à `albedo12.csv`.
- Transmissivité shortwave mensuelle :
  `ERA5 SW_down / moyenne_mensuelle(S0 * max(cos(i), 0))`.
- Suppression des corrections nuageuses arbitraires shortwave et longwave.
- Le code 3 lit les copies racine dans `ressources/albedo/`, pas
  `modele0_maintenance/`.

### Modèle 4

Première grille de surface couplée :

- Variable calculée : `T_surface(t, lat, lon)`.
- Grille globale 5 degrés du paquet compact modèle 3.
- Cellules indépendantes, sans transport horizontal.
- Flux radiatifs fournis par le modèle 3.
- Capacité thermique, flux latent et convection repris/clarifiés depuis le
  modèle 0.
- Intégration temporelle Backward Euler.
- Modèle de surface pédagogique forcé par les flux du modèle 3 : pas de
  circulation, pas d'océan dynamique, pas de modèle climatique complet.

### Modèle 5

Ajouts par rapport au modèle 4 rapide :

- Échanges radiatifs horizontaux infrarouges entre colonnes voisines.
- Échange calculé couche par couche et bande par bande avec les opacités CO2 +
  H2O du modèle 3.
- Géométrie sphérique des mailles et conservation de la puissance échangée aux
  interfaces.
- Terme `Q_horizontal` ajouté au bilan de surface.
- Paramètre `--facteur-horizontal` pour isoler l'effet du couplage horizontal.
- Paramètre `--couplage-couches` pour relier une anomalie de surface à
  l'émission latérale des couches.

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
./.venv/bin/python modele2/codes_python/modele2.py
```

Régénérer le profil vertical de pression et de CO2 :

```bash
./.venv/bin/python modele2/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --no-plot
```

La documentation détaillée du modèle 2 est dans `modele2/README.md`.

## Modèle 2.5

Lancer le noyau radiatif du modèle 2.5 :

```bash
./.venv/bin/python modele2_5/codes_python/modele2_5.py
```

Lancer les tests numériques séparés :

```bash
./.venv/bin/python modele2_5/ressources/tester_modele2_5.py
```

Régénérer les profils standard et CO2 :

```bash
./.venv/bin/python modele2_5/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 84 --step-m 100 --surface-co2-ppm 420 --output modele2_5/ressources/profil_vertical_atmosphere_co2.png --csv modele2_5/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

La documentation détaillée du modèle 2.5 est dans `modele2_5/README.md`.

## Modèle 3

Régénérer le paquet compact :

```bash
./.venv/bin/python -m modele3.ressources.generer_donnees --overwrite
```

Lancer une colonne depuis le paquet global :

```bash
cd modele3
./modele3.py
```

Avec des options depuis la racine :

```bash
./.venv/bin/python -m modele3 --lat 0 --lon 0 --mois 7 --temperature-surface 293.0 --moyenne-journaliere-sw
```

Lancer les tests :

```bash
./.venv/bin/python modele3/tests/tester_modele3.py
```

Documentation détaillée :

- `modele3/README.md`
- `modele3/documentation/THEORIE.md`
- `modele3/documentation/PROVENANCE_DONNEES.md`

## Modèle 4

Lancer le diagnostic mensuel global par défaut :

```bash
./.venv/bin/python -m modele4.codes_python.modele4
```

Lancer le moteur rapide, sortie toutes les 4 heures par défaut :

```bash
./.venv/bin/python -m modele4.codes_python.rapide
```

Lancer un test temporel court sur une cellule :

```bash
./.venv/bin/python -m modele4.modele4 --mode temporel --jours 0.020833333333333332 --max-latitudes 1 --max-longitudes 1 --frequence-sortie-pas 1 --output /tmp/modele4_test.npz
```

Lancer les tests :

```bash
./.venv/bin/python modele4/tests/tester_modele4.py
./.venv/bin/python modele4/tests/tester_rapide.py
```

Documentation détaillée :

- `modele4/README.md`
- `modele4/THEORIE.md`

## Modèle 5

Lancer le modèle couplé horizontal, sortie toutes les 4 heures par défaut :

```bash
./.venv/bin/python -m modele5.codes_python.modele5
```

Lancer une petite grille de développement :

```bash
./.venv/bin/python -m modele5.modele5 --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele5/sorties/simulation_dev.npz
```

Comparer au modèle 4 rapide sans échange horizontal :

```bash
./.venv/bin/python -m modele5.modele5 --facteur-horizontal 0 --output modele5/sorties/simulation_sans_horizontal.npz
```

Lancer les tests :

```bash
./.venv/bin/python modele5/tests/tester_modele5.py
```

Documentation détaillée :

- `modele5/README.md`

## Visualisation

Afficher une sortie `.npz` des modèles 4 ou 5 depuis la racine :

```bash
./.venv/bin/python planisphere.py
```

Sans argument, le script propose les fichiers présents dans `modele4/sorties/`
et `modele5/sorties/`. L'ancien chemin
`modele4/visualisation/planisphere.py` reste utilisable et délègue au script
racine.
