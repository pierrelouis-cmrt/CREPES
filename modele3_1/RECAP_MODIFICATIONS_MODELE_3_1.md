# Recap modifications modele 3 -> modele 3.1

## Objectif

Transformer le modele 3 en brique colonne stable pour le modele 4 :

```text
calculer_colonne_radiative(donnees_colonne, temperature_surface_k, co2_ppm)
```

Le calcul ne lit plus les gros fichiers bruts. Il consomme une colonne deja
extraite depuis un paquet compact `.npz`.

## Modifications de code

| Zone | Modele 3 | Modele 3.1 |
| --- | --- | --- |
| Structure | `modele3.py`, `donnees.py`, `physique/calculs.py`. | `modele3_1.py`, `donnees.py`, `physique.py`, `generer_donnees.py`. |
| Donnees normales | JSON Paris ou lecture locale ERA5 directe. | Paquet compact `donnees_colonnes_5deg_2024.npz`. |
| API principale | Peut charger des donnees dans le chemin CLI. | Recoit une colonne deja preparee ; compatible boucle modele 4. |
| Diagnostics | Couches et bandes souvent renvoyees. | Diagnostics lourds desactives par defaut. |
| Tests | Tests du modele 3 + JSON Paris. | Tests paquet, sources, emissivite, nuages SW/LW, boucle de colonnes. |

## Modifications physiques

| Sujet | Modele 3 | Modele 3.1 |
| --- | --- | --- |
| Emissivite | `0.98`, `0.985` ocean, branches neige/glace. | Constante `0.98` partout. |
| Albedo surface | `fal` ERA5 ou secours `0.30`. | Donnee explicite : CSV mensuels racine `ressources/albedo`, ou entree fournie. Secours seulement documente. |
| Albedo nuageux SW | `0.50 * cloud_total`. | Champ explicite `albedo_nuages_effectif` issu de CERES : `(toa_sw_all - toa_sw_clr_c) / solar`. |
| Transmission SW surface | Absente. | `transmissivite_sw_mensuelle = ERA5 SW_down / moyenne_mensuelle(S0*cos(i))`, bornee `[0, 1]`. |
| Nuages LW | `tau_nuage = 0.10 * fraction_nuageuse`. | Retire du chemin par defaut. Les fractions nuageuses restent diagnostiques. |
| CO2 + H2O | Addition des opacites avant transmission. | Conserve : `tau_total = tau_CO2 + tau_H2O`. |
| Court-onde | Simple et fortement approximatif. | Toujours simple, mais les coefficients caches sont supprimes et les limites sont explicites. |

## Systeme de donnees ajoute

Nouveau generateur :

```text
modele3_1/generer_donnees.py
```

Entrees actives :

```text
ressources/*.nc ou ressources/**/*.nc          # ERA5 locaux
ressources/albedo/albedo01.csv ... albedo12.csv
ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc
```

Sorties :

```text
modele3_1/donnees_precalculees/grille_5deg_2024/
  donnees_colonnes_5deg_2024.npz
  metadata.json
  README.md
```

Le paquet contient une grille globale `5 degres` :

```text
36 latitudes x 72 longitudes x 12 mois
```

Il stocke seulement les champs necessaires : pression de surface, albedos,
transmissivite court-onde mensuelle, diagnostics surface, flux ERA5 de
validation, moyennes de couches, masses air et H2O. Les tableaux sont
quantifies et documentes dans `metadata.json`.

## Ressources copiees depuis le modele 0

Les fichiers utiles du modele 0 ont ete copies vers la racine :

```text
ressources/albedo/albedo01.csv ... albedo12.csv
ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc
ressources/albedo/README.md
```

Le dossier `modele0_maintenance/` est intact. Le code 3.1 ne l'importe pas et
ne lit pas directement ses donnees. La mention du modele 0 sert uniquement a
expliquer la provenance historique des fichiers copies.

## Verification actuelle

Commandes executees :

```bash
./.venv/bin/python -m modele3_1.generer_donnees --overwrite
./.venv/bin/python modele3_1/tests/tester_modele3_1.py
./.venv/bin/python -m modele3_1.modele3_1 --lat 48.8566 --lon 2.3522 --mois 7 --temperature-surface 293.0 --moyenne-journaliere-sw
```

Resultats constates :

```text
paquet compact = 2.1 Mo
tests_modele3_1_ok
Paris utilise le point grille 47.5 N, 2.5 E
emissivite_surface = 0.98
albedo_surface = 0.1726
albedo_nuages_effectif = 0.1607
```

## Limites conservees volontairement

- Pas de temperature de surface integree dans le temps.
- Pas de grille dynamique dans le calcul de colonne.
- Pas de transfert solaire atmospherique complet.
- Pas d'ozone, aerosols, CH4, N2O.
- Pas de microphysique nuageuse.
- Pas de lecture MODIS/HDF dans 3.1.
- Pas de calibration cachee pour forcer l'accord avec ERA5.
