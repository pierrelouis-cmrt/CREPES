# Modele 3.1 - colonne radiative locale propre pour le modele 4

Le modele 3.1 calcule les flux radiatifs d'une colonne locale deja preparee.
Il ne fait pas evoluer la temperature de surface et ne lit pas les gros fichiers
bruts pendant le calcul. Son role est d'etre appele ensuite par le modele 4 pour
chaque cellule de grille.

## Contrat scientifique

Entrees minimales d'une colonne :

| Bloc | Variable | Role |
| --- | --- | --- |
| Surface | `latitude_deg`, `longitude_deg`, `mois` ou `jour_annee` | Geometrie solaire et localisation. |
| Surface | `pression_surface_pa` | Base locale de la colonne ; plus de `1013.25 hPa` fixe. |
| Surface | `albedo_surface` | Fraction court-onde reflechie par la surface. |
| Surface | `albedo_nuages_effectif` | Correction court-onde nuageuse issue de CERES ou fournie par l'appelant. |
| Surface | `transmissivite_sw_mensuelle` | Transmission atmospherique court-onde mensuelle derivee d'ERA5. |
| Surface | `emissivite_surface` | Fixee a `0.98` dans 3.1. |
| Couches | `pression_bas/haut`, `temperature_k`, `humidite_specifique_kgkg`, `masse_air`, `masse_h2o` | Transfert long-onde CO2 + H2O. |
| Parametres | `temperature_surface_k`, `co2_ppm` | Temperature imposee et concentration CO2 uniforme. |

Sorties principales :

```text
SW_incident_surface
SW_TOA_local
SW_down_surface
SW_absorbe_surface
LW_up_surface
LW_down_surface
LW_down_absorbe_surface
OLR
flux_net_radiatif_surface
```

## Donnees actives

Le calcul normal utilise le paquet versionnable :

```text
modele3_1/donnees_precalculees/grille_5deg_2024/
  metadata.json
  donnees_colonnes_5deg_2024.npz
  README.md
```

Ce paquet est genere depuis les ressources locales racine :

| Source racine | Variables lues | Usage 3.1 |
| --- | --- | --- |
| `ressources/db7d35d0a9c6110c5f6d54212de24b21.nc` | ERA5 `t`, `q`, `cc` sur niveaux de pression. | Moyennes par couche de temperature, humidite et fraction nuageuse diagnostique. |
| `ressources/4d43.../data_stream-moda_stepType-avgua.nc` | ERA5 `sp`, `t2m`, `skt`, `lsm`, `siconc`, `sd`, `tcc`, `lcc`, `mcc`, `hcc`. | Pression de surface, diagnostics surface et nuages. |
| `ressources/4d43.../data_stream-moda_stepType-avgad.nc` | ERA5 `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf`. | Flux de validation, pas de calibration cachee. |
| Geometrie solaire 3.1 + ERA5 `avg_sdswrf` | `S0 * max(cos(i), 0)` moyen mensuel et SW descendant ERA5. | `sw_toa_moyen_mensuel_w_m2`, `transmissivite_sw_mensuelle`. |
| `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Albedo de surface mensuel. | `albedo_surface`. |
| `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | CERES `toa_sw_all_mon`, `toa_sw_clr_c_mon`, `solar_mon`. | `albedo_nuages_effectif`. |

Les fichiers `ressources/albedo/*` sont des copies racine de ressources utiles
issues historiquement du modele 0. Le dossier `modele0_maintenance/` reste
intact et n'est pas lu par le code 3.1.

La provenance complete, y compris les variables CERES/ERA5 et les donnees
explicitement non utilisees, est detaillee dans `PROVENANCE_DONNEES.md`.

## Provenance CERES et albedo

Albedo de surface :

```text
albedo_surface = ALLSKY_SFC_SW_UP / ALLSKY_SFC_SW_DWN
```

Les CSV mensuels proviennent de NASA POWER via le travail historique du modele
0, puis ont ete copies dans `ressources/albedo/` pour que 3.1 n'ait pas de
dependance directe au modele 0.

Albedo nuageux effectif :

```text
albedo_nuages_effectif =
    (toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon
```

Cette grandeur CERES represente un effet radiatif nuageux court-onde effectif
au sommet de l'atmosphere. Elle ne pretend pas etre un albedo microphysique du
nuage. Elle remplace explicitement l'ancienne approximation opaque
`0.50 * cloud_total`.

## Formules

Court-onde recommande pour le modele 4 :

```text
SW_TOA_local(t) = S0 * max(cos(i(t)), 0)
transmissivite_sw_mensuelle =
    era5_sw_down_surface_w_m2 / moyenne_mensuelle(SW_TOA_local)
SW_down_surface(t) =
    transmissivite_sw_mensuelle * SW_TOA_local(t)
SW_absorbe_surface =
    SW_down_surface * (1 - albedo_surface)
```

Le mode diagnostic historique reste disponible avec
`mode_court_onde="toa_nuages_ceres"` :

```text
SW_absorbe_surface =
    SW_incident_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Long-onde de surface :

```text
LW_up_surface = 0.98 * sigma * T_surface^4
LW_down_absorbe_surface = 0.98 * LW_down_surface
```

Opacites infrarouges par couche et par bande :

```text
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
emissivite_couche = 1 - transmission
```

Il n'y a plus de `tau_nuage = 0.10 * fraction_nuageuse` dans le chemin physique
par defaut.

## Lancer

Generer ou regenerer le paquet compact :

```bash
./.venv/bin/python -m modele3_1.generer_donnees --overwrite
```

Calculer Paris, extrait depuis la grille 5 degres :

```bash
./.venv/bin/python -m modele3_1.modele3_1 \
  --lat 48.8566 \
  --lon 2.3522 \
  --mois 7 \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw \
  --mode-court-onde transmissivite_sw
```

Lancer les tests :

```bash
./.venv/bin/python modele3_1/tests/tester_modele3_1.py
```

## Limites assumees

- Pas d'evolution de `T_surface(t)`.
- Pas de dynamique atmospherique ni d'echanges horizontaux.
- Pas d'ozone, aerosols, CH4, N2O ou microphysique nuageuse.
- Pas de lecture directe MODIS/HDF dans 3.1 ; emissivite constante `0.98`.
- Court-onde volontairement simple : 3.1 garde `S0 * max(cos(i), 0)` et utilise
  ERA5 seulement pour une transmissivite mensuelle moyenne.
- Coefficients CO2/H2O effectifs herites du modele 3 et documentes dans
  `THEORIE.md`.
