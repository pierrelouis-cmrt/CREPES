# Provenance des donnees du modele 3.1

Ce fichier documente toutes les donnees utilisees par 3.1. Le principe est
simple : les fichiers lourds et les ressources albedo sont lus depuis
`ressources/`, puis transformes une fois en paquet compact dans
`modele3_1/donnees_precalculees/grille_5deg_2024/`.

## Regle de dependance

Le modele 3.1 ne lit pas directement `modele0_maintenance/` et n'importe aucun
module depuis ce dossier. Les fichiers d'albedo utiles qui existaient dans le
modele 0 ont ete copies dans `ressources/albedo/`. Cette copie racine est la
source active.

## Inventaire des sources actives

| Chemin actif | Nature | Variables lues | Transformation | Sortie paquet |
| --- | --- | --- | --- | --- |
| `ressources/db7d35d0a9c6110c5f6d54212de24b21.nc` | ERA5 mensuel sur niveaux de pression. | `t`, `q`, `cc`, `pressure_level`. | Selection grille 5 degres, puis moyenne en pression dans les couches 3.1. | `temperature_couche_k`, `humidite_specifique_couche_kgkg`, `fraction_nuageuse_couche`, `masse_air_couche_kg_m2`, `masse_h2o_couche_kg_m2`. |
| `ressources/4d43b9edb397c8d4595fc350432d5ac4/data_stream-moda_stepType-avgua.nc` | ERA5 mensuel single levels. | `sp`, `t2m`, `skt`, `lsm`, `siconc`, `sd`, `tcc`, `lcc`, `mcc`, `hcc`. | Selection grille 5 degres. `sp` est converti en hPa ; neige/glace = max(`siconc`, `sd > 0.01 m`). | `pression_surface_hpa`, `temperature_2m_k`, `skin_temperature_k`, `land_fraction`, `snow_ice_fraction`, nuages diagnostiques. |
| `ressources/4d43b9edb397c8d4595fc350432d5ac4/data_stream-moda_stepType-avgad.nc` | ERA5 flux moyens mensuels. | `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf`. | Selection grille 5 degres. `avg_tnlwrf` est stocke en valeur absolue pour l'OLR positif. | Flux ERA5 de validation. |
| `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Albedo de surface mensuel. | Valeurs grille CSV. | Selection au plus proche sur la grille 5 degres, bornage `[0, 1]`. | `albedo_surface`. |
| `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | CERES EBAF-TOA mensuel. | `toa_sw_all_mon`, `toa_sw_clr_c_mon`, `solar_mon`. | Formule effective nuageuse, selection au plus proche, bornage `[0, 0.95]`. | `albedo_nuages_effectif`. |

## Detail CERES

Fichier actif :

```text
ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc
```

Produit externe :

```text
NASA CERES EBAF-TOA Edition 4.2.1
```

Variables utilisees :

```text
toa_sw_all_mon    flux court-onde reflechi TOA tout temps
toa_sw_clr_c_mon  flux court-onde reflechi TOA ciel clair
solar_mon         flux solaire incident TOA
```

Formule :

```text
albedo_nuages_effectif =
    (toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon
```

Interpretation exacte : ce champ mesure l'effet radiatif court-onde
supplementaire des nuages dans CERES, vu au sommet de l'atmosphere et normalise
par le solaire entrant. Ce n'est pas un albedo local de gouttelettes, ni une
fraction nuageuse, ni une propriete optique spectrale.

Pourquoi l'utiliser : il remplace une constante cachee du modele 3
(`0.50 * cloud_total`) par une grandeur observationnelle directement reliee au
bilan radiatif court-onde.

Limite : appliquer un effet TOA directement dans une formule de surface reste
une approximation. Elle est explicite et documentee ; elle ne doit pas etre lue
comme un transfert solaire atmospherique complet.

Reference :

```text
https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1
```

## Detail albedo surface

Fichiers actifs :

```text
ressources/albedo/albedo01.csv ... ressources/albedo/albedo12.csv
```

Origine externe historique :

```text
NASA POWER
ALLSKY_SFC_SW_UP / ALLSKY_SFC_SW_DWN
```

Interpretation : rapport mensuel du flux solaire court-onde montant a la surface
sur le flux solaire court-onde descendant a la surface. Le resultat est une
estimation d'albedo de surface mensuel.

Usage 3.1 :

```text
SW_absorbe_surface =
    SW_incident_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

References :

```text
https://power.larc.nasa.gov/docs/tutorials/parameters/
https://power.larc.nasa.gov/docs/methodology/
```

## Detail ERA5

ERA5 fournit les profils atmospheriques, la pression de surface, des diagnostics
surface et des flux de validation. Le modele 3.1 ne calibre pas ses coefficients
sur les flux ERA5 pendant la generation ; il les conserve pour comparer les
ordres de grandeur.

References :

```text
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
```

## Donnees explicitement non utilisees

| Donnee disponible | Statut 3.1 | Raison |
| --- | --- | --- |
| MODIS HDF `MOD11C3.A2024/*.hdf` | Non lu. | L'emissivite est fixee a `0.98` pour eviter une complexite inutile a ce stade. |
| `cbh` ERA5 | Non stocke. | Pas de microphysique nuageuse ni de hauteur de base de nuage dans le calcul. |
| `z` ERA5 geopotentiel | Non stocke. | Les couches sont construites en pression ; pas de diagnostic altitude necessaire. |
| Flux ERA5 clear-sky | Non stockes dans le paquet actuel. | Non necessaires aux tests/calculeur 3.1. |
| 37 niveaux ERA5 bruts | Non stockes. | Moyennes de couches pretraitees suffisent au calcul. |

## Quantification

Les tableaux physiques sont quantifies avant ecriture dans le `.npz`. Les
facteurs sont dans `metadata.json` :

```text
valeur = valeur_stockee * scale_factor + offset
```

Les valeurs manquantes utilisent une sentinelle par type (`65535` pour
`uint16`, `-32768` pour `int16`) et sont reconstruites en `NaN` au chargement.
La quantification explique pourquoi le paquet complet tient dans environ
`2.1 Mo`.
