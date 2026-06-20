# Provenance des donnees du modele 3

Le calcul 3 lit uniquement le paquet compact :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Les fichiers bruts locaux sont lus seulement par :

```text
modele3/ressources/generer_donnees.py
```

## Sources actives

| Source | Variables | Transformation | Sortie paquet |
| --- | --- | --- | --- |
| ERA5 niveaux de pression, `ressources/*.nc` | `t`, `q`, `pressure_level` | Selection grille 5 degres, moyennes par couche de pression. | `temperature_couche_k`, `humidite_specifique_couche_kgkg`, `masse_air_couche_kg_m2`, `masse_h2o_couche_kg_m2` |
| ERA5 single levels, `ressources/**/*.nc` | `sp`, `t2m`, `skt`, `lsm`, `siconc`, `sd` | Selection grille 5 degres, pression en hPa, neige/glace = max(`siconc`, `sd > 0.01 m`). | `pression_surface_hpa`, temperatures surface, fractions terre/neige-glace |
| ERA5 flux mensuels, `ressources/**/*.nc` | `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf` | Selection grille 5 degres, OLR stocke positif. | Flux ERA5 de validation |
| Geometrie solaire 3 + ERA5 SW down | `avg_sdswrf`, `S0 * max(cos(i), 0)` | Moyenne mensuelle TOA par latitude, puis rapport ERA5/TOA borne `[0, 1]`. | `sw_toa_moyen_mensuel_w_m2`, `transmissivite_sw_mensuelle` |
| CSV albedo, `ressources/albedo/albedo01.csv` ... `albedo12.csv` | albedo mensuel | Selection au plus proche, bornage `[0, 1]`. | `albedo_surface` |

Les CSV d'albedo sont des copies racine de ressources historiquement produites
pour le modele 0. Le code 3 ne lit pas `modele0_maintenance/`.

## Transmissivite court-onde

```text
SW_TOA_moyen_mensuel =
    moyenne_mensuelle(1361 * max(cos(i), 0))

transmissivite_sw_mensuelle =
    era5_sw_down_surface_w_m2 / SW_TOA_moyen_mensuel
```

La transmissivite est bornee dans `[0, 1]`. Les valeurs bornees sont comptees
dans `metadata.json`.

## Donnees non utilisees

| Donnee disponible | Statut |
| --- | --- |
| MODIS HDF | Non utilise ; emissivite constante `0.98`. |
| ERA5 `z`, `cbh` | Non stockes ; les couches sont en pression. |
| 37 niveaux ERA5 bruts | Non stockes ; seules les couches pretraitees sont versionnees. |

## References externes

- Copernicus ERA5 monthly averaged pressure levels.
- Copernicus ERA5 monthly averaged single levels.
- NASA POWER pour la provenance historique des CSV d'albedo de surface.
