# Provenance des données du modèle 3

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
| ERA5 niveaux de pression, `ressources/*.nc` | `t`, `q`, `pressure_level` | Sélection grille 5 degrés, moyennes par couche de pression; couches < 0.1 hPa ignorées avant quantification. | `temperature_couche_k`, `humidite_specifique_couche_kgkg`, `masse_air_couche_kg_m2`, `masse_h2o_couche_kg_m2` |
| ERA5 single levels, `ressources/**/*.nc` | `sp`, `t2m`, `skt`, `lsm`, `siconc`, `sd` | Sélection grille 5 degrés, pression en hPa, neige/glace = max(`siconc`, `sd > 0.01 m`). | `pression_surface_hpa`, températures surface, fractions terre/neige-glace |
| ERA5 flux mensuels, `ressources/**/*.nc` | `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf` | Sélection grille 5 degrés, OLR stocké positif. | Flux ERA5 de validation |
| Géométrie solaire 3 + ERA5 SW down | `avg_sdswrf`, `S0 * max(cos(i), 0)` | Moyenne mensuelle TOA par latitude, puis rapport ERA5/TOA borné `[0, 1]`. | `sw_toa_moyen_mensuel_w_m2`, `transmissivite_sw_mensuelle` |
| CSV albédo, `ressources/albedo/albedo01.csv` ... `albedo12.csv` | albédo mensuel | Longitudes source et cible normalisées `-180..180`, sélection au plus proche, bornage `[0, 1]`. | `albedo_surface` |

Les CSV d'albédo sont des copies racine de ressources historiquement produites
pour le modèle 0. Le code 3 ne lit pas `modele0_maintenance/`.

## Transmissivité court-onde

```text
SW_TOA_moyen_mensuel =
    moyenne_mensuelle(1361 * max(cos(i), 0))

transmissivite_sw_mensuelle =
    era5_sw_down_surface_w_m2 / SW_TOA_moyen_mensuel
```

La transmissivité est bornée dans `[0, 1]`. Les valeurs bornées sont comptées
dans `metadata.json`.

Quand le calcul est appelé par mois avec `moyenne_journaliere_sw=True`, le
modèle utilise la moyenne mensuelle `sw_toa_moyen_mensuel_w_m2`. Le jour milieu
de mois reste seulement le jour représentatif des calculs instantanés.

## Données non utilisées

| Donnée disponible | Statut |
| --- | --- |
| MODIS HDF | Non utilisé ; émissivité constante `0.98`. |
| ERA5 `z`, `cbh` | Non stockés ; les couches sont en pression. |
| 37 niveaux ERA5 bruts | Non stockés ; seules les couches prétraitées sont versionnées. |

## Références externes

- Copernicus ERA5 monthly averaged pressure levels.
- Copernicus ERA5 monthly averaged single levels.
- NASA POWER pour la provenance historique des CSV d'albédo de surface.
