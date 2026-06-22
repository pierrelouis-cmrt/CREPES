# Modèle 3 - colonne radiative finale

Le modèle 3 calcule les flux radiatifs d'une colonne atmosphérique déjà préparée. Il ne lit pas les fichiers ERA5 bruts pendant le calcul et sert de brique stable pour le modèle 4.

## Structure

```text
modele3/
  modele3.py                  # calcul radiatif d'une colonne
  physique.py                   # constantes et formules physiques
  calibrer_coefficients_co2.py  # calibrage HITRAN/RADIS des coefficients CO2
  donnees.py                    # chargement du paquet compact
  ressources/
    README.md                   # ressources et génération du paquet
    generer_donnees.py          # génération du paquet compact
    donnees_precalculees/
      grille_5deg_2024/
        donnees_colonnes_5deg_2024.npz
        metadata.json
        README.md
  tests/
    tester_modele3.py
    README.md
  documentation/
    README.md                   # théorie, provenance et notes ERA5
    THEORIE.md
    PROVENANCE_DONNEES.md
    recap_ERA5.md
    RECHERCHE_SHORTWAVE_ET_OPTIMISATION.md

``` 

## Contrat d'une colonne


Le calcul attend une colonne extraite du paquet compact :

```text
surface:
  latitude_deg
  longitude_deg
  mois ou jour_annee
  pression_surface_pa
  albedo_surface
  transmissivite_sw_mensuelle

couches:
  pression_bas_hpa / pression_haut_hpa
  temperature_k
  humidite_specifique_kgkg
  masse_air_kg_m2
  masse_h2o_kg_m2
```

La fonction principale est :

```python
calculer_colonne_radiative(donnees_colonne, temperature_surface_k, co2_ppm)
```

## Physique conservée

Short-wave :

```text
SW_TOA_local = S0 * max(cos(i), 0)
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
S0 = 1361 W m-2
```

Long-wave :

```text
LW_up_surface = 0.98 * sigma * T_surface^4
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
```

Il n'y a pas de mode court-onde alternatif ni de coefficient nuageux radiatif.

Les coefficients CO2/H2O utilisés dans `physique.py` sont des coefficients
effectifs de profondeur optique, pas des constantes spectroscopiques. Leur
unité est sans dimension :

- `a_CO2` vaut pour une colonne de référence `CO2 = 280 ppm` et
  `delta_p = 101325 Pa`, puis il est remis à l'échelle par `CO2_ppm / 280` et
  `delta_p / 101325`.
- `a_H2O` vaut pour `10 kg m-2` de vapeur d'eau, puis il est remis à l'échelle
  par la masse de vapeur d'eau ERA5 de la couche.
- La cible CO2 conserve l'ordre de grandeur pédagogique du doublement
  `280 -> 560 ppm` hérité du modele 2.5. Les coefficients H2O ajoutent une
  opacité simple par grandes bandes, sans raies fines.

Ces nombres ne remplacent pas HITRAN, correlated-k ou une dépendance fine en température/pression ; ils servent à garder un noyau CO2 + H2O clair et défendable pour le projet.

Une méthode de recalibrage propre des coefficients CO2 est documentée dans
`CALIBRAGE_COEFFICIENTS_CO2.md`. Le script dédié calcule des transmissions de référence HITRAN via RADIS, ajuste les `a_CO2` par bande, recale le forçage
`280 -> 560 ppm`, puis quantifie l'impact des coeurs et des ailes.

## Données

Le paquet compact versionné est dans :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Il contient la grille globale 5 degrés, les champs surface utiles, les couches verticales prétraitées, les flux ERA5 de validation, l'albédo de surface et la transmissivité court-onde mensuelle.

Les CSV historiques d'albédo viennent d'un rapport de flux solaire
`SW_UP / SW_DOWN`. En nuit polaire, ce rapport peut donner `0` alors que la surface neige/glace reste réfléchissante. Le paquet corrige donc uniquement les mailles `albedo_surface == 0` avec `snow_ice_fraction > 0.05`, par un mélange simple entre l'albédo de secours `0.30` et une valeur neige/glace `0.65`.

Régénérer le paquet :

```bash
./.venv/bin/python -m modele3.ressources.generer_donnees --overwrite
```

Lancer une colonne depuis le paquet :

```bash
./.venv/bin/python -m modele3.codes_python.modele3 \
  --lat 0 \
  --lon 0 \
  --mois 7 \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw
```

Lancer les tests :

```bash
./.venv/bin/python modele3/tests/tester_modele3.py
```

## Limites assumées

- Pas d'évolution de `T_surface(t)` dans le modèle 3.
- Pas de transport horizontal.
- Pas d'ozone, aérosols, CH4, N2O ou microphysique nuageuse.
- Émissivité de surface constante `0.98`.
- Coefficients CO2/H2O effectifs, documentés dans
  `documentation/THEORIE.md`.
