# Modèle 3 - colonne radiative finale

Le modèle 3 calcule les flux radiatifs d'une colonne atmosphérique déjà
préparée. Il ne lit pas les fichiers ERA5 bruts pendant le calcul et sert de
brique stable pour le modèle 4.

## Structure

```text
modele3/
  modele3.py                  # calcul radiatif d'une colonne
  physique.py                   # constantes et formules physiques
  donnees.py                    # chargement du paquet compact
  ressources/
    generer_donnees.py          # génération du paquet compact
    donnees_precalculees/
      grille_5deg_2024/
        donnees_colonnes_5deg_2024.npz
        metadata.json
        README.md
  tests/
    tester_modele3.py
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

diagnostics_donnees:
  couches_ignorees_incompletes
  couches_ignorees_non_positives
```

La fonction principale est :

```python
calculer_colonne_radiative(donnees_colonne, temperature_surface_k, co2_ppm)
```

## Physique conservée

Shortwave :

```text
SW_TOA_local = S0 * max(cos(i), 0)
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
S0 = 1361 W m-2
```

Quand une colonne est demandee par `mois` et que `moyenne_journaliere_sw=True`,
le modèle utilise `sw_toa_moyen_mensuel_w_m2`, calcule sur le mois complet dans
le paquet. Sans moyenne journalière, le calcul reste instantané sur le jour
milieu de mois, qui est seulement un jour représentatif.

Longwave :

```text
LW_up_surface = 0.98 * sigma * T_surface^4
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
```

Il n'y a pas de mode shortwave alternatif ni de coefficient nuageux radiatif.

## Données

Le paquet compact versionné est dans :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Il contient la grille globale 5 degrés, les champs surface utiles, les couches
verticales prétraitées, les flux ERA5 de validation, l'albédo de surface et la
transmissivité shortwave mensuelle.

Les longitudes du paquet sont en convention `-180..180`. Les CSV d'albédo sont
normalisés dans cette même convention avant sélection au plus proche, afin de
ne pas rater les points proches de l'antimeridien.

Regenerer le paquet :

```bash
./.venv/bin/python -m modele3.ressources.generer_donnees --overwrite
```

Lancer une colonne depuis le paquet :

```bash
./.venv/bin/python -m modele3.modele3 \
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

## Limites assumees

- Pas d'évolution de `T_surface(t)` dans le modèle 3.
- Pas de transport horizontal.
- Pas d'ozone, aérosols, CH4, N2O ou microphysique nuageuse.
- Émissivité de surface constante `0.98`.
- Coefficients CO2/H2O effectifs, documentés dans `THEORIE.md`.
