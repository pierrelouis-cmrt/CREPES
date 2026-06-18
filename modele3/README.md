# Modele 3 - colonne radiative finale

Le modele 3 calcule les flux radiatifs d'une colonne atmospherique deja
preparee. Il ne lit pas les fichiers ERA5 bruts pendant le calcul et sert de
brique stable pour le modele 4.

## Structure

```text
modele3/
  modele3.py                  # calcul radiatif d'une colonne
  physique.py                   # constantes et formules physiques
  donnees.py                    # chargement du paquet compact
  ressources/
    generer_donnees.py          # generation du paquet compact
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
```

La fonction principale est :

```python
calculer_colonne_radiative(donnees_colonne, temperature_surface_k, co2_ppm)
```

## Physique conservee

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

## Donnees

Le paquet compact versionne est dans :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Il contient la grille globale 5 degres, les champs surface utiles, les couches
verticales pretraitees, les flux ERA5 de validation, l'albedo de surface et la
transmissivite court-onde mensuelle.

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

- Pas d'evolution de `T_surface(t)` dans le modele 3.
- Pas de transport horizontal.
- Pas d'ozone, aerosols, CH4, N2O ou microphysique nuageuse.
- Emissivite de surface constante `0.98`.
- Coefficients CO2/H2O effectifs, documentes dans `THEORIE.md`.
