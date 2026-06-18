# Modele 4 - grille de surface couplee au modele 3

Le modele 4 fait evoluer une temperature de surface sur la grille globale
5 degres preparee pour le modele 3. Chaque cellule est independante : il n'y a
pas encore de transport horizontal, de circulation atmospherique ou d'ocean
dynamique.

## Role

Le modele 4 ne recode pas la colonne radiative. Pour chaque cellule et chaque
pas de temps, il appelle le modele 3, recupere les flux radiatifs de surface et
integre le bilan d'energie :

```text
C_surface dT_surface/dt =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
  - Q_latent
  - Q_convection
```

Les termes `Q_latent`, `Q_convection` et la capacite surfacique reprennent les
idees et constantes du modele 0, mais dans des fonctions isolees et testables.

## Structure

```text
modele4/
  modele4.py               # moteur, CLI, integration temporelle
  surface.py               # capacite, chaleur latente, convection
  tests/tester_modele4.py  # tests numeriques minimaux
  README.md
  THEORIE.md
  plan.md
```

## Execution directe

Depuis la racine du projet, cette commande suffit :

```bash
./.venv/bin/python -m modele4.modele4
```

Par defaut, le modele produit une sortie mensuelle globale :

- grille complete 5 degres, donc `36 x 72` cellules ;
- 12 cartes, une par mois ;
- fichier ecrit dans `modele4/sorties/simulation_modele4.npz` ;
- barre de progression active.

La variable principale a alors la forme :

```text
temperature_surface_k[mois, latitude, longitude]
```

soit normalement :

```text
(12, 36, 72)
```

## Lancer une simulation courte temporelle

Depuis la racine du projet :

```bash
./.venv/bin/python -m modele4.modele4 \
  --mode temporel \
  --jours 0.020833333333333332 \
  --max-latitudes 1 \
  --max-longitudes 1 \
  --frequence-sortie-pas 1 \
  --output /tmp/modele4_test.npz
```

`0.020833333333333332` jour correspond a un seul pas de 1800 s.

Lancer une petite grille de developpement :

```bash
./.venv/bin/python -m modele4.modele4 \
  --mode temporel \
  --jours 1 \
  --max-latitudes 4 \
  --max-longitudes 8 \
  --output modele4/sorties/simulation_dev.npz
```

Lancer la grille complete 5 degres :

```bash
./.venv/bin/python -m modele4.modele4 \
  --mode temporel \
  --jours 1 \
  --output modele4/sorties/simulation_globale_1j.npz
```

La grille complete appelle beaucoup de colonnes radiatives. Pour developper, il
est preferable de commencer avec `--max-latitudes` et `--max-longitudes`.

## Options principales

- `--mode` : `mensuel` par defaut pour 12 cartes globales, ou `temporel` pour
  une integration pas-a-pas.
- `--jours` : duree de simulation en jours.
- `--dt` : pas de temps en secondes, `1800` par defaut.
- `--co2` : concentration CO2 transmise au modele 3.
- `--temperature-initiale` : valeur imposee partout ; sinon le modele utilise
  `skin_temperature_k`, puis `temperature_2m_k`, puis `288.15 K`.
- `--frequence-sortie-pas` : frequence d'ecriture de `T_surface`.
- `--convection` : `aucune`, `forcee`, `naturelle`, ou `toutes`.
- `--facteur-latent` : multiplicateur du flux latent ; `0` le desactive.
- `--vent` : vent constant en m/s pour la convection forcee.
- `--max-latitudes`, `--max-longitudes` : sous-grille rapide de developpement.
- `--rzsm-csv` : CSV RZSM du modele 0 pour utiliser la capacite thermique
  continentale issue de l'humidite du sol.
- `--no-progress` : desactive la barre de progression console.

Exemple avec le CSV RZSM du modele 0 :

```bash
./.venv/bin/python -m modele4.modele4 \
  --max-latitudes 4 \
  --max-longitudes 8 \
  --rzsm-csv modele0_maintenance/ressources/capacite_humidite/average_rzsm_tout.csv \
  --output modele4/sorties/simulation_dev_rzsm.npz
```

## Sortie NPZ

Le fichier `.npz` contient :

- `temperature_surface_k[mois, lat, lon]` en mode mensuel ;
- `temperature_surface_k[temps, lat, lon]` en mode temporel ;
- `mois` : `1..12` en mode mensuel ;
- `temps_s` ;
- `lat_deg`, `lon_deg` ;
- `capacite_surface_j_m2_k` ;
- flux moyens sur la simulation :
  `sw_absorbe_surface_moyen_w_m2`,
  `lw_down_absorbe_surface_moyen_w_m2`,
  `lw_up_surface_moyen_w_m2`,
  `flux_latent_moyen_w_m2`,
  `flux_convection_moyen_w_m2`,
  `flux_net_surface_moyen_w_m2` ;
- `metadata_json`, qui documente les options et sources.

## Tests

```bash
./.venv/bin/python modele4/tests/tester_modele4.py
```

Les tests verifient les briques de surface, une simulation courte sur une
cellule et l'ecriture du fichier `.npz`.

## Limites de la V1

- Pas de transport horizontal.
- Pas de capacite RZSM pretraitee dans le paquet compact ; le modele 4 peut
  charger le CSV RZSM du modele 0 avec `--rzsm-csv`, sinon il utilise un
  melange terre/ocean/glace base sur les constantes du modele 0.
- Pas de diffusion du sol : le module du modele 0 est conserve mais pas branche,
  car son interpretation en flux de surface est encore ambigu.
- Pas de recalcul des profils atmospheriques quand `T_surface` change.
- Pas d'ocean dynamique.
