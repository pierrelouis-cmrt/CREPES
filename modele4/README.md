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
  tests/tester_rapide.py   # tests du moteur rapide
  rapide.py                # moteur rapide vectorise, sortie 4h par defaut
  lancer.py                # TUI pour choisir le moteur et lancer les cas courants
  README.md
  THEORIE.md
  plan.md
```

## Deux scripts, deux roles

Le modele 4 a volontairement deux scripts d'execution. Ils resolvent le meme
bilan de surface, mais pas avec le meme niveau de recalcul radiatif.

### `modele4.py` : moteur classique

Role : reference physique et numerique.

Techniquement :

- il appelle le modele 3 dans la boucle de calcul ;
- en mode `temporel`, chaque cellule et chaque pas de temps repassent par la
  colonne radiative locale ;
- il resout le bilan d'energie avec un schema implicite et des iterations de
  Newton ;
- il est plus lent, surtout sur la grille globale.

Physiquement :

- les flux radiatifs de surface sont recalcules avec la temperature de surface
  courante ;
- c'est le moteur a utiliser pour verifier une petite experience, comparer au
  moteur rapide, ou garder une reference plus proche du couplage avec le modele
  3 ;
- il reste un modele local : pas de transport horizontal, pas d'ocean dynamique,
  pas de recalcul complet de l'atmosphere 3D.

### `rapide.py` : moteur rapide

Role : moteur pratique pour les essais courants et les longues simulations.

Techniquement :

- il appelle le modele 3 au debut pour pre-calculer des champs mensuels ;
- ensuite, la boucle temporelle est vectorisee avec `numpy` sur toute la grille ;
- il ne rappelle plus le modele 3 a chaque pas ;
- il ecrit par defaut une carte toutes les 4 heures ;
- il est beaucoup plus rapide que le moteur classique.

Physiquement :

- l'atmosphere radiative est approximee par des champs mensuels fixes pendant la
  simulation ;
- le court-onde garde le cycle jour/nuit via la geometrie solaire, mais utilise
  une transmissivite mensuelle ;
- le long-onde montant et la convection suivent la temperature de surface en
  temps reel ;
- c'est une approximation controlee du moteur classique, pas une reference
  exacte colonne par colonne.

### Choix recommande

- Pour travailler, tester des parametres ou lancer une simulation longue :
  utiliser `modele4.rapide`.
- Pour verifier la physique locale, comparer les resultats ou valider une
  modification : utiliser `modele4.modele4` sur une petite grille.
- Pour une premiere utilisation : lancer le TUI.

```bash
./.venv/bin/python -m modele4.lancer
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

## Moteur rapide

Le moteur rapide est separe du moteur complet :

```bash
./.venv/bin/python -m modele4.rapide
```

Par defaut il simule toute la grille pendant `1 jour`, avec `dt = 1800 s`, et
ecrit une carte toutes les `4 heures` :

```text
temperature_surface_k[temps_4h, latitude, longitude]
shape = (7, 36, 72)
```

Les sorties correspondent a :

```text
0h, 4h, 8h, 12h, 16h, 20h, 24h
```

Le script commence par appeler le modele 3 pour pre-calculer les champs
mensuels reutilises :

- albedo de surface ;
- transmissivite court-onde ;
- long-onde descendant absorbe par la surface ;
- temperature d'air ;
- flux latent ;
- capacite thermique.

Ensuite, la boucle temporelle est vectorisee avec `numpy` sur toute la grille.
Elle ne rappelle plus le modele 3 a chaque pas de temps.

Exemples :

```bash
# 1 an, une carte toutes les 4 heures
./.venv/bin/python -m modele4.rapide --jours 365

# 1 an, une carte par jour
./.venv/bin/python -m modele4.rapide --jours 365 --sortie-heures 24

# 1 an, une carte par heure
./.venv/bin/python -m modele4.rapide --jours 365 --sortie-heures 1

# Petite grille de test
./.venv/bin/python -m modele4.rapide --max-latitudes 4 --max-longitudes 8
```

Options principales du moteur rapide :

- `--jours` : duree de simulation, `1` par defaut.
- `--dt` : pas de temps interne, `1800 s` par defaut.
- `--sortie-heures` : frequence de sauvegarde, `4 h` par defaut.
- `--output` : fichier `.npz` de sortie.
- `--max-latitudes`, `--max-longitudes` : sous-grille de test.
- `--convection`, `--facteur-latent`, `--vent`, `--co2` : memes roles que dans
  le moteur complet.
- `--rzsm-csv` : source RZSM du modele 0, chargee par defaut pour la capacite
  surfacique ; les constantes ne servent que si RZSM manque.

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
  issue de l'humidite du sol. Par defaut, le modele charge
  `modele0_maintenance/ressources/capacite_humidite/average_rzsm_tout.csv`.
  Si la source ou une valeur locale manque, il retombe seulement alors sur
  `CP_SEC`.
- `--no-progress` : desactive la barre de progression console.

Exemple avec un CSV RZSM explicite :

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
./.venv/bin/python modele4/tests/tester_rapide.py
```

Les tests verifient les briques de surface, une simulation courte sur une
cellule et l'ecriture du fichier `.npz`.

## Limites de la V1

- Pas de transport horizontal.
- Pas de capacite RZSM pretraitee dans le paquet compact ; le modele 4 charge
  le CSV RZSM du modele 0 a l'execution et utilise `CP_SEC` seulement en
  fallback si cette source ou la valeur locale manque.
- Pas de diffusion du sol : le module du modele 0 est conserve mais pas branche,
  car son interpretation en flux de surface est encore ambigu.
- Pas de recalcul des profils atmospheriques quand `T_surface` change.
- Pas d'ocean dynamique.
