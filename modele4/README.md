# Modèle 4 - grille de surface pédagogique forcée par le modèle 3

Le modèle 4 fait évoluer une température de surface sur la grille globale
5 degrés préparée pour le modèle 3. Chaque cellule est indépendante : il n'y a
pas encore de transport horizontal, de circulation atmosphérique ou d'océan
dynamique. C'est un modèle de surface pédagogique forcé par les flux de colonne
du modèle 3, pas un modèle climatique complet.

## Rôle

Le modèle 4 ne recode pas la colonne radiative. Pour chaque cellule et chaque
pas de temps, il appelle le modèle 3, récupère les flux radiatifs de surface et
intègre le bilan d'énergie :

```text
C_surface dT_surface/dt =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
  - Q_latent
  - Q_convection
```

Les termes `Q_convection` et la capacité surfacique reprennent les idées et
constantes du modèle 0, mais dans des fonctions isolées et testables.
`Q_latent` est seulement une paramétrisation annuelle moyenne par type de zone :
elle retire de l'énergie à la surface, mais ne simule pas une évaporation
interactive ni un bilan hydrologique.

## Structure

```text
modele4/
  modele4.py               # moteur, CLI, intégration temporelle
  surface.py               # capacité, chaleur latente, convection
  tests/tester_modele4.py  # tests numériques minimaux
  tests/tester_rapide.py   # tests du moteur rapide
  rapide.py                # moteur rapide vectorisé, sortie 4h par défaut
  lancer.py                # TUI pour choisir le moteur et lancer les cas courants
  README.md
  THEORIE.md
  plan.md
```

## Deux scripts, deux rôles

Le modèle 4 a volontairement deux scripts d'exécution. Ils résolvent le même
bilan de surface, mais pas avec le même niveau de recalcul radiatif.

### `modele4.py` : moteur classique

Rôle : référence physique et numérique.

Techniquement :

- il appelle le modèle 3 dans la boucle de calcul ;
- en mode `temporel`, chaque cellule et chaque pas de temps repassent par la
  colonne radiative locale ;
- il résout le bilan d'énergie avec un schéma implicite et des itérations de
  Newton ;
- il est plus lent, surtout sur la grille globale.

Physiquement :

- les flux radiatifs de surface sont recalculés avec la température de surface
  courante ;
- c'est le moteur à utiliser pour vérifier une petite experience, comparer au
  moteur rapide, ou garder une référence plus proche du couplage avec le modèle
  3 ;
- il reste un modèle local : pas de transport horizontal, pas d'océan dynamique,
  pas de recalcul complet de l'atmosphère 3D.

### `rapide.py` : moteur rapide

Rôle : moteur pratique pour les essais courants et les longues simulations.

Techniquement :

- il appelle le modèle 3 au début pour pré-calculer des champs mensuels ;
- ensuite, la boucle temporelle est vectorisée avec `numpy` sur toute la grille ;
- il ne rappelle plus le modèle 3 à chaque pas ;
- il écrit par défaut une carte toutes les 4 heures ;
- il est beaucoup plus rapide que le moteur classique.

Physiquement :

- l'atmosphère radiative est approximee par des champs mensuels fixes pendant la
  simulation ;
- le court-onde garde le cycle jour/nuit via la géométrie solaire, mais utilise
  une transmissivité mensuelle ;
- le long-onde montant et la convection suivent la température de surface en
  temps réel ;
- c'est une approximation contrôlée du moteur classique, pas une référence
  exacte colonne par colonne.

### Choix recommandé

- Pour travailler, tester des paramètres ou lancer une simulation longue :
  utiliser `modele4.rapide`.
- Pour vérifier la physique locale, comparer les résultats ou valider une
  modification : utiliser `modele4.modele4` sur une petite grille.
- Pour une première utilisation : lancer le TUI.

```bash
./.venv/bin/python -m modele4.lancer
```

## Exécution directe

Depuis la racine du projet, cette commande suffit :

```bash
./.venv/bin/python -m modele4.modele4
```

Par défaut, le modèle produit un diagnostic mensuel global :

- grille complète 5 degrés, donc `36 x 72` cellules ;
- 12 cartes, une par mois ;
- chaque carte applique un seul pas implicite `dt` depuis l'état initial du
  mois avec un court-onde journalier moyen ;
- ce n'est pas une intégration complète de chaque mois ;
- fichier écrit dans `modele4/sorties/simulation_modele4.npz` ;
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

`0.020833333333333332` jour correspond à un seul pas de 1800 s.

Lancer une petite grille de développement :

```bash
./.venv/bin/python -m modele4.modele4 \
  --mode temporel \
  --jours 1 \
  --max-latitudes 4 \
  --max-longitudes 8 \
  --output modele4/sorties/simulation_dev.npz
```

Lancer la grille complète 5 degrés :

```bash
./.venv/bin/python -m modele4.modele4 \
  --mode temporel \
  --jours 1 \
  --output modele4/sorties/simulation_globale_1j.npz
```

La grille complète appelle beaucoup de colonnes radiatives. Pour développer, il
est préférable de commencer avec `--max-latitudes` et `--max-longitudes`.

## Moteur rapide

Le moteur rapide est séparé du moteur complet :

```bash
./.venv/bin/python -m modele4.rapide
```

Par défaut il simule toute la grille pendant `1 jour`, avec `dt = 1800 s`, et
écrit une carte toutes les `4 heures` :

```text
temperature_surface_k[temps_4h, latitude, longitude]
shape = (7, 36, 72)
```

Les sorties correspondent à :

```text
0h, 4h, 8h, 12h, 16h, 20h, 24h
```

Le script commence par appeler le modèle 3 pour pré-calculer les champs
mensuels réutilisés :

- albédo de surface ;
- transmissivité court-onde ;
- long-onde descendant absorbé par la surface ;
- température d'air ;
- flux latent ;
- capacité thermique.

Ensuite, la boucle temporelle est vectorisée avec `numpy` sur toute la grille.
Elle ne rappelle plus le modèle 3 à chaque pas de temps.

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

- `--jours` : durée de simulation, `1` par défaut.
- `--dt` : pas de temps interne, `1800 s` par défaut.
- `--sortie-heures` : fréquence de sauvegarde, `4 h` par défaut.
- `--output` : fichier `.npz` de sortie.
- `--max-latitudes`, `--max-longitudes` : sous-grille de test.
- `--convection`, `--facteur-latent`, `--vent`, `--co2` : mêmes rôles que dans
  le moteur complet.
- `--rzsm-csv` : source RZSM du modèle 0, chargée par défaut pour la capacité
  surfacique ; les constantes ne servent que si RZSM manque.

## Options principales

- `--mode` : `diagnostic-mensuel` par défaut pour 12 cartes à un pas,
  `mensuel` comme alias historique, ou `temporel` pour une intégration
  pas-à-pas.
- `--jours` : durée de simulation en jours.
- `--dt` : pas de temps en secondes, `1800` par défaut.
- `--co2` : concentration CO2 transmise au modèle 3.
- `--temperature-initiale` : valeur imposée partout ; sinon le modèle utilise
  `skin_temperature_k`, puis `temperature_2m_k`, puis `288.15 K`.
- `--frequence-sortie-pas` : fréquence d'écriture de `T_surface`.
- `--convection` : `aucune`, `forcee`, `naturelle`, ou `toutes`.
- `--facteur-latent` : multiplicateur du flux latent ; `0` le désactive.
- `--vent` : vent constant en m/s pour la convection forcée.
- `--max-latitudes`, `--max-longitudes` : sous-grille rapide de développement.
- `--rzsm-csv` : CSV RZSM du modèle 0 pour utiliser la capacité thermique
  issue de l'humidité du sol. Par défaut, le modèle charge
  `modele0_maintenance/ressources/capacite_humidite/average_rzsm_tout.csv`.
  Si la source ou une valeur locale manque, il retombe seulement alors sur
  `CP_SEC`.
- `--no-progress` : désactive la barre de progression console.

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

- `temperature_surface_k[mois, lat, lon]` en mode diagnostic mensuel ;
- `temperature_surface_k[temps, lat, lon]` en mode temporel ;
- `mois` : `1..12` en mode diagnostic mensuel ;
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

Les tests vérifient les briques de surface, les signes de flux, une simulation
courte sur une cellule, une cohérence complet/rapide sur un petit cas et
l'écriture du fichier `.npz`.

## Limites de la V1

- Pas de transport horizontal.
- Mode `diagnostic-mensuel` : 12 diagnostics saisonniers à un pas ; pour une
  vraie intégration temporelle, utiliser `--mode temporel` ou `modele4.rapide`.
- Flux latent : paramétrisation annuelle moyenne par continent/océan, constante
  dans le temps ; ce n'est pas une évaporation interactive.
- Pas de capacité RZSM prétraitée dans le paquet compact ; le modèle 4 charge
  le CSV RZSM du modèle 0 à l'exécution et utilise `CP_SEC` seulement en
  fallback si cette source ou la valeur locale manque.
- Pas de diffusion du sol : le module du modèle 0 est conservé mais pas branché,
  car son interprétation en flux de surface est encore ambigu.
- Pas de recalcul des profils atmosphériques quand `T_surface` change.
- Pas d'océan dynamique.
