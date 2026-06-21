# Théorie du modèle 4

## Variable prognostique

Le modèle 4 calcule une seule variable qui évolue dans le temps :

```text
T_surface(t, lat, lon)
```

La grille de référence est la grille globale 5 degrés du paquet compact du
modèle 3. Les cellules sont indépendantes. Cette hypothèse signifie que le
modèle 4 ne transporte pas d'énergie entre deux cellules voisines.
Le modèle 4 est donc un modèle de surface pédagogique forcé par des flux de
colonne du modèle 3, pas un modèle climatique complet.

## Bilan d'énergie de surface

Pour chaque cellule, l'équation intégrée est :

```text
C_surface dT_surface/dt =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
  - Q_latent
  - Q_convection
```

Convention de signe :

- un flux positif à droite chauffe la surface ;
- `Q_latent > 0` retire de l'énergie à la surface ;
- `Q_convection > 0` retire de l'énergie à la surface ;
- si l'air est plus chaud que la surface, `Q_convection` peut devenir négatif
  et réchauffer la surface.

Le modèle 3 fournit les trois flux radiatifs :

- `SW_absorbe_surface` ;
- `LW_down_absorbe_surface` ;
- `LW_up_surface`.

Le modèle 4 ne décrit pas ici le calcul interne de ces flux radiatifs. Il les
consomme comme flux de colonne locale, puis résout le bilan de surface.

## Court-onde de surface

Le terme court-onde utilisé dans le bilan est :

```text
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

`SW_down_surface` garde le cycle jour/nuit et saisonnier de la géométrie
solaire locale, corrigé par une transmissivité atmosphérique mensuelle issue du
paquet compact :

```text
SW_down_surface = tau_SW_mensuel * S0 * max(cos(i), 0)
```

Le modèle 4 ne réintègre pas l'ancien albédo nuageux du modèle 0. L'effet moyen
de l'atmosphère sur le court-onde est déjà porté par `tau_SW_mensuel`.

## Long-onde de surface

Le bilan utilise deux termes long-onde :

```text
LW_down_absorbe_surface
LW_up_surface
```

`LW_down_absorbe_surface` chauffe la surface. `LW_up_surface` est l'émission de
la surface et refroidit la surface. Dans le modèle 0, ce rôle était tenu par un
terme atmosphérique simplifié `sigma T_atm^4` et par `sigma T_surface^4`.

Le modèle 4 remplace cette atmosphère thermique constante par les flux calculés
par la colonne radiative locale. C'est le couplage principal entre les modèles
3 et 4.

## Capacité thermique surfacique

Le modèle 0 utilisait :

```text
C = cp * rho_bulk * e
```

avec :

- `rho_bulk = 2600 kg m-3` ;
- `e = 0.5 m` ;
- `cp_sec = 0.8 kJ kg-1 K-1` ;
- `cp_water = 4.187 kJ kg-1 K-1` ;
- `cp_ice = 2.09 kJ kg-1 K-1`.

Quand l'humidité RZSM était disponible, le modèle 0 calculait d'abord une
capacité calorifique effective :

```text
w = rho_w * RZSM / (rho_bulk * (1 - RZSM) + rho_w * RZSM)
cp = cp_sec + w * (cp_water - cp_sec)
C = cp * rho_bulk * e
```

Le paquet compact actuel ne contient pas RZSM. Le modèle 4 charge donc par
défaut le CSV RZSM conservé dans `modele0_maintenance/`, le regrille à 1 degré
comme le modèle 0, puis prend la valeur locale la plus proche. Quand RZSM est
disponible, la capacité est directement :

```text
C_surface = C_RZSM
```

Les constantes ne servent qu'en fallback si la source RZSM ou la valeur locale
manque :

```text
C_surface = cp_sec * rho_bulk * e
```

## Chaleur latente

Le modèle 0 exprimait le flux latent moyen avec :

```text
Q_latent = Delta_h_vap * rho_eau * E
```

où `E` est une hauteur annuelle d'évaporation convertie en m/s. Les valeurs
conservées sont :

```text
Europe         0.49 m/an
North America 0.47 m/an
South America 0.94 m/an
Oceania       0.41 m/an
Africa        0.58 m/an
Asia          0.37 m/an
Ocean         1.40 m/an
Antarctica    0.00 m/an
```

Le modèle 4 reprend la détection continent/océan du modèle 0 avec le shapefile
Natural Earth conservé dans `modele0_maintenance/ressources/carte/`. Le flux
latent d'une cellule est donc la valeur annuelle moyenne du continent trouvé au
centre de la cellule, ou la valeur océan si aucun polygone ne contient ce
point :

```text
continent = detecteur_shapefile(lat, lon) ou Ocean
Q_latent = facteur_latent * Q_latent_continent[continent]
```

Le flux latent est gardé positif ou nul. Il représente une perte d'énergie de
surface moyenne, prescrite par type de zone. Il ne représente pas une
évaporation instantanée réaliste, ne dépend pas de l'humidité locale courante et
ne ferme pas de cycle hydrologique. Comme dans `P_em_surf_evap` du modèle 0, le
flux est forcé à `0` au nord de `75 degrés`.

## Convection

Le flux convectif est :

```text
Q_convection = h * (T_surface - T_air)
```

Il est positif si la surface est plus chaude que l'air. `T_air` vient de
`temperature_2m_k` quand le paquet compact la fournit ; sinon le modèle utilise
`288 K`.

### Convection forcée

La convection forcée reprend la formulation Chevreaux du modèle 0 :

```text
Re = rho_air * v * L / mu
Nu = a * Re^m * Pr^(1/3)
h = Nu * lambda_air / L
```

avec deux régimes :

```text
si Re < 5e5 : a = 0.664, m = 0.5
sinon       : a = 0.037, m = 0.8
```

La V1 utilise un vent constant par défaut :

```text
v = 2.5 m/s
```

### Convection naturelle

La convection naturelle reprend la formulation Ornithorynquietant du modèle 0 :

```text
Ra = g * beta * (T_surface - T_air) * L^3 / nu^2 * Pr
Nu = a * |Ra|^(1/4)
h = Nu * lambda_air / L
```

avec :

```text
a = 0.54 si T_surface >= T_air
a = 0.27 sinon
```

Le mode `--convection toutes` additionne les deux flux.

## Intégration temporelle

Le modèle 0 utilisait un schéma implicite Backward Euler. Le modèle 4 conserve
ce choix :

```text
T_{n+1} = T_n + dt / C_surface * B(T_{n+1})
```

ou :

```text
B(T) =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface(T)
  - Q_latent
  - Q_convection(T)
```

La température suivante est trouvée par Newton sur :

```text
F(T) = T - T_n - dt / C_surface * B(T)
```

La dérivée utilise :

```text
d(LW_up_surface)/dT = 4 * emissivite_surface * sigma * T^3
```

et une dérivée numérique pour la convection.

## Deux moteurs de calcul

Le modèle 4 existe sous deux moteurs :

- `modele4.py`, appelé ici moteur classique ;
- `rapide.py`, appelé ici moteur rapide.

Ils partagent la même variable prognostique `T_surface` et le même bilan
d'énergie de surface. La différence importante n'est donc pas l'équation
globale, mais la manière dont les flux radiatifs sont recalculés pendant la
simulation.

### Moteur classique : référence locale

Le moteur classique appelle la colonne radiative du modèle 3 dans la boucle de
calcul. En mode temporel, pour chaque cellule et chaque pas de temps, il
reconstruit une colonne locale et demande au modèle 3 les flux de surface avec
la température de surface courante.

Techniquement, il utilise :

- une boucle explicite sur les latitudes et longitudes sélectionnées ;
- un appel au modèle 3 pour la cellule traitée ;
- un schéma implicite Backward Euler ;
- des itérations de Newton pour trouver `T_surface(t + dt)`.

Physiquement, cela signifie que le couplage radiatif local est le plus direct
possible dans cette V1. Les flux utilisés dans le bilan sont ceux de la colonne
locale au moment du calcul. Ce moteur sert donc de référence pour :

- tester une cellule ;
- tester une petite grille ;
- comparer une modification du modèle ;
- mesurer l'écart introduit par le moteur rapide.

Il n'est pas pour autant un modèle climatique complet. Les cellules restent
indépendantes et les profils atmosphériques du paquet compact ne sont pas
reconstruits dynamiquement par une circulation globale.

### Diagnostic mensuel : lecture saisonniere à un pas

Le mode CLI `diagnostic-mensuel` produit 12 cartes, une par mois. Chaque carte
part de l'état initial du mois dans le paquet du modèle 3, puis applique un seul
pas implicite `dt_s` avec un court-onde journalier moyen. Les cartes sont donc
des diagnostics saisonniers indépendants :

```text
T_diagnostic_mois = T_initiale_mois + un pas implicite du bilan de surface
```

Ce mode est utile pour inspecter rapidement les flux et l'ordre de grandeur de
la réponse de surface. Il ne simule pas les jours successifs d'un mois et ne
doit pas être interprété comme une intégration mensuelle complète. L'ancien nom
CLI `mensuel` reste accepté comme alias historique, mais les métadonnées de
sortie indiquent `mode_sortie = diagnostic_mensuel_un_pas`.

### Moteur rapide : approximation vectorisée

Le moteur rapide conserve la même équation de bilan, mais change
l'organisation physique et numérique du calcul.

Le moteur complet appelle une colonne du modèle 3 dans la boucle temporelle. Le
moteur rapide appelle le modèle 3 seulement en phase de pré-calcul mensuel pour
les termes qui varient lentement :

```text
LW_down_absorbe_surface[mois, lat, lon]
albedo_surface[mois, lat, lon]
tau_SW[mois, lat, lon]
T_air[mois, lat, lon]
Q_latent[mois, lat, lon]
C_surface[mois, lat, lon]
```

Ces champs mensuels sont ensuite considérés fixes pendant les pas de temps du
mois correspondant. Le moteur rapide ne redemande donc pas au modèle 3 comment
la colonne radiative réagit à chaque nouvelle température de surface.

Pendant la boucle temporelle, le moteur rapide calcule directement :

```text
SW_absorbe_surface(t) =
    S0 * max(cos(i(t)), 0)
    * tau_SW_mensuel
    * (1 - albedo_surface)

LW_up_surface(T) =
    emissivite_surface * sigma * T^4
```

Puis il met à jour toute la grille en une seule opération `numpy`.

Physiquement, les conséquences sont les suivantes :

- le cycle jour/nuit du court-onde est conservé par `cos(i(t))` ;
- l'atténuation court-onde reste mensuelle via `tau_SW_mensuel` ;
- le long-onde descendant reste celui pré-calculé pour le mois ;
- le long-onde montant suit bien `T_surface(t)` via `sigma T^4` ;
- la convection suit aussi `T_surface(t)` ;
- la réponse complète de la colonne atmosphérique au changement de surface
  n'est pas recalculée à chaque pas.

Pour éviter une boucle de Newton par cellule, la mise à jour rapide utilise une
linéarisation semi-implicite :

```text
T_{n+1} = T_n + dt * B(T_n) / (C + dt * D)
```

avec :

```text
D = d(LW_up_surface)/dT + h_convection
```

Ce schéma garde le refroidissement thermique principal stabilisé sans rendre le
code difficile à lire. C'est une approximation du moteur complet, pas un
remplacement exact colonne par colonne.

### Règle d'usage

Le moteur rapide est le moteur de travail courant : il est adapté aux grilles
globales, aux tests de paramètres et aux simulations longues.

Le moteur classique est le moteur de référence : il est adapté aux petites
grilles et aux validations, parce qu'il garde le recalcul radiatif local dans la
boucle de calcul.

## Données d'entrée

Le modèle 4 charge prioritairement :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Ce paquet donne :

- latitudes et longitudes ;
- profils atmosphériques mensuels prétraités pour les colonnes ;
- albédo de surface ;
- transmissivité court-onde mensuelle ;
- température de peau et température 2 m pour l'initialisation et la convection ;
- fraction terre/mer et neige/glace ;
- flux ERA5 utiles à la validation.

## Validation attendue

La V1 doit être jugée progressivement :

- stabilité numérique sur une cellule ;
- stabilité sur une petite grille ;
- comparaison de `T_surface` avec `skin_temperature_k` ;
- comparaison du court-onde net avec les flux ERA5 stockés ;
- comportement saisonnier nord/sud ;
- différences terre/océan et neige/glace.

## Hors périmètre

Ces éléments ne sont pas encore dans le modèle 4 :

- transport horizontal ;
- océan dynamique ;
- évaporation interactive ou bilan hydrologique fermé ;
- intégration complète des mois en mode diagnostic mensuel ;
- diffusion verticale du sol ;
- recalcul des profils atmosphériques quand la surface change ;
- vents spatialement variables ;
- continent explicite par cellule pour le latent.
