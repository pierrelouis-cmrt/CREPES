# Theorie du modele 4

## Variable prognostique

Le modele 4 calcule une seule variable qui evolue dans le temps :

```text
T_surface(t, lat, lon)
```

La grille de reference est la grille globale 5 degres du paquet compact du
modele 3. Les cellules sont independantes. Cette hypothese signifie que le
modele 4 ne transporte pas d'energie entre deux cellules voisines.

## Bilan d'energie de surface

Pour chaque cellule, l'equation integree est :

```text
C_surface dT_surface/dt =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
  - Q_latent
  - Q_convection
```

Convention de signe :

- un flux positif a droite chauffe la surface ;
- `Q_latent > 0` retire de l'energie a la surface ;
- `Q_convection > 0` retire de l'energie a la surface ;
- si l'air est plus chaud que la surface, `Q_convection` peut devenir negatif
  et rechauffer la surface.

Le modele 3 fournit les trois flux radiatifs :

- `SW_absorbe_surface` ;
- `LW_down_absorbe_surface` ;
- `LW_up_surface`.

Le modele 4 ne decrit pas ici le calcul interne de ces flux radiatifs. Il les
consomme comme flux de colonne locale, puis resout le bilan de surface.

## Court-onde de surface

Le terme court-onde utilise dans le bilan est :

```text
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

`SW_down_surface` garde le cycle jour/nuit et saisonnier de la geometrie
solaire locale, corrige par une transmissivite atmospherique mensuelle issue du
paquet compact :

```text
SW_down_surface = tau_SW_mensuel * S0 * max(cos(i), 0)
```

Le modele 4 ne reintegre pas l'ancien albedo nuageux du modele 0. L'effet moyen
de l'atmosphere sur le court-onde est deja porte par `tau_SW_mensuel`.

## Long-onde de surface

Le bilan utilise deux termes long-onde :

```text
LW_down_absorbe_surface
LW_up_surface
```

`LW_down_absorbe_surface` chauffe la surface. `LW_up_surface` est l'emission de
la surface et refroidit la surface. Dans le modele 0, ce role etait tenu par un
terme atmospherique simplifie `sigma T_atm^4` et par `sigma T_surface^4`.

Le modele 4 remplace cette atmosphere thermique constante par les flux calcules
par la colonne radiative locale. C'est le couplage principal entre les modeles
3 et 4.

## Capacite thermique surfacique

Le modele 0 utilisait :

```text
C = cp * rho_bulk * e
```

avec :

- `rho_bulk = 2600 kg m-3` ;
- `e = 0.5 m` ;
- `cp_sec = 0.8 kJ kg-1 K-1` ;
- `cp_water = 4.187 kJ kg-1 K-1` ;
- `cp_ice = 2.09 kJ kg-1 K-1`.

Quand l'humidite RZSM etait disponible, le modele 0 calculait d'abord une
capacite calorifique effective :

```text
w = rho_w * RZSM / (rho_bulk * (1 - RZSM) + rho_w * RZSM)
cp = cp_sec + w * (cp_water - cp_sec)
C = cp * rho_bulk * e
```

Le paquet compact actuel ne contient pas RZSM. Si le CSV RZSM du modele 0 est
fourni avec `--rzsm-csv`, la partie continentale utilise la formule RZSM. Sinon
la V1 du modele 4 utilise les constantes du modele 0 et les fractions deja
presentes dans le paquet :

```text
C_land  = cp_sec   * rho_bulk * e
C_ocean = cp_water * rho_w    * e
C_ice   = cp_ice   * rho_bulk * e

C_surface =
    f_snow_ice * C_ice
  + (1 - f_snow_ice) * (f_land * C_land + (1 - f_land) * C_ocean)
```

Ce choix est plus coherent pour une grille globale que le fallback sec unique :
les oceans ont une inertie plus grande que les continents, sans introduire un
ocean dynamique.

## Chaleur latente

Le modele 0 exprimait le flux latent moyen avec :

```text
Q_latent = Delta_h_vap * rho_eau * E
```

ou `E` est une hauteur annuelle d'evaporation convertie en m/s. Les valeurs
conservees sont :

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

Dans la V1 globale, le paquet compact fournit une fraction terre/mer mais pas
un continent par cellule. Le modele 4 utilise donc :

```text
Q_land = moyenne des continents non oceaniques
Q_ocean = valeur ocean du modele 0

Q_latent =
  facteur_latent
  * (f_land * Q_land + (1 - f_land) * Q_ocean)
  * (1 - f_snow_ice)
```

Le flux latent est garde positif ou nul. Il represente une perte d'energie de
surface moyenne. La modulation jour/nuit du modele 0 n'est pas reprise telle
quelle, car elle pouvait rendre le flux latent negatif la nuit ; ce serait une
source de chaleur non physique pour cette V1 globale.

## Convection

Le flux convectif est :

```text
Q_convection = h * (T_surface - T_air)
```

Il est positif si la surface est plus chaude que l'air. `T_air` vient de
`temperature_2m_k` quand le paquet compact la fournit ; sinon le modele utilise
`288 K`.

### Convection forcee

La convection forcee reprend la formulation Chevreaux du modele 0 :

```text
Re = rho_air * v * L / mu
Nu = a * Re^m * Pr^(1/3)
h = Nu * lambda_air / L
```

avec deux regimes :

```text
si Re < 5e5 : a = 0.664, m = 0.5
sinon       : a = 0.037, m = 0.8
```

La V1 utilise un vent constant par defaut :

```text
v = 2.5 m/s
```

### Convection naturelle

La convection naturelle reprend la formulation Ornithorynquietant du modele 0 :

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

## Integration temporelle

Le modele 0 utilisait un schema implicite Backward Euler. Le modele 4 conserve
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

La temperature suivante est trouvee par Newton sur :

```text
F(T) = T - T_n - dt / C_surface * B(T)
```

La derivee utilise :

```text
d(LW_up_surface)/dT = 4 * emissivite_surface * sigma * T^3
```

et une derivee numerique pour la convection.

## Moteur rapide

Le moteur rapide conserve la meme equation de bilan, mais change
l'organisation du calcul.

Le moteur complet appelle une colonne du modele 3 dans la boucle temporelle. Le
moteur rapide appelle le modele 3 seulement en phase de pre-calcul mensuel pour
les termes qui varient lentement :

```text
LW_down_absorbe_surface[mois, lat, lon]
albedo_surface[mois, lat, lon]
tau_SW[mois, lat, lon]
T_air[mois, lat, lon]
Q_latent[mois, lat, lon]
C_surface[mois, lat, lon]
```

Pendant la boucle temporelle, le moteur rapide calcule directement :

```text
SW_absorbe_surface(t) =
    S0 * max(cos(i(t)), 0)
    * tau_SW_mensuel
    * (1 - albedo_surface)

LW_up_surface(T) =
    emissivite_surface * sigma * T^4
```

Puis il met a jour toute la grille en une seule operation `numpy`.

Pour eviter une boucle de Newton par cellule, la mise a jour rapide utilise une
linearisation semi-implicite :

```text
T_{n+1} = T_n + dt * B(T_n) / (C + dt * D)
```

avec :

```text
D = d(LW_up_surface)/dT + h_convection
```

Ce schema garde le refroidissement thermique principal stabilise sans rendre le
code difficile a lire. C'est une approximation du moteur complet, pas un
remplacement exact colonne par colonne.

## Donnees d'entree

Le modele 4 charge prioritairement :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
```

Ce paquet donne :

- latitudes et longitudes ;
- profils atmospheriques mensuels pretraites pour les colonnes ;
- albedo de surface ;
- transmissivite court-onde mensuelle ;
- temperature de peau et temperature 2 m pour l'initialisation et la convection ;
- fraction terre/mer et neige/glace ;
- flux ERA5 utiles a la validation.

## Validation attendue

La V1 doit etre jugee progressivement :

- stabilite numerique sur une cellule ;
- stabilite sur une petite grille ;
- comparaison de `T_surface` avec `skin_temperature_k` ;
- comparaison du court-onde net avec les flux ERA5 stockes ;
- comportement saisonnier nord/sud ;
- differences terre/ocean et neige/glace.

## Hors perimetre

Ces elements ne sont pas encore dans le modele 4 :

- transport horizontal ;
- ocean dynamique ;
- diffusion verticale du sol ;
- recalcul des profils atmospheriques quand la surface change ;
- vents spatialement variables ;
- continent explicite par cellule pour le latent.
