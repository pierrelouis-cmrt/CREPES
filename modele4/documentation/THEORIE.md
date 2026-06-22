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

Le modele separe trois effets :

1. la geometrie solaire locale ;
2. l'attenuation moyenne par l'atmosphere ;
3. la reflexion par l'albedo de surface.

Le flux solaire au sommet de l'atmosphere est calcule avec la geometrie du
projet :

```text
SW_TOA_local(t) = S0 * max(cos(i(t)), 0)
```

`tau_SW_mensuel` n'est pas une epaisseur optique. C'est une transmissivite
effective mensuelle, donc un rapport de flux compris entre 0 et 1 :

- `tau_SW_mensuel = 1` : toute la lumiere court-onde arrive a la surface ;
- `tau_SW_mensuel = 0` : rien n'arrive a la surface ;
- en pratique, il contient l'effet moyen des nuages, de la diffusion, des
  aerosols et de l'absorption atmospherique.

On l'utilise parce que `S0 * max(cos(i), 0)` donne un flux au sommet de
l'atmosphere, alors que le bilan de surface a besoin du flux descendant a la
surface. Le facteur `tau_SW_mensuel` permet de garder le cycle jour/nuit du
modele tout en ramenant le niveau moyen vers ERA5.

Dans le paquet compact, il est determine avant la simulation :

```text
tau_SW_mensuel =
    era5_sw_down_surface_w_m2
  / moyenne_mensuelle(S0 * max(cos(i), 0))
```

Le rapport est mis a 0 quand il n'y a pas de soleil mensuel, puis borne entre 0
et 1. Le modele 4 ne recalcule pas ce facteur : il le lit dans le paquet modele
3 sous le nom `transmissivite_sw_mensuelle`.

Le flux descendant a la surface est alors :

```text
SW_down_surface(t) = tau_SW_mensuel * SW_TOA_local(t)
```

L'albedo intervient seulement apres, car il represente la part reflechie par la
surface, pas l'attenuation de l'atmosphere. Le flux absorbe par la surface est :

```text
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

Etape par etape :

- calculer `SW_TOA_local(t)` avec la position du Soleil ;
- multiplier par `tau_SW_mensuel` pour obtenir le court-onde descendant a la
  surface ;
- lire `albedo_surface` dans le paquet compact ;
- corriger seulement les albedos nuls sur neige/glace avec un repli physique ;
- absorber la fraction `1 - albedo_surface`.

Le modele 4 ne reintegre pas l'ancien albedo nuageux du modele 0. L'effet moyen
des nuages sur le court-onde descendant est deja porte par `tau_SW_mensuel`.
Ajouter un albedo nuageux explicite ici compterait deux fois cet effet.

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
- `rho_w = 1000 kg m-3` ;
- `rho_ice = 917 kg m-3` ;
- `e = 0.5 m` ;
- `cp_sec = 0.8 kJ kg-1 K-1` ;
- `cp_water = 4.187 kJ kg-1 K-1` ;
- `cp_ice = 2.09 kJ kg-1 K-1`.

Provenance : `rho_w`, `rho_bulk`, `e`, `cp_sec`, `cp_water` et `cp_ice`
sont repris de `modele0_maintenance/codes_python/physique/capacite_surface.py`.
La valeur `rho_ice = 917 kg m-3` est la densite usuelle de la glace pure,
ajoutee ici pour ne pas traiter les cellules glace/neige comme du sol humide.

Quand l'humidite RZSM etait disponible, le modele 0 calculait d'abord une
capacite calorifique effective :

```text
w = rho_w * RZSM / (rho_bulk * (1 - RZSM) + rho_w * RZSM)
cp = cp_sec + w * (cp_water - cp_sec)
C = cp * rho_bulk * e
```

Le paquet compact actuel ne contient pas RZSM. Le modele 4 charge donc par
defaut le CSV RZSM conserve dans `modele0_maintenance/`, le regrille a 1 degre
comme le modele 0, puis prend la valeur locale la plus proche.

Dans le modele 4, cette capacite RZSM est appliquee a la fraction de terre non
enneigee/glacee. Les fractions `land_fraction` et `snow_ice_fraction` viennent
du paquet modele 3. La capacite finale reste volontairement simple :

```text
C_surface =
    f_glace_neige * C_glace_neige
  + (1 - f_glace_neige) * f_terre * C_terre
  + (1 - f_glace_neige) * (1 - f_terre) * C_ocean
```

avec :

```text
C_terre = C_RZSM si RZSM est fini, sinon cp_sec * rho_bulk * 0.5 m
C_ocean = cp_water * rho_w * 1 m
C_glace_neige = cp_ice * rho_ice * 1 m
```

L'epaisseur active `1 m` pour l'ocean et la glace/neige est un choix d'ordre de
grandeur pedagogique. Elle donne a l'eau et a la glace une inertie de surface
plus plausible qu'un fallback de sol sec, sans pretendre representer une couche
melangee oceanique ni la diffusion verticale complete. Le modele reste local :
ce parametre sert seulement a stabiliser et hierarchiser la reponse thermique.

Le fallback sec ne concerne donc plus l'ocean : il sert seulement pour la part
terrestre quand la source RZSM ou la valeur locale manque.

```text
C_sec = cp_sec * rho_bulk * e
```

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

Le modele 4 reprend la detection continent/ocean du modele 0 avec le shapefile
Natural Earth conserve dans `modele0_maintenance/ressources/carte/`. Le flux
latent d'une cellule est donc la valeur du continent trouve au centre de la
cellule, ou la valeur ocean si aucun polygone ne contient ce point :

```text
continent = detecteur_shapefile(lat, lon) ou Ocean
Q_latent = facteur_latent * Q_latent_continent[continent]
```

Le flux latent est garde positif ou nul. Il represente une perte d'energie de
surface moyenne. Comme dans `P_em_surf_evap` du modele 0, le flux est force a
`0` au nord de `75 degres`.

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

## Deux moteurs de calcul

Le modele 4 existe sous deux moteurs :

- `modele4.py`, appele ici moteur classique ;
- `rapide.py`, appele ici moteur rapide.

Ils partagent la meme variable prognostique `T_surface` et le meme bilan
d'energie de surface. La difference importante n'est donc pas l'equation
globale, mais la maniere dont les flux radiatifs sont recalcules pendant la
simulation.

### Moteur classique : reference locale

Le moteur classique appelle la colonne radiative du modele 3 dans la boucle de
calcul. En mode temporel, pour chaque cellule et chaque pas de temps, il
reconstruit une colonne locale et demande au modele 3 les flux de surface avec
la temperature de surface courante.

Techniquement, il utilise :

- une boucle explicite sur les latitudes et longitudes selectionnees ;
- un appel au modele 3 pour la cellule traitee ;
- un schema implicite Backward Euler ;
- des iterations de Newton pour trouver `T_surface(t + dt)`.

Physiquement, cela signifie que le couplage radiatif local est le plus direct
possible dans cette V1. Les flux utilises dans le bilan sont ceux de la colonne
locale au moment du calcul. Ce moteur sert donc de reference pour :

- tester une cellule ;
- tester une petite grille ;
- comparer une modification du modele ;
- mesurer l'ecart introduit par le moteur rapide.

Il n'est pas pour autant un modele climatique complet. Les cellules restent
independantes et les profils atmospheriques du paquet compact ne sont pas
reconstruits dynamiquement par une circulation globale.

### Moteur rapide : approximation vectorisee

Le moteur rapide conserve la meme equation de bilan, mais change
l'organisation physique et numerique du calcul.

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

Ces champs mensuels sont ensuite consideres fixes pendant les pas de temps du
mois correspondant. Le moteur rapide ne redemande donc pas au modele 3 comment
la colonne radiative reagit a chaque nouvelle temperature de surface.

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

Physiquement, les consequences sont les suivantes :

- le cycle jour/nuit du court-onde est conserve par `cos(i(t))` ;
- l'attenuation court-onde reste mensuelle via `tau_SW_mensuel` ;
- le long-onde descendant reste celui pre-calcule pour le mois ;
- le long-onde montant suit bien `T_surface(t)` via `sigma T^4` ;
- la convection suit aussi `T_surface(t)` ;
- la reponse complete de la colonne atmospherique au changement de surface
  n'est pas recalculee a chaque pas.

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

### Regle d'usage

Le moteur rapide est le moteur de travail courant : il est adapte aux grilles
globales, aux tests de parametres et aux simulations longues.

Le moteur classique est le moteur de reference : il est adapte aux petites
grilles et aux validations, parce qu'il garde le recalcul radiatif local dans la
boucle de calcul.

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
