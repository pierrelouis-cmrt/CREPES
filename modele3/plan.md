# Plan modèle 3 - colonne radiative locale

## Objectif

Le modèle 3 doit être une **colonne radiative locale** pour un point
`(latitude, longitude)` donné.

Il représente une seule future cellule du maillage terrestre, mais il ne gère
pas encore la grille globale et ne fait pas évoluer la température de surface
dans le temps.

Pour le premier cas de référence, on va utiliser :

```text
Paris : lat = 48.8566, lon = 2.3522
```

Le modèle 3 répond à la question :

```text
Pour cette colonne locale et pour une température de surface imposée,
quels sont les flux radiatifs montants et descendants ?
```

Il doit donc rester une évolution directe du modèle 2.5 :

- modèle 2.5 : colonne CO2 globale moyenne, atmosphère standard, flux IR ;
- modèle 3 : colonne locale, atmosphère issue de données, CO2 + H2O simple,
  nuages simples, émissivité de surface, flux radiatifs.

/!\ L'évolution de `T_surface(t)` sera réservée au modèle 4.

## Principe physique

Le modèle 3 ne résout pas de bilan énergétique de surface. Il prend
`T_surface` comme une entrée.

Entrées principales :

```text
lat, lon
jour ou mois
T_surface
p_surface
profil T(p)
profil q(p)
CO2_ppm
nébulosité simple
albedo_surface
emissivite_surface
```

Sorties principales :

```text
SW_incident_surface
SW_absorbe_surface
LW_up_surface
LW_down_surface
OLR
flux_net_radiatif_surface
diagnostics par couche et par bande
```

Le flux net radiatif de surface peut être défini comme :

```text
flux_net_radiatif_surface =
    SW_absorbe_surface
  + epsilon_surface * LW_down_surface
  - LW_up_surface
```

avec :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
```

## Données disponibles

Le modèle doit être écrit pour exploiter les données disponibles quand elles
sont présentes, avec des valeurs de secours simples quand elles manquent.

### Données atmosphériques verticales

Variables attendues par mois, latitude, longitude et niveau de pression :

- température atmosphérique `T(p)` ;
- humidité spécifique `q(p)` ;
- fraction nuageuse verticale si disponible ;
- géopotentiel si nécessaire pour diagnostic.

Usage :

- remplacer l'atmosphère standard 1976 du modèle 2.5 ;
- construire des températures moyennes par couche ;
- construire une humidité moyenne par couche ;
- estimer une opacité H2O simple.

### Données de surface

Variables attendues par mois, latitude et longitude :

- pression de surface ;
- albédo ou albédo prévisionnel ;
- couverture nuageuse totale, basse, moyenne, haute ;
- masque terre-mer ;
- neige/glace si disponible ;
- température de peau ou température à 2 m pour initialisation/validation
  seulement.

Important : `skin temperature` ne doit pas être utilisée comme température
imposée obligatoire. Elle sert à tester et comparer le modèle.

### Émissivité MODIS

Variables utiles :

- `Emis_31` ;
- `Emis_32`.

Approximation initiale :

```text
epsilon_surface_land = moyenne(Emis_31, Emis_32)
epsilon_ocean = 0.985
epsilon_snow_ice = 0.98
```

Si les fichiers MODIS ne sont pas disponibles, on prend une valeur de backup :

```text
epsilon_surface = 0.98
```

## Construction de la colonne

La colonne doit dépendre de la pression de surface locale. Faut pas garder
`1013.25 hPa` fixe comme base.

Niveaux de référence :

```text
p_edges_ref_hpa = [850, 700, 500, 300, 200, 100, 50, 20, 10, 1]
```

Pour une colonne donnée :

```text
p_edges_hpa = [p_surface_hpa] + niveaux de référence strictement inférieurs à p_surface_hpa
```

Exemple basse altitude :

```text
[1010, 850, 700, 500, 300, 200, 100, 50, 20, 10, 1]
```

Exemple montagne :

```text
[750, 700, 500, 300, 200, 100, 50, 20, 10, 1]
```

Chaque couche doit contenir :

```text
p_bas
p_haut
delta_p = p_bas - p_haut
T_moyen
q_moyen
CO2_ppm
cloud_fraction éventuelle
```

Si `delta_p <= 0`, la couche est ignorée.

Les moyennes `T_moyen` et `q_moyen` doivent être obtenues par interpolation ou
moyenne pondérée en pression depuis les niveaux disponibles.

## CO2

Pour la première version :

```text
CO2_ppm = 420
```

La valeur doit être configurable, mais le profil vertical peut rester uniforme.

Opacité CO2 :

```text
tau_CO2_bande = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Les coefficients `a_CO2_bande` restent ceux du modèle 2.5 pour l'instant. Ne
pas faire RADIS/HITRAN dans cette version.

## Vapeur d'eau

Utiliser l'humidité spécifique `q`, en `kg/kg`.

Pour chaque couche :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

Ajouter une opacité H2O effective simple.

Forme minimale :

```text
tau_H2O_bande = a_H2O_bande * facteur_humidite
```

Le choix exact de `facteur_humidite` peut être simple au départ, par exemple
proportionnel à `masse_H2O` normalisée par une masse de référence (à justifier proprement avec des sources claires).

Règle importante :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande
transmission = exp(-D * tau_total_bande)
emissivite = 1 - transmission
```

Ne pas additionner des flux CO2 et H2O calculés séparément. Les opacités doivent
être additionnées avant le calcul de transmission.

## Nuages

Rester minimal.

Utiliser selon disponibilité :

- `low_cloud_cover` ;
- `medium_cloud_cover` ;
- `high_cloud_cover` ;
- sinon `total_cloud_cover`.

Placement initial :

```text
low cloud    -> couche basse
medium cloud -> couche moyenne
high cloud   -> couche haute
```

Effet court-onde simple :

```text
albedo_cloud = coefficient_cloud_sw * cloud_fraction
SW_absorbe_surface = SW_incident_surface * (1 - albedo_surface) * (1 - albedo_cloud)
```

Effet long-onde simple :

- augmenter l'émissivité effective de la couche correspondante ;
- utiliser la température de cette couche pour l'émission du nuage ;
- ne pas modéliser la microphysique.

## Solaire

Reprendre la géométrie solaire du modèle 0 :

- jour de l'année ;
- heure solaire locale ;
- latitude ;
- longitude ;
- déclinaison ;
- cosinus positif de l'incidence solaire.

Flux incident simplifié :

```text
SW_incident_surface = S0 * max(cos_incidence, 0)
```

Le modèle 3 peut ensuite appliquer l'albédo de surface et l'albédo des nuages
pour produire `SW_absorbe_surface`.

## Surface

La surface n'est pas intégrée thermiquement dans le modèle 3. Elle fournit
seulement :

```text
T_surface
epsilon_surface
albedo_surface
```

Flux thermique émis :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
```

Dans les bandes spectrales, appliquer aussi `epsilon_surface` au flux de Planck
émis par la surface.

Absorption du long-onde descendant par la surface :

```text
LW_down_absorbe_surface = epsilon_surface * LW_down_surface
```

## Temporalité

Le modèle doit pouvoir être appelé pour :

- un mois donné ;
- ou un jour donné avec interpolation mensuelle simple.

Dans la première version, une interpolation linéaire ou l'utilisation directe de
la valeur mensuelle suffit.

Le modèle 3 ne boucle pas nécessairement sur toute l'année. Il doit surtout
exposer une fonction colonne réutilisable.

## Validation minimale

Pour Paris, avec une température de surface imposée raisonnable, comparer les
ordres de grandeur avec les données de validation disponibles :

- `LW_down_surface` contre le flux long-onde descendant ERA5 ;
- `SW_absorbe_surface` contre le flux court-onde net ERA5 ;
- `OLR` contre le flux long-onde au sommet ERA5 si disponible.

La validation attendue est qualitative et en ordre de grandeur. Le but est
d'obtenir une colonne locale propre et explicable, pas une réanalyse parfaite.

## Hors périmètre

Ne pas faire dans le modèle 3 :

- évolution de `T_surface(t)` ;
- grille mondiale ;
- échanges horizontaux ;
- bilan thermique complet de surface ;
- dynamique atmosphérique ;
- rétroaction de la surface sur `T(p)` ;
- RADIS/HITRAN ;
- correlated-k ;
- ozone, CH4, N2O ;
- microphysique détaillée des nuages.

## Priorité d'implémentation

1. Reprendre le noyau radiatif du modèle 2.5.
2. Permettre un appel pour un point `(lat, lon)`.
3. Construire les couches depuis `p_surface`.
4. Remplacer le profil standard par `T(p)` local.
5. Ajouter `q(p)` et une opacité H2O simple.
6. Ajouter l'émissivité de surface.
7. Ajouter les nuages simples.
8. Sortir des flux clairs et des diagnostics.
9. Comparer quelques flux à ERA5.

## Lien avec le modèle 4

Le modèle 4 utilisera le modèle 3 comme module radiatif. Il appellera la colonne
avec une température de surface courante, récupérera les flux, puis intégrera le
bilan de surface sur une grille terrestre.

## Peut être un futur modèle 3.1

- améliorer la temporalité, jour par jour
- améliorer la calibration des coef optiques pour CO2 et H2O
