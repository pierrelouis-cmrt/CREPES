# Théorie du modèle 3

Le modèle 3 est une colonne radiative locale. Il répond à la question :

```text
Pour un point local et une température de surface imposée,
quels sont les flux radiatifs montants et descendants ?
```

Il ne résout donc pas le bilan thermique de surface. La température de surface
est une entrée, pas une variable intégrée dans le temps.

Dans le code, les formules physiques élémentaires sont regroupées dans
`physique/calculs.py`. Le fichier `modele3.py` garde la logique de colonne :
construction des couches, propagation des flux, diagnostics et interface CLI.

## Héritage du modèle 2.5

Le modèle 3 reprend le noyau long-onde du modèle 2.5 :

- loi de Stefan-Boltzmann pour le flux thermique total de surface ;
- intégration de Planck par bande spectrale ;
- propagation infrarouge montante et descendante couche par couche ;
- facteur diffusif `D = 1.66` ;
- bandes CO2 à `15 µm` et `4.3 µm`, avec le découpage coeur/ailes ;
- coefficients CO2 effectifs calibrés dans le 2.5 sur le forçage `280 -> 560 ppm`.

La profondeur optique CO2 reste :

```text
tau_CO2_bande = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Puis :

```text
transmission = exp(-D * tau_total_bande)
emissivite = 1 - transmission
```

## Ce que le modèle 3 ajoute

### Colonne locale

Le modèle 2.5 utilisait une colonne moyenne avec `1013.25 hPa` en bas. Le
modèle 3 construit la colonne depuis la pression de surface locale :

```text
p_edges_hpa = [p_surface_hpa] + niveaux de référence inférieurs à p_surface_hpa
```

Les niveaux de référence restent :

```text
850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa
```

Les moyennes de température, d'humidité et de fraction nuageuse sont calculées
par interpolation en pression et intégration pondérée par `dp`.

### Données atmosphériques

Quand les fichiers existent dans `ressources/`, le modèle lit :

- `t` : température sur niveaux de pression ;
- `q` : humidité spécifique ;
- `cc` : fraction nuageuse verticale si disponible ;
- `sp` : pression de surface ;
- `fal` : albédo prévisionnel ;
- `lcc`, `mcc`, `hcc`, `tcc` : couvertures nuageuses ;
- flux ERA5 de validation : `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`.

ERA5 fournit des données mensuelles globales sur grille régulière et 37 niveaux
de pression de `1000 hPa` à `1 hPa` pour les champs de pression. La définition
ERA5 de `q` est bien une masse de vapeur d'eau par kilogramme d'air humide, ce
qui justifie son usage direct dans une masse colonne `q * delta_p / g`.

### Vapeur d'eau

Pour une couche :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
facteur_humidite = masse_H2O / masse_H2O_reference
tau_H2O_bande = a_H2O_bande * facteur_humidite
```

Le point important du plan est respecté : les flux CO2 et H2O ne sont pas
calculés séparément. Les opacités sont additionnées avant la transmission :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande + tau_nuage
```

Les bandes H2O ajoutées sont volontairement simples :

- `5.5-7.5 µm` pour la bande vibration-rotation autour de `6.3 µm` ;
- `8-13 µm` pour une absorption effective faible dans la fenêtre atmosphérique ;
- `18-80 µm` pour le domaine rotationnel/far-IR.

Les coefficients H2O sont effectifs. Ils ne prétendent pas remplacer une base
ligne par ligne comme HITRAN ; ils servent à obtenir un ordre de grandeur
physique dans ce modèle pédagogique.

### Nuages

Le modèle reste minimal :

- `low_cloud_cover` est placé dans les couches basses ;
- `medium_cloud_cover` dans les couches moyennes ;
- `high_cloud_cover` dans les couches hautes ;
- `total_cloud_cover` sert de secours.

Les seuils suivent l'esprit des définitions ERA5 : bas au-dessus de `0.8 p_s`,
moyen entre `0.45 p_s` et `0.8 p_s`, haut sous `0.45 p_s`.

Effet court-onde :

```text
albedo_cloud = coefficient_cloud_sw * cloud_fraction_total
SW_absorbe_surface = SW_incident_surface * (1 - albedo_surface) * (1 - albedo_cloud)
```

Effet long-onde :

```text
tau_total = tau_CO2 + tau_H2O + tau_nuage
```

Le nuage est donc une opacité grise effective dans les bandes traitées. Il n'y a
pas de microphysique, pas de contenu liquide/glace, pas de taille de gouttes.

### Surface

La surface fournit seulement :

```text
T_surface
epsilon_surface
albedo_surface
```

Le flux émis par la surface est :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
```

Le long-onde descendant absorbé par la surface est :

```text
LW_down_absorbe_surface = epsilon_surface * LW_down_surface
```

L'émissivité suit une version simplifiée du plan :

- terre : valeur de l'extrait JSON si elle existe, sinon secours `0.98` ;
- océan : `0.985` ;
- neige/glace : `0.98` ;
- secours : `0.98`.

La lecture directe des fichiers MODIS HDF4 a été retirée de cette version pour
garder le modèle plus lisible. Si on veut utiliser MODIS, le plus simple est de
préparer l'émissivité en amont et de l'écrire dans l'extrait JSON.

### Solaire

La géométrie solaire est reprise du modèle 0 :

```text
declinaison = 23.44 deg * sin(2*pi*(284 + jour)/365)
cos_incidence = sin(lat)*sin(declinaison)
              + cos(lat)*cos(declinaison)*cos(angle_horaire)
SW_incident_surface = S0 * max(cos_incidence, 0)
```

Par défaut, le modèle utilise l'heure solaire fournie. L'option CLI
`--moyenne-journaliere-sw` moyenne cette même formule sur 24 h pour comparer
plus proprement un ordre de grandeur avec les flux mensuels ERA5.

## Flux de sortie

Le modèle renvoie :

```text
SW_incident_surface
SW_absorbe_surface
LW_up_surface
LW_down_surface
LW_down_absorbe_surface
OLR
flux_net_radiatif_surface
diagnostics par couche et par bande
```

Le flux net radiatif de surface est :

```text
flux_net_radiatif_surface =
    SW_absorbe_surface
  + epsilon_surface * LW_down_surface
  - LW_up_surface
```

## Stratégie pour les gros fichiers

Les gros fichiers de `ressources/` ne sont pas suivis par Git. Le modèle 3 est
donc capable de fonctionner de trois façons :

1. lecture directe des NetCDF ERA5 locaux si `ressources/` est présent ;
2. lecture d'un extrait JSON compact produit par `preparer_point.py` ;
3. fallback analytique simple si aucune donnée locale n'est disponible.

Le cas versionné `donnees_exemple/paris_2024_m07.json` contient les valeurs
ERA5 déjà extraites pour Paris en juillet, plus les paramètres de surface utiles
au calcul. Il permet de pousser sur GitHub un cas reproductible sans pousser les
fichiers originaux.

## Validation minimale

Cas Paris, juillet, `T_surface = 293 K`, moyenne journalière SW :

| Flux | Modèle 3 W/m² | ERA5 W/m² | Commentaire |
| --- | ---: | ---: | --- |
| `LW_down_surface` | 350.38 | 361.32 | Bon ordre de grandeur. |
| `OLR` | 244.78 | 244.88 | Très proche pour ce cas calibré. |
| `SW_absorbe_surface` | 268.53 | 178.70 | Trop élevé : absorption/diffusion SW atmosphérique absente. |

`SW` signifie `ShortWave`, c'est-à-dire le rayonnement solaire court-onde. La
valeur importante à retenir pour le cas Paris est :

```text
SW_absorbe_surface_modele3 = 268.53 W/m2
SW_net_surface_ERA5        = 178.70 W/m2
ecart                      = +89.83 W/m2
```

Le modèle 3 surestime donc le court-onde net de surface. C'est attendu dans
cette version, car le calcul solaire est volontairement minimal :

```text
SW_absorbe_surface =
    S0 * moyenne(max(cos_incidence, 0))
  * (1 - albedo_surface)
  * (1 - albedo_cloud_effectif)
```

ERA5 ne calcule pas ce flux avec cette approximation. Son `avg_snswrf` est un
flux court-onde net à la surface issu d'un transfert radiatif atmosphérique
complet et d'une moyenne mensuelle : absorption par gaz atmosphériques,
diffusion, aérosols, opacité nuageuse, recouvrement nuageux, variabilité
horaire et conditions météo du mois. Le modèle 3 ne contient pas encore ces
processus court-onde. Pour le modèle 4, cet écart devra être traité avant
d'intégrer la température de surface, sinon le bilan radiatif chauffera trop la
surface.

Cette validation est qualitative. Elle ne doit pas être lue comme une
reconstruction ERA5.

## Sources

- Copernicus Climate Data Store, ERA5 monthly averaged data on pressure levels :
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means
- Copernicus Climate Data Store, ERA5 monthly averaged data on single levels :
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
- NASA Earthdata / LP DAAC, MOD11C3 v061, Land Surface Temperature/Emissivity :
  https://lpdaac.usgs.gov/products/mod11c3v061/
- IPCC AR6 WGI, chapitre 7, bilan énergétique, vapeur d'eau, nuages et forçages :
  https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/
- HITRAN, base de référence pour les paramètres spectroscopiques atmosphériques :
  https://hitran.org/
- Myhre et al. (1998), formule logarithmique CO2 classique :
  https://doi.org/10.1029/98GL01908
- Gordon et al. (2022), HITRAN2020 :
  https://doi.org/10.1016/j.jqsrt.2021.107949
