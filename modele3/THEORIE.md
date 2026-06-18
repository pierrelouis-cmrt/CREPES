# Theorie du modele 3.1

Le modele 3.1 est une colonne radiative locale :

```text
(colonne, T_surface, CO2_ppm) -> flux radiatifs
```

La temperature de surface est imposee. Le modele 3.1 ne resout pas le bilan
thermique de surface ; il calcule seulement les flux radiatifs pour une colonne
donnee. Le modele 4 utilisera ces flux pour faire evoluer `T_surface(t)`.

Le calcul 3.1 garde le noyau radiatif utile du modele 3, mais il remplace les
parties qui avaient ete corrigees ensuite : le court-onde utilise une
transmissivite ERA5 mensuelle, l'emissivite de surface est constante, et les
nuages ne sont plus une opacite radiative explicite.

## Heritage du modele 2.5

Le noyau long-onde vient du modele 2.5. Le modele 3 l'a reutilise pour une
colonne locale, et le modele 3.1 le conserve sous une forme plus stable pour le
modele 4.

Les elements repris sont :

- loi de Stefan-Boltzmann pour le flux thermique total de surface ;
- integration de Planck par bande spectrale ;
- propagation infrarouge montante et descendante couche par couche ;
- facteur diffusif `D = 1.66` ;
- bandes CO2 a `15 um` et `4.3 um`, avec decoupage coeur/ailes ;
- coefficients CO2 effectifs calibres dans le 2.5 sur le forcage
  `280 -> 560 ppm`.

La profondeur optique CO2 conservee est :

```text
tau_CO2_bande = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Puis la transmission et l'emissivite de couche sont calculees par :

```text
transmission = exp(-D * tau_total_bande)
emissivite_couche = 1 - transmission
```

Dans 3.1, `tau_total_bande` ne reprend pas les nuages du modele 3. Il contient
seulement les contributions CO2 et H2O :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande
```

Ainsi, l'heritage du 2.5 concerne le transfert long-onde spectral et la logique
CO2 effective. Le court-onde, la surface, les donnees et les nuages suivent les
choix propres au modele 3.1 decrits plus bas.

## Colonne locale

Le modele reste local : chaque calcul concerne un point de grille, un mois ou
un jour de l'annee, et une colonne verticale deja preparee.

La colonne est construite en amont depuis la pression de surface locale :

```text
p_edges_hpa = [p_surface_hpa] + niveaux de reference inferieurs a p_surface_hpa
```

Les niveaux de reference herites du modele 3 sont conserves :

```text
850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa
```

Les moyennes de temperature et d'humidite specifique sont calculees par couche
de pression. Le generateur interpole les profils ERA5 `t` et `q`, puis moyenne
sur l'intervalle de pression de la couche :

```text
T_couche = moyenne_pression(T(p), p_haut, p_bas)
q_couche = moyenne_pression(q(p), p_haut, p_bas)
```

La masse d'air de la couche vient directement de son epaisseur en pression :

```text
delta_p = p_bas - p_haut
masse_air = delta_p / g
```

La masse de vapeur d'eau associee est :

```text
masse_H2O = q_couche * masse_air
```

Dans 3.1, ces couches sont stockees dans un paquet compact. Le calcul radiatif
normal ne relit pas les gros fichiers ERA5 bruts.

## Court-onde

La geometrie solaire reste celle du projet :

```text
cos(i) =
    sin(latitude) * sin(declinaison)
  + cos(latitude) * cos(declinaison) * cos(angle_horaire)

SW_TOA_local = 1361 * max(cos(i), 0)
```

Le transfert atmosphere-surface est represente par une transmissivite mensuelle
issue d'ERA5 :

```text
transmissivite_sw_mensuelle =
    ERA5_SW_down_surface / moyenne_mensuelle(SW_TOA_local)
```

Le flux descendant a la surface et le flux absorbe par la surface sont ensuite :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

Cette partie remplace le court-onde du modele 3. Il n'y a plus de mode
court-onde alternatif dans 3.1, et il n'y a plus d'albedo nuageux effectif
multiplie explicitement dans la formule.

## Long-onde

La surface emet selon Stefan-Boltzmann :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
epsilon_surface = 0.98
```

L'emissivite de surface est constante dans 3.1. Les distinctions du modele 3
entre terre, ocean, neige ou glace ne sont pas reprises.

Le flux infrarouge est traite par bandes spectrales. Pour chaque bande, le flux
de surface est obtenu par integration de Planck sur l'intervalle de longueurs
d'onde de la bande. Le flux montant est propage de la surface vers le sommet de
l'atmosphere ; le flux descendant est construit en sens inverse a partir de
l'emission des couches.

Pour chaque couche et chaque bande infrarouge :

```text
tau_CO2 = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
tau_H2O = a_H2O_bande * (masse_H2O / 10 kg m-2)
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
emissivite_couche = 1 - transmission
```

Le facteur diffusif herite du modele 3 reste :

```text
D = 1.66
```

Les bandes CO2 conservees de la logique du modele 3 couvrent la bande `15 um`
avec un decoupage coeur/ailes, et la bande `4.3 um` avec le meme principe. Les
bandes H2O effectives sont :

```text
5.5-7.5 um   : bande vibration-rotation autour de 6.3 um
8-13 um      : absorption faible dans la fenetre atmospherique
18-80 um     : domaine rotationnel / far-IR
```

Les coefficients de bandes sont effectifs. Ils gardent un noyau CO2 + H2O
simple et lisible ; ils ne remplacent pas HITRAN, RADIS ou une methode
correlated-k.

## Vapeur d'eau

ERA5 fournit l'humidite specifique `q`, c'est-a-dire une masse de vapeur d'eau
par kilogramme d'air humide. Le generateur moyenne `q` dans chaque couche, puis
convertit cette humidite en masse colonne :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

La reference :

```text
MASSE_H2O_REFERENCE_KG_M2 = 10.0
```

fixe l'echelle des coefficients H2O dans :

```text
tau_H2O = a_H2O_bande * (masse_H2O / MASSE_H2O_REFERENCE_KG_M2)
```

Le point important reste celui du modele 3 : CO2 et H2O ne sont pas propages
comme deux flux separes. Leurs opacites sont additionnees avant de calculer la
transmission.

## Nuages

Les nuages ne sont pas un terme radiatif explicite dans 3.1. Leur effet moyen
sur le court-onde de surface est inclus dans la transmissivite ERA5 mensuelle :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
```

Dans le long-onde, il n'y a pas de `tau_nuage` :

```text
tau_total = tau_CO2 + tau_H2O
```

Les mecanismes du modele 3 qui associaient `low_cloud_cover`,
`medium_cloud_cover`, `high_cloud_cover` ou `total_cloud_cover` a un albedo
nuageux court-onde ou a une opacite grise long-onde ne sont donc pas repris.

## Flux de sortie

Le modele renvoie les flux principaux :

```text
SW_TOA_local
SW_down_surface
SW_absorbe_surface
LW_up_surface
LW_down_surface
LW_down_absorbe_surface
OLR
flux_net_radiatif_surface
diagnostics par bande
```

Le long-onde descendant absorbe par la surface est :

```text
LW_down_absorbe_surface = epsilon_surface * LW_down_surface
```

Le flux net radiatif de surface est :

```text
flux_net_radiatif_surface =
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
```

L'OLR combine le flux infrarouge qui traverse les bandes traitees et la part du
flux de surface situee hors des bandes modelisees.

## Donnees et limites reprises

Le paquet compact versionne contient les champs necessaires au calcul normal :
coordonnees, pression de surface, albedo de surface, transmissivite court-onde
mensuelle, couches verticales pretraitees et flux ERA5 de validation.

Ce qui est volontairement absent de 3.1 :

- pas d'evolution de `T_surface(t)` ;
- pas de transport horizontal ;
- pas de lecture directe ERA5 pendant `calculer_colonne_radiative` ;
- pas de fallback analytique dans le calcul normal ;
- pas d'emissivite variable de surface ;
- pas d'albedo nuageux court-onde explicite ;
- pas d'opacite nuageuse long-onde explicite ;
- pas d'ozone, aerosols, CH4, N2O ou microphysique nuageuse.

Ces absences sont des choix du modele 3.1. Elles ne doivent pas etre corrigees
en reutilisant les anciennes formules du modele 3 dans la theorie.

## Validation

Le paquet conserve des flux ERA5 mensuels pour comparer les ordres de grandeur.
Pour Paris, point de grille `47.5 N, 2.5 E`, juillet, `T_surface = 293 K`,
moyenne journaliere SW :

```text
SW_absorbe_surface   = 190.45 W m-2
ERA5 SW_net_surface  = 188.80 W m-2
LW_down_surface      = 349.91 W m-2
ERA5 LW_down_surface = 364.20 W m-2
OLR modele           = 286.15 W m-2
ERA5 OLR             = 252.90 W m-2
```

Le court-onde est volontairement cale sur une transmissivite mensuelle moyenne.
Le long-onde reste un modele effectif CO2 + H2O. La validation du modele 3,
notamment son ancien exces de court-onde sur Paris, est remplacee par cette
validation 3.1.
