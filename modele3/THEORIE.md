# Théorie du modèle 3

Le modèle 3 est une colonne radiative locale :

```text
(colonne, T_surface, CO2_ppm) -> flux radiatifs
```

La température de surface est imposée. Le modèle 3 ne résout pas le bilan
thermique de surface ; il calcule seulement les flux radiatifs pour une colonne
donnée. Le modèle 4 utilisera ces flux pour faire évoluer `T_surface(t)`.

Le modèle 3 est la version finale de la colonne radiative locale : il garde le
noyau radiatif utile des itérations précédentes, avec un court-onde corrigé
par transmissivité ERA5 mensuelle, une émissivité de surface constante et sans
opacité radiative explicite des nuages.

## Héritage du modèle 2.5

Le noyau long-onde vient du modèle 2.5. Il est conservé ici sous une forme plus
stable pour le modèle 4.

Les éléments repris sont :

- loi de Stefan-Boltzmann pour le flux thermique total de surface ;
- intégration de Planck par bande spectrale ;
- propagation infrarouge montante et descendante couche par couche ;
- facteur diffusif `D = 1.66` ;
- bandes CO2 à `15 um` et `4.3 um`, avec découpage cœur/ailes ;
- coefficients CO2/H2O effectifs stockés dans `modele3/physique.py`.

La profondeur optique CO2 conservée est :

```text
tau_CO2_bande = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Puis la transmission et l'émissivité de couche sont calculées par :

```text
transmission = exp(-D * tau_total_bande)
emissivite_couche = 1 - transmission
```

Dans le modèle 3, `tau_total_bande` ne reprend pas les anciennes corrections
nuageuses. Il contient seulement les contributions CO2 et H2O :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande
```

Ainsi, l'héritage du 2.5 concerne le transfert long-onde spectral et la logique
CO2 effective. Le court-onde, la surface, les données et les nuages suivent les
choix propres au modèle 3 décrits plus bas.

## Colonne locale

Le modèle reste local : chaque calcul concerne un point de grille, un mois ou
un jour de l'année, et une colonne verticale déjà préparée.

La colonne est construite en amont depuis la pression de surface locale :

```text
p_edges_hpa = [p_surface_hpa] + niveaux de référence inférieurs à p_surface_hpa
```

Les niveaux de référence hérités du modèle 3 sont conservés :

```text
850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa
```

Une couche dont `p_bas <= p_haut` n'est pas une couche physique. Le générateur
ignore les couches plus fines que `0.1 hPa` avant stockage, car le paquet est
quantifié au dixième d'hPa. Le chargeur expose aussi un diagnostic
`couches_ignorees_non_positives` si une source future contient encore une
couche nulle ou négative.

Les moyennes de température et d'humidité spécifique sont calculées par couche
de pression. Le générateur interpole les profils ERA5 `t` et `q`, puis moyenne
sur l'intervalle de pression de la couche :

```text
T_couche = moyenne_pression(T(p), p_haut, p_bas)
q_couche = moyenne_pression(q(p), p_haut, p_bas)
```

La masse d'air de la couche vient directement de son épaisseur en pression :

```text
delta_p = p_bas - p_haut
masse_air = delta_p / g
```

La masse de vapeur d'eau associée est :

```text
masse_H2O = q_couche * masse_air
```

Dans le modèle 3, ces couches sont stockées dans un paquet compact. Le calcul
radiatif normal ne relit pas les gros fichiers ERA5 bruts.

## Court-onde

La géométrie solaire reste celle du projet :

```text
cos(i) =
    sin(latitude) * sin(declinaison)
  + cos(latitude) * cos(declinaison) * cos(angle_horaire)

SW_TOA_local = 1361 * max(cos(i), 0)
```

Le transfert atmosphère-surface est représenté par une transmissivité mensuelle
issue d'ERA5 :

```text
transmissivite_sw_mensuelle =
    ERA5_SW_down_surface / moyenne_mensuelle(SW_TOA_local)
```

Le flux descendant à la surface et le flux absorbé par la surface sont ensuite :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

Pour un calcul mensuel moyen (`mois` sans `jour_annee` explicite et
`moyenne_journaliere_sw=True`), `SW_TOA_local` est la moyenne mensuelle stockée
dans le paquet. Pour un calcul instantané, le modèle utilise le jour milieu du
mois comme jour représentatif, ce qui garde le cycle jour/nuit mais ne prétend
pas être une moyenne mensuelle.

Cette partie remplace l'ancien court-onde simplifié. Il n'y a plus de mode
court-onde alternatif dans le modèle 3, et il n'y a plus d'albédo nuageux
effectif multiplié explicitement dans la formule.

## Long-onde

La surface émet selon Stefan-Boltzmann :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
epsilon_surface = 0.98
```

L'émissivité de surface est constante dans le modèle 3. Les distinctions entre
terre, océan, neige ou glace ne sont pas reprises.

Le flux infrarouge est traité par bandes spectrales. Pour chaque bande, le flux
de surface est obtenu par intégration de Planck sur l'intervalle de longueurs
d'onde de la bande. Le flux montant est propagé de la surface vers le sommet de
l'atmosphère ; le flux descendant est construit en sens inverse à partir de
l'émission des couches.

Pour chaque couche et chaque bande infrarouge :

```text
tau_CO2 = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
tau_H2O = a_H2O_bande * (masse_H2O / 10 kg m-2)
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
emissivite_couche = 1 - transmission
```

### Sections efficaces implicites

Le modèle ne stocke pas de section efficace spectrale explicite
`sigma(lambda, T, p)`. Dans un calcul spectroscopique complet, la profondeur
optique serait de la forme :

```text
tau_lambda = intégrale(n_gaz * sigma_lambda(T, p) * ds)
```

Ici cette physique est condensée dans les coefficients de bande `a_CO2_bande`
et `a_H2O_bande`. Ces coefficients donnent directement une profondeur optique
effective pour une colonne de référence, puis le modèle la remet à l'échelle
avec la concentration de CO2, l'épaisseur de pression ou la masse de vapeur
d'eau. La section efficace est donc implicite dans `a_bande` ; ce n'est pas une
surface moléculaire unique, ni une grandeur à extrapoler hors du domaine de
validité pédagogique du modèle.

Le facteur diffusif hérité du noyau précédent reste :

```text
D = 1.66
```

Les bandes CO2 conservées du noyau précédent couvrent la bande `15 um`
avec un découpage cœur/ailes, et la bande `4.3 um` avec le même principe. Les
bandes H2O effectives sont :

```text
5.5-7.5 um   : bande vibration-rotation autour de 6.3 um
8-13 um      : absorption faible dans la fenêtre atmosphérique
18-80 um     : domaine rotationnel / far-IR
```

Les coefficients de bandes sont effectifs. Ils gardent un noyau CO2 + H2O
simple et lisible ; ils ne remplacent pas HITRAN, RADIS ou une méthode
correlated-k.

## Vapeur d'eau

ERA5 fournit l'humidité spécifique `q`, c'est-à-dire une masse de vapeur d'eau
par kilogramme d'air humide. Le générateur moyenne `q` dans chaque couche, puis
convertit cette humidité en masse colonne :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

La référence :

```text
MASSE_H2O_REFERENCE_KG_M2 = 10.0
```

fixe l'échelle des coefficients H2O dans :

```text
tau_H2O = a_H2O_bande * (masse_H2O / MASSE_H2O_REFERENCE_KG_M2)
```

Le point important reste celui du modèle 3 : CO2 et H2O ne sont pas propagés
comme deux flux séparés. Leurs opacités sont additionnées avant de calculer la
transmission.

## Nuages

Les nuages ne sont pas un terme radiatif explicite dans le modèle 3. Leur effet
moyen sur le court-onde de surface est inclus dans la transmissivité ERA5
mensuelle :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
```

Dans le long-onde, il n'y a pas de `tau_nuage` :

```text
tau_total = tau_CO2 + tau_H2O
```

Les anciens mécanismes qui associaient `low_cloud_cover`,
`medium_cloud_cover`, `high_cloud_cover` ou `total_cloud_cover` à un albédo
nuageux court-onde ou à une opacité grise long-onde ne sont donc pas repris.

## Flux de sortie

Le modèle renvoie les flux principaux :

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

Le long-onde descendant absorbé par la surface est :

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

L'OLR combine le flux infrarouge qui traverse les bandes traitées et la part du
flux de surface située hors des bandes modélisées.

## Données et limites reprises

Le paquet compact versionné contient les champs nécessaires au calcul normal :
coordonnées, pression de surface, albédo de surface, transmissivité court-onde
mensuelle, couches verticales prétraitées et flux ERA5 de validation.

Ce qui est volontairement absent du modèle 3 :

- pas d'évolution de `T_surface(t)` ;
- pas de transport horizontal ;
- pas de lecture directe ERA5 pendant `calculer_colonne_radiative` ;
- pas de fallback analytique dans le calcul normal ;
- pas d'émissivité variable de surface ;
- pas d'albédo nuageux court-onde explicite ;
- pas d'opacité nuageuse long-onde explicite ;
- pas d'ozone, aérosols, CH4, N2O ou microphysique nuageuse.

Ces absences sont des choix du modèle 3. Elles ne doivent pas être corrigées
en réutilisant les anciennes formules dans la théorie.

## Validation

Le paquet conserve des flux ERA5 mensuels pour comparer les ordres de grandeur.
Le court-onde est volontairement calé sur une transmissivité mensuelle moyenne.
Le long-onde reste un modèle effectif CO2 + H2O. La validation du modèle 3
s'appuie donc sur des comparaisons grille par grille avec ERA5, sans cas local
nommé dans la théorie.
