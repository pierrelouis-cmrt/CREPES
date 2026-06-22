# Theorie du modele 3

Le modele 3 est une colonne radiative locale :

```text
(colonne, T_surface, CO2_ppm) -> flux radiatifs
```

La temperature de surface est imposee. Le modele 3 ne resout pas le bilan
thermique de surface ; il calcule seulement les flux radiatifs pour une colonne
donnee. Le modele 4 utilisera ces flux pour faire evoluer `T_surface(t)`.

Le modele 3 est la version finale de la colonne radiative locale : il garde le
noyau radiatif utile des iterations precedentes, avec un court-onde corrige
par transmissivite ERA5 mensuelle, une emissivite de surface constante et sans
opacite radiative explicite des nuages.

## Heritage du modele 2.5

Le noyau long-onde vient du modele 2.5. Il est conserve ici sous une forme plus
stable pour le modele 4.

Les elements repris sont :

- loi de Stefan-Boltzmann pour le flux thermique total de surface ;
- integration de Planck par bande spectrale ;
- propagation infrarouge montante et descendante couche par couche ;
- facteur diffusif `D = 1.66` ;
- bandes CO2 a `15 um` et `4.3 um`, avec decoupage coeur/ailes ;
- coefficients CO2 effectifs herites du 2.5, avec une methode de recalibrage
  HITRAN/RADIS dediee dans `CALIBRAGE_COEFFICIENTS_CO2.md`.

La profondeur optique CO2 conservee est :

```text
tau_CO2_bande = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Puis la transmission et l'emissivite de couche sont calculees par :

```text
transmission = exp(-D * tau_total_bande)
emissivite_couche = 1 - transmission
```

Dans le modele 3, `tau_total_bande` ne reprend pas les anciennes corrections
nuageuses. Il contient seulement les contributions CO2 et H2O :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande
```

Ainsi, l'heritage du 2.5 concerne le transfert long-onde spectral et la logique
CO2 effective. Le court-onde, la surface, les donnees et les nuages suivent les
choix propres au modele 3 decrits plus bas.

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

Dans le modele 3, ces couches sont stockees dans un paquet compact. Le calcul
radiatif normal ne relit pas les gros fichiers ERA5 bruts.

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

Cette partie remplace l'ancien court-onde simplifie. Il n'y a plus de mode
court-onde alternatif dans le modele 3, et il n'y a plus d'albedo nuageux
effectif multiplie explicitement dans la formule.

### Albedo neige/glace en nuit polaire

Les CSV d'albedo herites du modele 0 viennent d'un rapport mensuel
`SW_UP / SW_DOWN`. Lorsque `SW_DOWN` est nul ou quasi nul pendant la nuit
polaire, l'albedo n'est pas observable par ce rapport. Les CSV peuvent alors
porter `0`, ce qui serait physiquement faux pour une maille neigeuse ou glacee
et peut contaminer une interpolation journaliere autour des mois polaires.

Le modele 3 garde une correction limitee au paquet de donnees :

```text
si albedo_surface == 0 et snow_ice_fraction > 0.05 :
    albedo_surface =
        0.30 + snow_ice_fraction * (0.65 - 0.30)
```

`0.30` est le repli de surface general deja utilise par le modele. `0.65`
represente une surface dominee par neige/glace a l'echelle pedagogique du
modele. La correction n'est pas appliquee aux surfaces sans neige/glace.

## Long-onde

La surface emet selon Stefan-Boltzmann :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
epsilon_surface = 0.98
```

L'emissivite de surface est constante dans le modele 3. Les distinctions entre
terre, ocean, neige ou glace ne sont pas reprises.

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

### Sections efficaces implicites

Le modele ne stocke pas de section efficace spectrale explicite
`sigma(lambda, T, p)`. Dans un calcul spectroscopique complet, la profondeur
optique serait de la forme :

```text
tau_lambda = integrale(n_gaz * sigma_lambda(T, p) * ds)
```

Ici cette physique est condensee dans les coefficients de bande `a_CO2_bande`
et `a_H2O_bande`. Ces coefficients donnent directement une profondeur optique
effective pour une colonne de reference, puis le modele la remet a l'echelle
avec la concentration de CO2, l'epaisseur de pression ou la masse de vapeur
d'eau. La section efficace est donc implicite dans `a_bande` ; ce n'est pas une
surface moleculaire unique, ni une grandeur a extrapoler hors du domaine de
calibration du modele.

Les unites et references internes sont :

| Coefficient | Unite dans le modele | Origine projet | Cible / role |
| --- | --- | --- | --- |
| `a_CO2_bande` | profondeur optique effective sans dimension pour `CO2 = 280 ppm` et `delta_p = 101325 Pa` | noyau long-onde du modele 2.5, repris dans le modele 3 | conserver l'ordre de grandeur pedagogique du forcage relatif `280 -> 560 ppm` |
| `a_H2O_bande` | profondeur optique effective sans dimension pour `10 kg m-2` de vapeur d'eau | ajout modele 3 branche sur les masses H2O issues d'ERA5 | representer les grandes bandes H2O : 6.3 um, fenetre 8-13 um, far-IR |

Dans `physique.py`, `ECHELLE_OPACITE_CO2 = 0.0327228010` est donc un facteur
d'echelle effectif du noyau CO2, pas une section efficace. De meme, les valeurs
H2O `25.60`, `0.48` et `14.40` fixent des profondeurs optiques de grandes
bandes pour une masse colonne de reference ; elles ne codent pas des raies
spectrales individuelles.

Le facteur diffusif herite du noyau precedent reste :

```text
D = 1.66
```

Les bandes CO2 conservees du noyau precedent couvrent la bande `15 um`
avec un decoupage coeur/ailes, et la bande `4.3 um` avec le meme principe. Les
bandes H2O effectives sont :

```text
5.5-7.5 um   : bande vibration-rotation autour de 6.3 um
8-13 um      : absorption faible dans la fenetre atmospherique
18-80 um     : domaine rotationnel / far-IR
```

Les coefficients de bandes sont effectifs. Ils gardent un noyau CO2 + H2O
simple et lisible ; ils ne remplacent pas HITRAN, RADIS ou une methode
correlated-k. Le script `calibrer_coefficients_co2.py` sert precisement a
deriver des `a_CO2_bande` plus tracables depuis des transmissions HITRAN/RADIS,
puis a recaler leur facteur global sur le forcage `280 -> 560 ppm`.

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

Les nuages ne sont pas un terme radiatif explicite dans le modele 3. Leur effet
moyen sur le court-onde de surface est inclus dans la transmissivite ERA5
mensuelle :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
```

Dans le long-onde, il n'y a pas de `tau_nuage` :

```text
tau_total = tau_CO2 + tau_H2O
```

Les anciens mecanismes qui associaient `low_cloud_cover`,
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

Ce qui est volontairement absent du modele 3 :

- pas d'evolution de `T_surface(t)` ;
- pas de transport horizontal ;
- pas de lecture directe ERA5 pendant `calculer_colonne_radiative` ;
- pas de fallback analytique dans le calcul normal ;
- pas d'emissivite variable de surface ;
- pas d'albedo nuageux court-onde explicite ;
- pas d'opacite nuageuse long-onde explicite ;
- pas d'ozone, aerosols, CH4, N2O ou microphysique nuageuse.

Ces absences sont des choix du modele 3. Elles ne doivent pas etre corrigees
en reutilisant les anciennes formules dans la theorie.

## Validation

Le paquet conserve des flux ERA5 mensuels pour comparer les ordres de grandeur.
Le court-onde est volontairement cale sur une transmissivite mensuelle moyenne.
Le long-onde reste un modele effectif CO2 + H2O. La validation du modele 3
s'appuie donc sur des comparaisons grille par grille avec ERA5, sans cas local
nomme dans la theorie.
