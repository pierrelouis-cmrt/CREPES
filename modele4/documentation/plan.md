# Plan modèle 4 - grille terrestre et température de surface

## Objectif

Le modèle 4 doit produire une température de surface `T_surface(t, lat, lon)`
sur une grille de la surface terrestre.

Il ne doit pas réimplémenter le transfert radiatif colonne par colonne. Il doit
appeler le modèle 3 comme module radiatif local.

Responsabilités :

- modèle 3 : calculer les flux radiatifs d'une colonne locale pour une
  température de surface donnée ;
- modèle 4 : construire la grille, appeler les colonnes et intégrer le bilan de
  surface dans le temps.

## Principe

Pour chaque cellule de grille :

```text
lat, lon -> colonne locale -> flux radiatifs -> bilan de surface -> T_surface
```

Équation cible :

```text
C_surface dT_surface/dt =
    SW_absorbe_surface_corrige
  + LW_down_absorbe_surface
  - LW_up_surface
  - termes de surface repris du modèle 0
```

Le modèle 4 reprend la logique du modèle 0 pour les bilans de surface, mais en
remplaçant le flux longwave atmosphérique constant par les flux calculés par le
modèle 3.

## Décision shortwave pour la première version

Le flux shortwave renvoyé par le modèle 3 reste un diagnostic pédagogique de
géométrie solaire simplifiée. Pour faire évoluer `T_surface` dans le modèle 4,
il ne doit pas être utilisé tel quel, car il surestime fortement le flux net de
surface.

Le mode par défaut du modèle 4 doit conserver le calcul solaire historique du
projet :

```text
SW_TOA_local(t) = S0 * max(cos(i(t)), 0)
```

avec `S0 = 1361 W/m2`, la constante solaire déjà utilisée dans le modèle 0 et
le modèle 3. La correction consiste à ajouter une transmissivité atmosphérique
mensuelle dérivée d'ERA5 :

```text
tau_SW_mensuel =
    era5_sw_down_surface_w_m2
  / moyenne_mensuelle(S0 * max(cos(i), 0))

SW_down_surface(t) =
    tau_SW_mensuel * S0 * max(cos(i(t)), 0)

SW_absorbe_surface_corrige =
    SW_down_surface(t) * (1 - albedo_surface)
```

Ce choix suit la forme standard du bilan shortwave net de surface
`R_ns = (1 - alpha) R_s`, conserve l'albédo explicite du projet, garde le
cycle jour/nuit et saisonnier calculé par le modèle, et utilise ERA5 seulement
pour représenter l'effet atmosphérique moyen que le projet ne modélise pas
encore.

Deux modes de comparaison peuvent être gardés :

```text
mode_entree_mensuelle = era5_sw_down_surface_w_m2 * (1 - albedo_surface)
mode_validation       = era5_sw_net_surface_w_m2
```

Les anciennes corrections nuageuses TOA de type `albedo_nuages_effectif`
restent des diagnostics historiques. Elles ne doivent pas être réintroduites
comme transmission solaire de surface dans le modèle 4.

La justification détaillée et les sources sont dans
`modele3/documentation/RECHERCHE_SHORTWAVE_ET_OPTIMISATION.md`.

## Entrées attendues

Pour chaque cellule :

- latitude et longitude ;
- température de surface initiale ;
- capacité thermique surfacique ;
- données de surface disponibles : albédo, terre/mer, neige/glace,
  émissivité ;
- données atmosphériques mensuelles utilisées par le modèle 3.

## Données versionnables pour Git

Les fichiers bruts de `ressources/` ne doivent pas être suivis par Git : les
NetCDF ERA5 et les éventuels fichiers MODIS préparés en amont sont trop
volumineux et ne conviennent pas à un dépôt GitHub classique. Pour le modèle 4,
la bonne stratégie est de suivre dans Git des fichiers déjà prétraités, compacts
et suffisants pour lancer une simulation globale basse résolution.

Choix retenu pour la première version :

```text
resolution_grille = 5 degres
latitudes = centres de cellules tous les 5 degres
longitudes = centres de cellules tous les 5 degres
nombre_cellules = 36 * 72 = 2592
```

Le modèle 4 ne doit pas versionner les 37 niveaux ERA5 bruts. Il doit
pré-calculer localement, à partir de `ressources/`, les grandeurs directement
utilisées par le modèle 3 :

- pressions bas/haut des couches du modèle 3 ;
- température moyenne de couche `T_moyen` ;
- humidité spécifique moyenne de couche `q_moyen` ;
- pression de surface mensuelle ;
- albédo de surface mensuel ;
- émissivité de surface constante `0.98`, donc pas de carte dédiée à stocker ;
- masque terre/mer et neige/glace si utile ;
- éventuellement flux ERA5 de validation mensuels.

Format recommandé :

- éviter le CSV mondial brut ;
- stocker les tableaux en `.npz` compressé ;
- utiliser des tableaux `int16` avec `scale_factor` et `offset` documentés ;
- garder la metadata dans le `.npz` pour décrire la grille, les unités, les
  facteurs d'échelle, l'année/source et les variables ;
- viser des fichiers suivis par Git nettement sous `30 Mo`.

Le générateur de référence doit être commun au modèle 3 et au modèle 4. Il
vit côté `modele3`, car il prépare d'abord les colonnes radiatives que le
modèle 3 sait consommer, puis le modèle 4 itère sur ces mêmes colonnes.

Exemple de dossier versionnable :

```text
modele3/ressources/donnees_precalculees/grille_5deg_2024/
  donnees_colonnes_5deg_2024.npz
  README.md
```

Les tableaux `int16` ne stockent pas directement les valeurs physiques. Chaque
variable doit être reconstruite par :

```text
valeur_physique = valeur_int16 * scale_factor + offset
```

Exemples de quantification acceptable :

```text
T_moyen_K              -> pas 0.01 K
q_moyen_kgkg           -> pas 1e-7 kg/kg
pression_hPa           -> pas 0.1 hPa
albedo/fraction         -> pas 1e-4
flux_W_m2              -> pas 0.1 W/m2
```

Le workflow attendu est donc :

1. un membre du groupe qui possède les gros fichiers lance le prétraitement ;
2. le script lit `ressources/`, interpole sur la grille 5°, construit les
   couches du modèle 3 et écrit le `.npz` compact ;
3. ces fichiers pré-calculés sont suivis par Git ;
4. les autres membres clonent le dépôt et peuvent lancer le modèle 4 sans
   télécharger les gros fichiers d'origine.

Commande cible à implémenter :

```bash
./.venv/bin/python -m modele3.codes_python.generer_donnees \
  --resolution 5 \
  --annee 2024 \
  --ressources-dir ressources \
  --albedo-dir modele0_maintenance/ressources/albedo \
  --output modele3/ressources/donnees_precalculees/grille_5deg_2024
```

Le modèle 4 devra ensuite charger prioritairement ces données pré-calculées
produites par `modele3.codes_python.generer_donnees`. Si elles sont absentes
mais que `ressources/` existe, il pourra proposer la commande de génération.
S'il n'y a ni données pré-calculées ni `ressources/`, il devra échouer avec un
message clair plutôt que lancer une simulation globale fausse.

## Boucle temporelle

Le modèle 4 peut reprendre un pas de temps proche du modèle 0 :

```text
dt = 1800 s
```

À chaque pas :

1. déterminer le jour, le mois et l'heure solaire locale ;
2. lire/interpoler les données mensuelles de la cellule ;
3. calculer `SW_TOA_local(t) = S0 * max(cos(i(t)), 0)` ;
4. appliquer la transmissivité mensuelle `tau_SW_mensuel` issue d'ERA5 ;
5. appeler le modèle 3 avec la température de surface courante ;
6. récupérer les flux longwave du modèle 3 ;
7. utiliser `SW_absorbe_surface_corrige` dans le bilan de surface ;
8. calculer les autres termes de surface repris du modèle 0 ;
9. mettre à jour `T_surface`.

## Grille

Première version :

- grille globale `5 degres` ;
- `36` latitudes et `72` longitudes, donc `2592` cellules ;
- cellules indépendantes ;
- aucun échange horizontal.

La résolution pourra rester configurable dans le code, mais la version
référence suivie par Git doit être la grille `5 degres`. Cela fixe une cible de
taille raisonnable pour les fichiers pré-calculés et évite que chaque membre du
groupe produise une grille différente.

## Sorties attendues

Pour une simulation :

```text
T_surface_K[temps, latitude, longitude]
```

Diagnostics utiles :

- flux radiatifs moyens ;
- température minimale/maximale par cellule ;
- cartes mensuelles ou journalières ;
- série temporelle pour une cellule de référence.

## Validation

Comparer progressivement :

- cartes de température simulée contre `skin temperature` ou `2m temperature` ;
- flux shortwave corrigé contre `era5_sw_net_surface_w_m2` ;
- flux longwave du modèle 3 contre ERA5 ;
- comportement saisonnier entre hémisphères ;
- différence terre/océan ;
- différence altitude basse / montagne.

## Hors périmètre initial

Ne pas ajouter tout de suite :

- échanges horizontaux ;
- circulation atmosphérique ;
- océan dynamique ;
- rétroaction de la surface sur les profils atmosphériques ;
- calibration spectroscopique avancée.

Le premier modèle 4 doit surtout prouver que le modèle 3 peut être appelé de
manière stable sur une grille et que le bilan de surface produit une évolution
cohérente de `T_surface`.
