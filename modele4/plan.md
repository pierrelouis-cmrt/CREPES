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
    SW_absorbe_surface
  + LW_down_absorbe_surface
  - LW_up_surface
  - termes de surface repris du modèle 0
```

Le modèle 4 reprend la logique du modèle 0 pour les bilans de surface, mais en
remplaçant le long-onde atmosphérique constant par les flux calculés par le
modèle 3.

## Entrées attendues

Pour chaque cellule :

- latitude et longitude ;
- température de surface initiale ;
- capacité thermique surfacique ;
- données de surface disponibles : albédo, terre/mer, neige/glace,
  émissivité ;
- données atmosphériques mensuelles utilisées par le modèle 3.

## Boucle temporelle

Le modèle 4 peut reprendre un pas de temps proche du modèle 0 :

```text
dt = 1800 s
```

À chaque pas :

1. déterminer le jour, le mois et l'heure solaire locale ;
2. lire/interpoler les données mensuelles de la cellule ;
3. appeler le modèle 3 avec la température de surface courante ;
4. récupérer les flux radiatifs ;
5. calculer les autres termes de surface repris du modèle 0 ;
6. mettre à jour `T_surface`.

## Grille

Première version :

- grille basse résolution ;
- cellules indépendantes ;
- aucun échange horizontal.

La résolution doit rester configurable pour permettre un test rapide avant une
grille plus fine.

## Sorties attendues

Pour une simulation :

```text
T_surface_K[temps, latitude, longitude]
```

Diagnostics utiles :

- flux radiatifs moyens ;
- température minimale/maximale par cellule ;
- cartes mensuelles ou journalières ;
- série temporelle pour une cellule de référence, par exemple Paris.

## Validation

Comparer progressivement :

- cartes de température simulée contre `skin temperature` ou `2m temperature` ;
- flux court-onde et long-onde contre ERA5 ;
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
