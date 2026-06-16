# Plan modèle 4 - grille terrestre et température de surface

## Objectif

Le modèle 4 doit produire une température de surface `T_surface(t, lat, lon)`
sur une grille de la surface terrestre.

Il ne doit pas réimplémenter le transfert radiatif colonne par colonne. Il doit
appeler le modèle 3.1 comme module radiatif local.

Responsabilités :

- modèle 3.1 : calculer les flux radiatifs d'une colonne locale pour une
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
modèle 3.1.

## Entrées attendues

Pour chaque cellule :

- latitude et longitude ;
- température de surface initiale ;
- capacité thermique surfacique ;
- données de surface disponibles : albédo, terre/mer, neige/glace,
  émissivité ;
- données atmosphériques mensuelles utilisées par le modèle 3.1.

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
utilisées par le modèle 3.1 :

- pressions bas/haut des couches du modèle 3.1 ;
- température moyenne de couche `T_moyen` ;
- humidité spécifique moyenne de couche `q_moyen` ;
- fraction nuageuse de couche ;
- pression de surface mensuelle ;
- albédo de surface mensuel ;
- émissivité de surface constante `0.98`, donc pas de carte dédiée à stocker ;
- masque terre/mer et neige/glace si utile ;
- éventuellement flux ERA5 de validation mensuels.

Format recommandé :

- éviter le CSV mondial brut ;
- stocker les tableaux en `.npz` compressé ;
- utiliser des tableaux `int16` avec `scale_factor` et `offset` documentés ;
- garder un petit fichier `metadata.json` décrivant la grille, les unités, les
  facteurs d'échelle, l'année/source et les variables ;
- viser des fichiers suivis par Git nettement sous `30 Mo`.

Le générateur de référence doit être commun au modèle 3.1 et au modèle 4. Il
vit côté `modele3_1`, car il prépare d'abord les colonnes radiatives que le
modèle 3.1 sait consommer, puis le modèle 4 itère sur ces mêmes colonnes.

Exemple de dossier versionnable :

```text
modele3_1/donnees_precalculees/grille_5deg_2024/
  metadata.json
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
albedo/cloud/fraction   -> pas 1e-4
flux_W_m2              -> pas 0.1 W/m2
```

Le workflow attendu est donc :

1. un membre du groupe qui possède les gros fichiers lance le prétraitement ;
2. le script lit `ressources/`, interpole sur la grille 5°, construit les
   couches du modèle 3.1 et écrit les `.npz`/`metadata.json` compacts ;
3. ces fichiers pré-calculés sont suivis par Git ;
4. les autres membres clonent le dépôt et peuvent lancer le modèle 4 sans
   télécharger les gros fichiers d'origine.

Commande cible à implémenter :

```bash
./.venv/bin/python -m modele3_1.generer_donnees \
  --resolution 5 \
  --annee 2024 \
  --ressources-dir ressources \
  --albedo-dir modele0_maintenance/ressources/albedo \
  --output modele3_1/donnees_precalculees/grille_5deg_2024
```

Le modèle 4 devra ensuite charger prioritairement ces données pré-calculées
produites par `modele3_1`. Si elles sont absentes mais que `ressources/`
existe, il pourra proposer la commande de génération. S'il n'y a ni données
pré-calculées ni `ressources/`, il devra échouer avec un message clair plutôt
que lancer une simulation globale fausse.

## Boucle temporelle

Le modèle 4 peut reprendre un pas de temps proche du modèle 0 :

```text
dt = 1800 s
```

À chaque pas :

1. déterminer le jour, le mois et l'heure solaire locale ;
2. lire/interpoler les données mensuelles de la cellule ;
3. appeler le modèle 3.1 avec la température de surface courante ;
4. récupérer les flux radiatifs ;
5. calculer les autres termes de surface repris du modèle 0 ;
6. mettre à jour `T_surface`.

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

Le premier modèle 4 doit surtout prouver que le modèle 3.1 peut être appelé de
manière stable sur une grille et que le bilan de surface produit une évolution
cohérente de `T_surface`.
