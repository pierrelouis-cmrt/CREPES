# Plan modele 3.1 - colonne finale pour le modele 4

Ce document est un plan de polish, pas une implementation. L'objectif est de
partir tres largement du modele 3 actuel, de retirer les approximations trop
opaques, et d'obtenir une colonne radiative assez propre pour etre reutilisee
dans le modele 4 comme brique independante de chaque cellule de grille.

Le but n'est pas de construire un modele climatique professionnel. Le coeur du
modele reste le transfert radiatif infrarouge simplifie autour du CO2, avec la
vapeur d'eau car son effet est dominant. Le reste doit etre minimal, explicite
et source.

## Decisions figees pour 3.1

- Le modele 3.1 est la brique colonne appelee par le modele 4.
- Le modele 3.1 ne lit pas les gros fichiers bruts pendant un calcul normal.
- Un paquet `.npz` compact, genere a partir de `ressources/`, devient la source
  commune pour le modele 3.1, le modele 4 et les tests.
- La grille de reference versionnable est une grille globale de `5 degres`
  (`36 x 72 = 2592` cellules).
- L'emissivite est constante : `emissivite_surface = 0.98`.
- L'albedo de surface vient des donnees, pas d'une constante cachee.
- L'albedo nuageux vient de CERES/modele 0, pas de `0.50 * cloud_total`.
- Les nuages long-onde ne sont pas modelises par defaut avec un coefficient
  arbitraire.
- Le court-onde reste simple et explicitement limite.
- Le JSON Paris peut rester un outil legacy temporaire, mais il ne doit plus
  etre la source principale des tests.

## Reponses courtes a l'audit actuel

### Emissivite

Non, il n'y a pas besoin de distinguer les emissivites par type de surface dans
la version 3.1.

Dans le modele 3 actuel, la logique distingue surtout :

- surface par defaut : `0.98` ;
- neige/glace : `0.98` ;
- ocean : `0.985`.

La difference `0.98 -> 0.985` est petite devant les incertitudes du modele. Sur
le cas Paris de reference, elle change le flux net de surface de moins de
`1 W/m2`. Garder une logique terre/ocean/neige complique donc le code sans
apporter un gain physique utile au niveau actuel.

Decision 3.1 :

```text
emissivite_surface = 0.98
```

partout, sauf si une experience future fournit explicitement une emissivite de
surface deja pretraitee et documentee. La lecture directe MODIS reste hors
perimetre 3.1.

### Albedo du sol

Le modele 3 actuel ne prend pas toujours un albedo constant.

Quand les donnees ERA5 locales sont disponibles, `donnees.py` lit `fal`
(`forecast albedo`) et l'utilise comme `albedo_surface`. Pour l'extrait Paris
juillet versionne, on a par exemple :

```text
albedo_surface = 0.16380389034748077
```

Donc ce n'est pas le `0.30` constant.

En revanche, si aucune donnee n'est disponible, le secours actuel est bien :

```text
ALBEDO_SURFACE_DEFAUT = 0.30
```

Le modele 0 ne prend pas non plus un albedo constant : il charge des CSV
mensuels `albedo01.csv` a `albedo12.csv`, regenerables depuis NASA POWER avec
le rapport :

```text
ALLSKY_SFC_SW_UP / ALLSKY_SFC_SW_DWN
```

Decision 3.1 :

1. L'API physique ne doit jamais inventer l'albedo : elle recoit
   `albedo_surface`.
2. Pour une colonne locale construite depuis ERA5, garder `fal` si present,
   car il est coherent avec les autres champs ERA5 de la colonne.
3. Pour le modele 4 global et versionnable, utiliser par defaut les CSV
   mensuels du modele 0, car ils couvrent deja la grille et sont presents dans
   le depot.
4. Garder `0.30` seulement comme secours explicite, avec un diagnostic
   `source_albedo_surface = "secours_0.30"`.

### Albedo des nuages

Le modele 3 actuel contient une approximation trop opaque :

```text
albedo_nuage = 0.50 * cloud_total
```

Ce coefficient `0.50` ne doit pas survivre dans le modele 3.1. Il peut donner
un ordre de grandeur, mais il n'est pas assez justifie pour une version finale.

Le modele 0 fait mieux : il calcule un albedo nuageux effectif depuis CERES :

```text
albedo_nuages = (toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon
```

Ce n'est pas un albedo microphysique pur du nuage. C'est un facteur effectif de
reflexion supplementaire des nuages vu depuis le sommet de l'atmosphere,
normalise par le solaire entrant. Mais c'est une quantite observationnelle,
issue d'un produit radiatif CERES, et elle correspond bien a l'usage simple du
modele 0 :

```text
SW_absorbe_surface = SW_incident * (1 - albedo_nuages) * (1 - albedo_surface)
```

Decision 3.1 :

- supprimer `COEFFICIENT_NUAGE_SW` ;
- ajouter une entree `albedo_nuages_effectif` ;
- alimenter cette entree avec la methode CERES du modele 0 ;
- documenter clairement que c'est une correction radiative effective, pas une
  propriete physique locale du nuage ;
- si la donnee manque, prendre `0.0` avec diagnostic, pas un coefficient cache.

## Priorites physiques pour la version 3.1

### Priorite 1 - Nettoyer les constantes arbitraires

A faire :

- remplacer l'emissivite type-de-surface par `0.98` constant ;
- supprimer `COEFFICIENT_NUAGE_SW = 0.50` ;
- ne plus calculer `albedo_nuage` depuis `cloud_total` ;
- traiter `cloud_total`, `low_cloud`, `medium_cloud`, `high_cloud` comme des
  diagnostics ou entrees futures, pas comme une formule d'albedo cachee.

Pourquoi :

- l'emissivite detaillee a un impact faible ici ;
- l'albedo de surface et l'albedo nuageux ont un impact tres fort sur le bilan
  court-onde ;
- les coefficients caches rendent le modele difficile a defendre.

### Priorite 2 - Stabiliser le contrat de donnees pour le modele 4

Chaque colonne du modele 4 doit pouvoir appeler exactement la meme fonction du
modele 3.1.

Contrat minimal d'une colonne :

```text
surface:
  latitude_deg
  longitude_deg
  mois ou jour_annee
  pression_surface_pa
  albedo_surface
  albedo_nuages_effectif
  emissivite_surface = 0.98

profil:
  pressions_hpa
  temperatures_k
  humidites_specifiques_kgkg

parametres:
  temperature_surface_k
  co2_ppm
```

Champs optionnels :

```text
surface:
  cloud_total
  low_cloud
  medium_cloud
  high_cloud
  land_fraction
  snow_ice_fraction
  skin_temperature_k
  temperature_2m_k

validation_flux:
  flux ERA5 ou CERES disponibles pour comparaison
```

Regle importante : la fonction principale doit etre simple a appeler depuis une
boucle de grille :

```text
resultat = calculer_colonne_radiative(donnees_colonne, temperature_surface_k, co2_ppm)
```

Elle ne doit pas lire des fichiers elle-meme.

Traitement temporel :

- si l'appel donne seulement un `mois`, utiliser directement les cartes du
  mois ;
- si l'appel donne un `jour_annee`, convertir vers une valeur mensuelle
  interpolee cycliquement entre les mois voisins ;
- ne pas ajouter de meteo journaliere dans 3.1 ;
- eviter les ruptures artificielles au passage d'un mois a l'autre si le
  modele 4 integre `T_surface(t)` au pas de temps journalier.

Bornage :

- borner `albedo_surface` dans `[0, 1]` ;
- borner `albedo_nuages_effectif` dans `[0, 0.95]` ;
- si un bornage corrige une valeur externe, l'indiquer dans les diagnostics.

### Priorite 3 - Garder le court-onde simple

Le court-onde n'est pas le coeur scientifique du projet. Ne pas ajouter de
transmission atmospherique solaire generique, ozone, aerosols ou absorption
spectrale shortwave dans la version 3.1, sauf si une source tres solide donne
un coefficient directement applicable aux hypotheses du modele.

Formule cible 3.1 :

```text
SW_incident_TOA_local = geometrie_solaire(latitude, jour, heure ou moyenne_journaliere)
SW_absorbe_surface = SW_incident_TOA_local
                    * (1 - albedo_nuages_effectif)
                    * (1 - albedo_surface)
```

Cette formule reste grossiere. Elle doit etre presentee comme telle dans le
README. Le gain principal de la 3.1 est d'enlever le coefficient nuageux
mysterieux, pas de resoudre le transfert solaire complet.

### Priorite 4 - Clarifier les nuages long-onde

Le modele 3 actuel ajoute aussi :

```text
tau_nuage = 0.10 * fraction_nuageuse
```

dans l'infrarouge. Ce coefficient est lui aussi effectif et peu justifie.

Constat sur Paris juillet :

- le retirer change peu le flux net de surface, environ quelques `W/m2` ;
- il change beaucoup l'OLR, donc il peut donner une bonne validation TOA pour
  une mauvaise raison.

Decision 3.1 :

- ne pas garder `COEFFICIENT_NUAGE_LW` dans le chemin physique par defaut ;
- garder les fractions nuageuses dans les diagnostics ;
- si on veut un mode "nuages LW effectifs" plus tard, il doit etre optionnel,
  nomme explicitement et source par une grandeur radiative observee, par
  exemple un effet radiatif nuageux CERES, pas par une constante cachee.

### Priorite 5 - Garder CO2 + H2O comme coeur du modele

A conserver depuis le modele 3 :

- bandes infrarouges CO2 simplifiees ;
- bandes H2O effectives ;
- addition des opacites avant transmission :

```text
tau_total = tau_CO2 + tau_H2O
transmission = exp(-D * tau_total)
emissivite_couche = 1 - transmission
```

A ameliorer dans 3.1 :

- documenter clairement que les coefficients de bandes sont effectifs ;
- renommer les variables pour separer :
  - opacite CO2 ;
  - opacite H2O ;
  - eventuel effet nuageux, qui ne doit plus etre implicite ;
- verifier que `MASSE_H2O_REFERENCE_KG_M2 = 10.0` est visible dans la theorie
  et dans les diagnostics.

Ne pas ajouter par defaut :

- absorption H2O shortwave ;
- ozone ;
- CH4, N2O ;
- aerosols ;
- correlated-k ;
- RADIS/HITRAN.

## Architecture de code cible

La version 3.1 doit rester lisible par quelqu'un qui suit la physique, pas par
quelqu'un qui veut surtout admirer une architecture Python.

Structure cible simple :

```text
modele3_1/
  README.md
  THEORIE.md
  plan.md
  plan_generation_donnees.md
  modele3_1.py
  physique.py
  donnees.py
  generer_donnees.py
  donnees_precalculees/
    grille_5deg_2024/
      metadata.json
      donnees_colonnes_5deg_2024.npz
      README.md
  donnees_exemple/
    paris_2024_m07.json
  tests/
    tester_modele3_1.py
```

Regles de style :

- pas de classes pour representer les couches ;
- pas de `dataclass` obligatoire ;
- pas de sous-packages multiples ;
- dictionnaires simples pour `surface`, `profil`, `couches`, `resultat` ;
- fonctions courtes, nommees d'apres la grandeur physique calculee ;
- exceptions simples seulement quand l'entree rend le calcul impossible ;
- diagnostics explicites plutot que gestion d'erreur sophistiquee.

Repartition :

- `modele3_1.py` : construction des couches, propagation des flux, fonction
  publique, CLI courte ;
- `physique.py` : constantes, solaire, Planck, masses colonne, opacites ;
- `donnees.py` : lecture JSON, extraction ERA5 optionnelle, lecture/reprise des
  donnees pretraitees et extraction d'une colonne depuis le paquet `.npz` ;
- `generer_donnees.py` : generation du paquet compact depuis les grosses
  donnees locales de `ressources/` et les donnees d'albedo du modele 0 ;
- `tests/` : quelques tests numeriques directs.

## Plan de codage detaille

### Etape 0 - Produire le paquet de donnees compact

Avant de finaliser le calcul global, implementer le plan separe :

```text
modele3_1/plan_generation_donnees.md
```

Le script cible est :

```text
modele3_1/generer_donnees.py
```

Il doit lire les donnees lourdes locales ignorees par Git, construire la grille
`5 degres`, pretraiter les couches utiles au modele 3.1, puis ecrire un paquet
`.npz` compact suivi par Git.

Ce paquet remplace le JSON Paris comme source principale. Le modele 3.1 extrait
Paris depuis le paquet quand on veut tester une colonne precise.

### Etape 1 - Copier la base utile du modele 3

Partir de :

- `modele3/modele3.py` ;
- `modele3/physique/calculs.py` ;
- `modele3/donnees.py` ;
- `modele3/donnees_exemple/paris_2024_m07.json` ;
- `modele3/tests/tester_modele3.py`.

Puis simplifier la structure vers les trois fichiers actifs :

```text
modele3_1.py
physique.py
donnees.py
```

### Etape 2 - Fixer l'emissivite

Dans `donnees.py` :

- supprimer `EMISSIVITE_OCEAN` ;
- supprimer `EMISSIVITE_NEIGE_GLACE` ;
- supprimer `_emissivite_simple` ;
- mettre `emissivite_surface = 0.98` pour toutes les colonnes.

Dans les sorties :

- garder `emissivite_surface` visible ;
- ajouter `source_emissivite_surface = "constante_0.98"`.

### Etape 3 - Remplacer l'albedo nuageux

Dans `physique.py` :

- supprimer `COEFFICIENT_NUAGE_SW` ;
- supprimer `albedo_nuage_effectif(cloud_total)`.

Dans `donnees.py` :

- reprendre la methode du modele 0 :

```text
load_monthly_cloud_albedo_from_ceres(lat, lon)
```

- charger la valeur du mois ;
- ecrire dans `surface["albedo_nuages_effectif"]`.

Dans `modele3_1.py` :

- lire directement `surface["albedo_nuages_effectif"]` ;
- ne jamais recalculer l'albedo nuageux depuis `cloud_total`.

### Etape 4 - Choisir proprement l'albedo de surface

Ordre de priorite :

1. `albedo_surface` deja fourni dans un JSON ou par le modele 4 ;
2. `fal` ERA5 si on construit une colonne locale depuis ERA5 ;
3. CSV mensuel du modele 0 pour une grille globale ;
4. secours `0.30`, avec diagnostic visible.

Important pour le modele 4 :

- precharger les cartes mensuelles d'albedo en dehors de la boucle si possible ;
- passer la valeur scalaire a chaque colonne ;
- ne pas faire ouvrir 12 CSV par chaque colonne.

### Etape 5 - Retirer le nuage long-onde par defaut

Dans `physique.py` :

- supprimer `COEFFICIENT_NUAGE_LW` du calcul par defaut ;
- mettre :

```text
tau_total = tau_CO2 + tau_H2O
```

Dans les diagnostics :

- conserver `fraction_nuageuse` si disponible ;
- ne pas afficher `tau_nuage` sauf dans un futur mode optionnel.

### Etape 6 - Nettoyer les diagnostics

Sorties minimales :

```text
SW_incident_surface
SW_absorbe_surface
LW_up_surface
LW_down_surface
LW_down_absorbe_surface
OLR
flux_net_radiatif_surface
```

Diagnostics utiles :

```text
sources:
  albedo_surface
  albedo_nuages_effectif
  emissivite_surface

diagnostics_bandes:
  bande
  famille
  lambda_min_um
  lambda_max_um
  tau_CO2_total
  tau_H2O_total
  flux_surface
  flux_sommet
  flux_descendant_surface
```

Eviter les diagnostics trop volumineux par defaut. Pour le modele 4, il faudra
pouvoir desactiver les diagnostics couche-par-bande pour ne pas exploser la
memoire.

### Etape 7 - Mettre a jour README et theorie

Le `README.md` doit dire clairement :

- le modele 3.1 est une colonne locale ;
- il sera appele par le modele 4 pour chaque cellule ;
- l'emissivite est constante `0.98` ;
- l'albedo de surface vient soit d'ERA5 `fal`, soit des CSV mensuels du modele
  0, soit d'un JSON ;
- l'albedo des nuages reprend la methode CERES du modele 0 ;
- le court-onde reste simplifie et ne represente pas un transfert atmospherique
  complet ;
- les nuages long-onde ne sont pas modelises par defaut ;
- le coeur du modele est CO2 + H2O en infrarouge.

Le `THEORIE.md` doit separer :

- ce qui est une formule physique standard ;
- ce qui est un coefficient effectif du projet ;
- ce qui vient d'une source externe ;
- ce qui est volontairement ignore.

### Etape 8 - Tests a ajouter

Tests minimaux :

1. L'emissivite vaut toujours `0.98` pour terre, ocean, neige/glace.
2. L'albedo de surface Paris reste celui du JSON si le JSON le fournit.
3. Le secours `0.30` n'est utilise que si aucune autre source n'existe.
4. L'albedo nuageux n'est jamais calcule par `0.50 * cloud_total`.
5. Un `albedo_nuages_effectif` fourni en entree est utilise tel quel.
6. Le calcul de colonne fonctionne sans fichiers lourds, avec le paquet `.npz`
   versionne.
7. La fonction publique peut etre appelee en boucle sur plusieurs colonnes
   independantes.
8. Le mode sans diagnostics lourds renvoie seulement les flux principaux.

Validation a regarder sans sur-calibrer :

- Paris juillet doit continuer a tourner sans fichiers lourds ;
- Paris doit etre extrait du paquet global, pas fabrique comme donnee
  synthetique isolee ;
- les flux long-onde doivent rester dans le bon ordre de grandeur ;
- le court-onde peut rester imparfait, mais l'origine de l'ecart doit etre
  lisible ;
- si l'OLR s'eloigne apres retrait du nuage long-onde arbitraire, ne pas
  reintroduire un coefficient cache pour le recoller artificiellement.

## Ce qu'on ne fait pas dans 3.1

Ne pas ajouter :

- modele climatique complet ;
- dynamique atmospherique ;
- echanges horizontaux entre colonnes ;
- evolution de `T_surface(t)` ;
- ozone ;
- aerosols ;
- chimie atmospherique ;
- microphysique nuageuse ;
- lecture directe HDF MODIS ;
- transmission shortwave generique non sourcee ;
- calibration cachee pour forcer le modele a coller a ERA5.
- donnees synthetiques globales cachees dans les tests.

Ces points peuvent etre notes comme limites, mais pas transformes en code 3.1.

## Critere d'acceptation

Le modele 3.1 est pret pour le modele 4 quand :

- une colonne se calcule avec une fonction unique et stable ;
- le paquet `.npz` compact existe et suffit pour lancer les tests sans
  `ressources/` ;
- toutes les entrees physiques importantes sont explicites dans `surface`,
  `profil` ou les parametres ;
- l'emissivite ne depend plus de branches inutiles ;
- l'albedo de surface n'est pas constant sauf secours documente ;
- l'albedo nuageux vient de CERES/modele 0 ou d'une entree fournie ;
- aucun coefficient nuageux cache ne reste dans le calcul par defaut ;
- les sorties principales sont stables et testees ;
- la documentation dit clairement ce que le modele sait faire et ce qu'il ne
  pretend pas faire.

## Sources et fichiers a utiliser

Sources locales :

- `modele3/donnees.py` : lecture ERA5 `fal`, secours `0.30`, emissivites
  actuelles ;
- `modele3/physique/calculs.py` : coefficients nuageux actuels a supprimer ;
- `modele3/README.md` : validation Paris et limites actuelles ;
- `ressources/albedo/albedo01.csv` a `albedo12.csv` : albedo surface mensuel,
  copie racine des CSV historiques du modele 0 ;
- `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` :
  albedo nuageux effectif CERES, copie racine du fichier historique du modele 0.

Note d'implementation : le modele 0 reste intact et sert seulement a tracer la
provenance historique de ces fichiers. Le code 3.1 ne lit pas directement
`modele0_maintenance/` et n'importe aucun module depuis ce dossier.

Sources externes :

- Copernicus ERA5 monthly averaged single levels :
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means
- NASA CERES, description generale du programme :
  https://ceres.larc.nasa.gov/
- NASA ASDC CERES EBAF-TOA Edition 4.2.1 :
  https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1
- NASA POWER Parameter Dictionary :
  https://power.larc.nasa.gov/docs/tutorials/parameters/
- NASA POWER Methodology :
  https://power.larc.nasa.gov/docs/methodology/
- MODIS MOD11C3 emissivite mensuelle, garde comme piste future seulement :
  https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod11c3-061
