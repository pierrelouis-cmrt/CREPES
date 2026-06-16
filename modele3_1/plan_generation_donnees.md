# Plan generation donnees compactes modele 3.1 / modele 4

Objectif : creer un script dedie qui transforme les grosses donnees locales de
`ressources/` en un paquet `.npz` compact, versionnable et suffisant pour faire
tourner le modele 3.1 et le modele 4 sur une grille globale de `5 degres`.

Le modele ne doit plus dependre d'un extrait artificiel limite a Paris. Paris
peut rester un cas de test, mais il doit etre extrait du meme paquet global que
le reste de la Terre.

## Decision centrale

Le paquet de donnees pretraite devient le contrat commun :

```text
donnees brutes ignorees par Git -> script de generation -> paquet NPZ suivi par Git
                                                   |
                                                   +-> modele 3.1, une colonne
                                                   +-> modele 4, toutes les colonnes
                                                   +-> tests
```

Le modele 3.1 ne lit pas les NetCDF/HDF lourds pendant un calcul normal. Il lit
un paquet compact deja prepare. Le modele 4 lit exactement le meme paquet, mais
sur toute la grille.

## Emplacement propose

Script :

```text
modele3_1/generer_donnees.py
```

Sortie versionnable :

```text
modele3_1/donnees_precalculees/grille_5deg_2024/
  metadata.json
  donnees_colonnes_5deg_2024.npz
  README.md
```

Cette sortie doit etre petite et suivie par Git. Les gros fichiers source
restent dans `ressources/`, qui est ignore par Git.

## Entrees

Sources locales attendues pour l'implementation 3.1 :

```text
ressources/
  *.nc ERA5 mensuels surface/flux/profils
  MOD11C3.A2024/*.hdf eventuel, non obligatoire en 3.1
  albedo/
  albedo01.csv ... albedo12.csv
  CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc
```

Les CSV d'albedo et le fichier CERES existaient historiquement dans
`modele0_maintenance/ressources/albedo/`. Pour 3.1, les fichiers utiles ont ete
copies dans `ressources/albedo/`. C'est cette copie racine qui est la source
active du generateur. Le modele 3.1 ne lit pas et n'importe pas directement le
dossier `modele0_maintenance/`.

Le generateur utilise donc par defaut :

- `--ressources-dir ressources` pour les donnees lourdes ignorees ;
- `--albedo-dir ressources/albedo` pour les CSV d'albedo et CERES copies a la
  racine.

Le script ne doit pas telecharger de donnees. Il lit seulement ce qui est deja
present localement.

## Grille standard

Grille de reference :

```text
resolution = 5 degres
latitudes = -87.5, -82.5, ..., 82.5, 87.5
longitudes = -177.5, -172.5, ..., 172.5, 177.5
shape = 36 x 72
nombre_cellules = 2592
```

Ces points sont des centres de cellules. Pour les moyennes globales, le paquet
doit aussi contenir un poids de surface :

```text
poids_latitude = cos(latitude)
```

normalise ou documente dans `metadata.json`.

## Variables a garder

Garder uniquement ce que le modele 3.1, le modele 4 et les tests utilisent.

Coordonnees :

```text
lat_deg[lat]
lon_deg[lon]
poids_surface[lat, lon]
mois[12]
pression_bords_reference_hpa[layer_edge]
```

Surface mensuelle :

```text
pression_surface_hpa[mois, lat, lon]
albedo_surface[mois, lat, lon]
albedo_nuages_effectif[mois, lat, lon]
land_fraction[mois, lat, lon]
snow_ice_fraction[mois, lat, lon]
temperature_2m_k[mois, lat, lon]
skin_temperature_k[mois, lat, lon]
```

Couches mensuelles deja pretraitees :

```text
temperature_couche_k[mois, couche, lat, lon]
humidite_specifique_couche_kgkg[mois, couche, lat, lon]
fraction_nuageuse_couche[mois, couche, lat, lon]
masse_air_couche_kg_m2[mois, couche, lat, lon]
masse_h2o_couche_kg_m2[mois, couche, lat, lon]
```

Validation minimale :

```text
era5_lw_down_surface_w_m2[mois, lat, lon]
era5_sw_net_surface_w_m2[mois, lat, lon]
era5_olr_w_m2[mois, lat, lon]
era5_sw_down_surface_w_m2[mois, lat, lon]
```

Ne pas stocker :

- les 37 niveaux ERA5 bruts ;
- les variables non utilisees par le calcul ;
- les diagnostics couche-par-bande, qui dependent du CO2 et du calcul ;
- les fichiers MODIS bruts ;
- les cartes haute resolution.

## Quantification compacte

Le `.npz` doit etre ecrit avec `numpy.savez_compressed`, mais la vraie economie
vient surtout de la quantification.

Types recommandes :

```text
temperature_couche_k              int16, scale 0.01, offset 250.00
temperature_2m_k / skin_temp_k    int16, scale 0.01, offset 250.00
pression_surface_hpa              uint16, scale 0.1,  offset 0.0
masse_air_couche_kg_m2            uint16, scale 0.1,  offset 0.0
humidite_specifique_kgkg          uint16, scale 5e-7, offset 0.0
masse_h2o_couche_kg_m2            uint16, scale 0.001, offset 0.0
albedo / fraction / land / snow   uint16, scale 1e-4, offset 0.0
flux_w_m2                         int16, scale 0.1,  offset 0.0
lat/lon                           float32
```

Chaque variable quantifiee doit avoir dans `metadata.json` :

```text
dtype_stocke
unite_physique
scale_factor
offset
valeur_manquante
source
```

Reconstruction :

```text
valeur = valeur_stockee * scale_factor + offset
```

Les valeurs manquantes doivent etre marquees par une sentinelle unique par type
et gerees avant reconstruction.

## Traitement physique pendant la generation

Le generateur fait les operations couteuses une seule fois :

1. lire les NetCDF ERA5 disponibles ;
2. selectionner/interpoler les champs sur la grille 5 degres ;
3. construire les couches de pression du modele 3.1 ;
4. moyenner `T(p)`, `q(p)` et `cc(p)` par couche ;
5. calculer `masse_air` et `masse_h2o` par couche ;
6. charger l'albedo de surface mensuel ;
7. charger l'albedo nuageux effectif CERES du modele 0 ;
8. borner les fractions dans leurs domaines physiques ;
9. quantifier les tableaux ;
10. ecrire `.npz`, `metadata.json` et un petit `README.md`.

Le calcul radiatif lui-meme reste dans le modele 3.1. Le generateur ne calcule
pas les bandes CO2/H2O, l'OLR modele, ni le flux net de surface modele.

## Albedo surface

Ordre de priorite :

1. CSV mensuels du modele 0 pour la grille globale standard ;
2. `fal` ERA5 si on genere un paquet local/experimental sans CSV ;
3. secours `0.30`, uniquement si on a explicitement autorise
   `--allow-fallbacks`.

Pour la version de reference Git, il faut eviter les secours silencieux. Si les
CSV manquent, le script doit echouer avec un message clair.

## Albedo nuages

Utiliser la methode du modele 0 :

```text
albedo_nuages_effectif = (toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon
```

depuis CERES EBAF-TOA, interpolee ou selectionnee au plus proche sur la grille
5 degres.

Ce champ est stocke directement dans le paquet. Le modele 3.1 ne recalcule pas
un albedo nuageux depuis `cloud_total`.

## Emissivite

Ne pas stocker une carte d'emissivite dans le paquet 3.1 de reference.

Le modele 3.1 utilise :

```text
emissivite_surface = 0.98
```

partout. Cela evite de transporter une variable inutile et rend le contrat plus
simple.

## Commande cible

Commande normale :

```bash
./.venv/bin/python -m modele3_1.generer_donnees \
  --resolution 5 \
  --annee 2024 \
  --ressources-dir ressources \
  --albedo-dir ressources/albedo \
  --output modele3_1/donnees_precalculees/grille_5deg_2024
```

Options utiles :

```text
--overwrite
--dry-run
--allow-fallbacks
--only-metadata
--compression-level
```

`--dry-run` doit afficher les fichiers trouves, les variables detectees, la
taille estimee du paquet et les variables qui manquent.

## Chargement cote modele

Ajouter ensuite un chargeur simple :

```text
charger_paquet_grille(chemin)
extraire_colonne(paquet, lat, lon, mois=None, jour_annee=None)
iterer_colonnes(paquet)
```

Modele 3.1 :

- utilise `extraire_colonne` pour calculer une seule colonne ;
- peut prendre Paris comme n'importe quelle autre position ;
- ne garde plus un JSON Paris comme source principale.

Modele 4 :

- charge le paquet une seule fois ;
- boucle sur `lat, lon` ;
- passe les colonnes au modele 3.1 ;
- reutilise les memes poids de surface pour les moyennes globales.

Tests :

- testent le chargeur sur le paquet Git ;
- testent Paris en l'extrayant du paquet ;
- verifient que les tableaux ont les shapes attendues ;
- verifient que les valeurs reconstruites restent dans les bornes physiques.

## Taille cible

La cible raisonnable est :

```text
paquet complet 5 degres < 20 Mo
objectif confortable   < 10 Mo
```

Avec `12 mois x 10 couches x 36 x 72` et des tableaux `int16/uint16`, les
variables principales doivent tenir dans quelques Mo avant compression zip. Si
la sortie depasse largement cette taille, c'est probablement qu'on stocke trop
de variables ou des `float64`.

## Echecs acceptables

Le script doit echouer clairement si :

- les NetCDF ERA5 necessaires manquent ;
- les CSV d'albedo surface manquent pour la generation de reference ;
- le fichier CERES manque pour l'albedo nuageux ;
- une variable attendue n'est pas trouvee ;
- la sortie existe deja et `--overwrite` n'est pas donne.

Il ne doit pas fabriquer une grille globale avec des donnees synthetiques sans
que l'utilisateur l'ait explicitement demande.

## Critere d'acceptation

Le generateur est pret quand :

- il produit un dossier `grille_5deg_2024` suivi par Git ;
- le paquet peut etre charge sans les donnees brutes de `ressources/` ;
- le modele 3.1 peut calculer Paris depuis ce paquet ;
- le modele 4 peut iterer sur les `2592` colonnes ;
- les tests n'ont plus besoin d'un JSON Paris isole ;
- le `metadata.json` suffit a comprendre les sources, unites, echelles et
  shapes sans ouvrir le code.
