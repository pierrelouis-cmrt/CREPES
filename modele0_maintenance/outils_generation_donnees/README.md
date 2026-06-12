# Génération des données CREPES

À lancer depuis la racine `CREPES/` :

```bash
python3 outils_generation_donnees/generer_donnees.py
```

Ce dossier ne garde que les générateurs maintenus :

- `generer_donnees.py` : grilles de température du moteur actuel et conversion `12_mois`.
- `albedo/generer_albedo_surface.py` : CSV mensuels d'albédo via NASA POWER.

Les anciens scripts remplacés par le moteur actuel ont été retirés. Une donnée
active est soit générable par une commande ci-dessous, soit une source externe
locale explicitement identifiée.

## Statuts

| Statut | Sens |
| --- | --- |
| Générable proprement | Produit par `generer_donnees.py` avec le moteur CREPES actuel. |
| Générable par le moteur actuel | Dérivé d'une sortie du moteur actuel. |
| Générable par script actuel | Produit par un script maintenu de ce dossier. |
| Source externe locale | Fichier d'entrée à fournir/remplacer manuellement. |

Les grilles utilisent `codes_python/modele_courbe.py` : bilan Carcajous,
albédo sol, albédo nuages CERES, chaleur latente, capacité RZSM, convection
forcée et convection naturelle actives par défaut.

## Commandes

```bash
python3 outils_generation_donnees/generer_donnees.py --status
python3 outils_generation_donnees/generer_donnees.py --list
python3 outils_generation_donnees/generer_donnees.py --run tout-complet --dry-run --yes
python3 outils_generation_donnees/generer_donnees.py --run tout-rapide --force --yes
python3 outils_generation_donnees/generer_donnees.py --run grille-hires-rapide --force --yes
python3 outils_generation_donnees/generer_donnees.py --run temperatures-12mois --force --yes
python3 outils_generation_donnees/generer_donnees.py --run albedo-surface-nasa --dry-run --yes
```

`albedo-surface-nasa` appelle NASA POWER ; le lancer réellement peut être long
et nécessite le réseau.

## Cibles

| Cible | Sorties | Durée/source | Usage |
| --- | --- | --- | --- |
| `grille-lowres-rapide` | `ressources/grilles/grid_lowres_fast.npy` | 7 jours par défaut | Planisphère/sphère basse résolution rapide. |
| `grille-hires-rapide` | `ressources/grilles/grid_hires_fast.npy` | 7 jours par défaut | Planisphère/sphère haute résolution rapide. |
| `grille-lowres-1an` | `ressources/grilles/grid_lowres_1yr.npy` | 365 jours | Grille annuelle basse résolution. |
| `grille-hires-1an` | `ressources/grilles/grid_hires_1yr.npy` | 365 jours | Grille annuelle haute résolution, calcul long. |
| `grille-lowres-stabilisee` | `ressources/grilles/grid_lowres_stabilized.npy` | 730 jours, garde la deuxième année | Grille stabilisée basse résolution. |
| `grille-hires-stabilisee` | `ressources/grilles/grid_hires_stabilized.npy` | 730 jours, garde la deuxième année | Grille stabilisée haute résolution, calcul le plus lourd. |
| `temperatures-12mois` | `ressources/12_mois/*.csv` | dérive `grid_lowres_1yr.npy` | Viewer `affichage_3D_rapide.py`. |
| `albedo-surface-nasa` | `ressources/albedo/albedo01.csv` à `albedo12.csv` | NASA POWER | Albédo de surface mensuel. |

Groupes :

| Groupe | Contenu |
| --- | --- |
| `tout-rapide` / `grilles-rapides` | les deux grilles rapides |
| `tout-standard` / `grilles-standard` | les deux grilles annuelles |
| `tout-complet` / `grilles-toutes` | rapides, annuelles et stabilisées |
| `grilles-stabilisees` | les deux grilles stabilisées |
| `donnees-derivees` | `temperatures-12mois` |

## Options utiles

```bash
--fast-days 3              # durée des grilles rapides
--sans-convection          # coupe les deux convections
--convection forcee        # forcée seule
--convection naturelle     # naturelle seule
--vent 4.0                 # vent constant
--vent-api                 # vent NASA/cache, déconseillé sur grille complète
--output-dir /tmp/test     # sortie de test hors ressources/
--dtype float32            # fichiers plus petits, moins précis
--monthly-source-grid PATH # source pour temperatures-12mois
--albedo-year 2023         # année NASA POWER
--api-sleep 0.1            # pause entre appels API albédo
--api-timeout 30           # timeout API albédo
```

Chaque grille `.npy` écrit aussi un `.npy.json` avec convection, vent, durée,
résolution, moteur et temps de calcul.

## Entrées lues par les grilles

| Donnée | Fichier actif | Statut |
| --- | --- | --- |
| Albédo de surface | `ressources/albedo/albedo01.csv` à `albedo12.csv` | générable par `albedo-surface-nasa` |
| Albédo des nuages | `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | source externe locale |
| Humidité RZSM | `ressources/capacite_humidite/average_rzsm_tout.csv` | source externe locale |
| Continents | `ressources/carte/ne_110m_admin_0_countries.shp` | source externe locale |

Si une entrée change, régénérer seulement les grilles nécessaires, par exemple :

```bash
python3 outils_generation_donnees/generer_donnees.py --run tout-rapide --force --yes
python3 outils_generation_donnees/generer_donnees.py --run grille-hires-rapide --force --yes
```

Autres sources locales non générées par code : `ressources/cotes/` pour les
contours Natural Earth et `documents_sources/*.pdf` pour les documents copiés
depuis les groupes.

## Visualisation

Les scripts de planisphère et de sphère lisent les grilles, ils ne les
régénèrent pas :

```bash
python3 codes_python/visualisation/modele_planisphere_haute_res.py --grille rapide
python3 codes_python/visualisation/modele_sphere_haute_res.py --grille rapide
```

Variantes : `--grille auto`, `--grille rapide`, `--grille 1an`,
`--grille stabilisee`.
