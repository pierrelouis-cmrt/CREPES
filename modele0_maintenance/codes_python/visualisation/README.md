# Visualisations

Ces scripts affichent les températures déjà calculées ; ils ne régénèrent pas
les grilles. Le choix et la création des données sont expliqués dans le
[README des générateurs](../../outils_generation_donnees/README.md).

## Scripts

| Script | Affichage | Donnée lue |
| --- | --- | --- |
| `modele_planisphere_basse_res.py` | Planisphère interactif basse résolution. | Grille `.npy` basse résolution. |
| `modele_planisphere_haute_res.py` | Planisphère interactif haute résolution. | Grille `.npy` haute résolution. |
| `modele_sphere_basse_res.py` | Globe 3D interactif basse résolution. | Grille `.npy` basse résolution. |
| `modele_sphere_haute_res.py` | Globe 3D interactif haute résolution. | Grille `.npy` haute résolution. |
| `affichage_3D_rapide.py` | Globe 3D mensuel avec curseur horaire. | `ressources/12_mois/*.csv`. |
| `interface_carte_courbe.py` | Carte cliquable et courbe du point choisi. | Moteur ponctuel. |
| `visualisation_commune.py` | Chargement des grilles, axes et contours. | Module interne. |

## Variantes de grille

Les quatre scripts de carte et de globe acceptent `--grille` : `auto`,
`rapide`, `1an` ou `stabilisee`. Le mode `auto` privilégie une grille annuelle,
puis stabilisée, puis rapide. Les cartes affichent des °C ; les grilles stockent
des kelvins. Sans shapefile de côtes, les visualisations restent disponibles
mais sans contours.

Les commandes de lancement se trouvent dans le
[README des codes Python](../README.md).
