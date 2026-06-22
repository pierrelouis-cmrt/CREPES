# Visualisation — modèle 0

Ces scripts affichent des grilles ou des séries déjà disponibles ; ils ne
génèrent pas les ressources eux-mêmes.

## Lancer

Depuis la racine du dépôt :

```bash
python modele0_maintenance/codes_python/visualisation/modele_planisphere_basse_res.py --grille rapide
python modele0_maintenance/codes_python/visualisation/modele_sphere_basse_res.py --grille rapide
python modele0_maintenance/codes_python/visualisation/affichage_3D_rapide.py --month janvier --hour 12
python modele0_maintenance/codes_python/visualisation/interface_carte_courbe.py
```

Les scripts de carte et de globe acceptent `--grille auto`, `rapide`, `1an` ou
`stabilisee`. Les options `--save` et `--no-show`, lorsqu'elles sont proposées,
permettent d'écrire une image sans ouvrir de fenêtre.

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele_planisphere_basse_res.py` | Planisphère interactif basse résolution. |
| `modele_planisphere_haute_res.py` | Planisphère interactif haute résolution. |
| `modele_sphere_basse_res.py` | Globe 3D basse résolution. |
| `modele_sphere_haute_res.py` | Globe 3D haute résolution. |
| `affichage_3D_rapide.py` | Globe 3D à partir des températures mensuelles. |
| `interface_carte_courbe.py` | Carte cliquable et courbe de température locale. |
| `visualisation_commune.py` | Fonctions internes de chargement et d'affichage. |

La création des grilles est décrite dans le
[README des générateurs](../../outils_generation_donnees/README.md).
