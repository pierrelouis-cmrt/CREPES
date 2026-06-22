# Visualisation

Ce dossier contient le planisphère interactif commun aux sorties `.npz` des
modèles 4 et 5. Il lit les résultats sans les modifier et affiche une tranche
temporelle de la variable choisie sur une carte mondiale.

## Lancer

Depuis la racine du dépôt, ouvrir directement une sortie :

```bash
python visualisation/planisphere.py --fichier modele4/sorties/simulation_modele4_rapide.npz
```

Pour choisir parmi les sorties des deux modèles :

```bash
python visualisation/planisphere.py --sorties modele4/sorties modele5/sorties
```

Pour enregistrer une image sans ouvrir de fenêtre :

```bash
python visualisation/planisphere.py --fichier modele5/sorties/simulation_modele5.npz --save visualisation/planisphere.png --no-show
```

## Options utiles

| Option | Rôle |
| --- | --- |
| `--fichier` | Fichier `.npz` à ouvrir directement. |
| `--sorties` | Un ou plusieurs dossiers proposés dans le menu de sélection. |
| `--variable` | Variable à afficher ; `temperature_surface_k` par défaut. |
| `--jour`, `--heure` | Tranche temporelle initialement affichée. |
| `--vmin`, `--vmax` | Bornes fixes de l'échelle de couleurs. |
| `--save` | Chemin du PNG à écrire. |
| `--no-show` | Génère l'image sans ouvrir de fenêtre. |
| `--no-tui` | Sélectionne automatiquement la sortie la plus récente. |

## Structure

| Fichier | Rôle |
| --- | --- |
| `planisphere.py` | Chargement des sorties `.npz` et affichage interactif. |

Le fichier `.npz` doit fournir une variable de forme `[temps, latitude,
longitude]`, ainsi que les axes `lat_deg` et `lon_deg`. Les sorties des
[modèle 4](../modele4/sorties/README.md) et
[modèle 5](../modele5/sorties/README.md) suivent cette convention.
