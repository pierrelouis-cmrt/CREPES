# Grilles de température

Les fichiers `.npy` stockent les températures de surface précalculées pour les planisphères et les globes. Ils sont générés par `outils_generation_donnees/generer_donnees.py` et lus en mémoire mappée par les visualisations.

| Fichiers | Signification |
| --- | --- |
| `grid_lowres_fast.npy`, `grid_hires_fast.npy` | Calcul court pour essais. |
| `grid_lowres_1yr.npy`, `grid_hires_1yr.npy` | Simulation annuelle. |
| `grid_lowres_stabilized.npy`, `grid_hires_stabilized.npy` | Deux ans calculés, seconde année conservée. |

Chaque fichier a un compagnon `.npy.json` avec sa forme, le pas de temps, les réglages de convection et le temps de génération. Les tableaux ont la forme `(temps, latitude, longitude)` et stockent des kelvins.

Le choix d'une cible et les protections contre l'écrasement sont détaillés dans le [README des générateurs](../../outils_generation_donnees/README.md).
