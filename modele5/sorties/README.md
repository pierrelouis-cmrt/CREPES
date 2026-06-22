# Sorties — modèle 5

Ce dossier contient les fichiers `.npz` produits par le modèle 5 et les images
qui peuvent être exportées par le planisphère.

## Produire et afficher une sortie

Depuis la racine du dépôt :

```bash
python -m modele5.codes_python.modele5 --output modele5/sorties/simulation_dev.npz
python -m modele5.codes_python.planisphere --fichier modele5/sorties/simulation_dev.npz --save modele5/sorties/planisphere_dev.png --no-show
```

Le fichier `.npz` contient la température de surface, les axes de temps et de
grille, les flux moyens de surface et les diagnostics d'échange horizontal.

## Structure

| Élément | Rôle |
| --- | --- |
| `*.npz` | Résultats de simulation du modèle 5. |
| `*.png` | Exports facultatifs du planisphère. |
| `README.md` | Convention de sortie et commandes associées. |

Pour modifier les paramètres de calcul, voir le
[README des codes](../codes_python/README.md).
