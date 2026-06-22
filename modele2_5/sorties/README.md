# Sorties — modèle 2.5

Ce dossier reçoit le profil vertical de référence généré pour le modèle 2.5.
Le profil combine l'atmosphère standard, la pression, la température et la
concentration de CO₂ sur la grille verticale choisie.

## Structure

| Fichier | Rôle |
| --- | --- |
| `profil_vertical_atmosphere_co2.csv` | Valeurs numériques du profil vertical. |
| `profil_vertical_atmosphere_co2.png` | Graphique de diagnostic du profil. |

## Régénérer le profil

Depuis la racine du dépôt :

```bash
python modele2_5/ressources/profil_vertical_atmosphere_co2.py --no-plot
```

Par défaut, cette commande écrit les deux fichiers de ce dossier. Les options
`--max-altitude-km`, `--step-m`, `--surface-co2-ppm` et
`--co2-gradient-ppm-per-km` permettent de modifier le profil. `--csv` et
`--output` permettent de choisir d'autres chemins de sortie.

Le noyau radiatif qui exploite ce profil est décrit dans le
[README des codes Python](../codes_python/README.md).
