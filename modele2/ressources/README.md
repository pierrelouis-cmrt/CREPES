# Ressources — modèle 2

Ce dossier contient l'outil qui construit le profil vertical de référence
utilisé par le noyau radiatif du modèle 2. Les fichiers numériques produits
sont écrits par défaut dans le dossier `modele2/sorties/`.

## Générer le profil

Depuis la racine du dépôt :

```bash
python modele2/ressources/profil_vertical_atmosphere_co2.py --no-plot
```

Les options `--csv` et `--output` permettent de choisir les chemins de sortie.

## Structure

| Élément | Rôle |
| --- | --- |
| `profil_vertical_atmosphere_co2.py` | Génère le profil de pression, température, CO₂ et concentration moléculaire. |
| `../sorties/` | CSV et PNG générés par défaut. |

Pour exécuter la colonne radiative, voir le [README parent](../README.md).
