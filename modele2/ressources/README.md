# Ressources — modèle 2

Ce dossier contient l'outil qui construit le profil vertical de référence
utilisé par le noyau radiatif du modèle 2. Les fichiers numériques produits
restent dans le sous-dossier `données/`, qui est volontairement documenté à
part par son contenu.

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
| `données/` | CSV et PNG générés ; données détaillées non décrites dans ce README. |

Pour exécuter la colonne radiative, voir le [README parent](../README.md).
