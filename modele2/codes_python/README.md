# Codes Python — modèle 2

Ce dossier contient le noyau radiatif du modèle 2 : la propagation infrarouge
dans six couches, avec une opacité effective par bande CO₂.

## Lancer

Depuis la racine du dépôt :

```bash
python modele2/codes_python/modele2.py
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele2.py` | Construit les couches, calcule leurs propriétés radiatives et affiche les flux au sommet et à la surface. |

Le profil vertical utilisé par le calcul est documenté dans le
[README des ressources](../ressources/README.md).
