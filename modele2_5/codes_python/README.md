# Codes Python — modèle 2.5

Ce dossier contient le noyau radiatif de la colonne à dix couches du modèle
2.5. Il importe le profil et les paramètres préparés dans `../ressources/`.

## Lancer

Depuis la racine du dépôt :

```bash
python modele2_5/codes_python/modele2_5.py
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele2_5.py` | Construit les couches de pression, propage les flux par sous-bande et affiche les diagnostics radiatifs. |

La génération du profil, la calibration et les tests sont décrits dans le
[README des ressources](../ressources/README.md).
