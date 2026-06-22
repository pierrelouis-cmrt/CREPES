# Sorties — modèle 4

Ce dossier reçoit les fichiers `.npz` produits par les moteurs du modèle 4.
Les fichiers déjà présents sont des résultats de simulation ; ils peuvent être
remplacés en choisissant explicitement le même chemin de sortie.

## Produire une sortie

Depuis la racine du dépôt :

```bash
python -m modele4.codes_python.rapide --jours 1 --output modele4/sorties/simulation_dev.npz
```

Les moteurs écrivent notamment `temperature_surface_k`, les axes de temps et
de grille, des flux moyens et `metadata_json`. Le contenu exact varie selon le
moteur et son mode d'exécution.

## Structure

| Élément | Rôle |
| --- | --- |
| `*.npz` | Résultats de simulation du modèle 4. |
| `README.md` | Convention de sortie et commande de production. |

Pour afficher une sortie, utiliser
[`visualisation/planisphere.py`](../../visualisation/planisphere.py) depuis la
racine du dépôt.
