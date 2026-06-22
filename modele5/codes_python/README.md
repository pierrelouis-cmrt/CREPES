# Codes Python — modèle 5

Ce dossier contient le moteur de température de surface avec échanges radiatifs
horizontaux et son outil de visualisation.

## Lancer

Depuis la racine du dépôt :

```bash
# Simulation
python -m modele5.codes_python.modele5

# Écrire un planisphère sans ouvrir de fenêtre
python -m modele5.codes_python.planisphere --fichier modele5/sorties/simulation_modele5.npz --save modele5/sorties/planisphere.png --no-show
```

Le moteur accepte notamment `--jours`, `--dt`, `--sortie-heures`, `--co2`,
`--max-latitudes`, `--max-longitudes`, `--facteur-horizontal`,
`--temperature-air` et `--output`.

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele5.py` | Intègre le bilan de surface avec le terme horizontal. |
| `planisphere.py` | Affiche ou exporte les cartes d'une sortie `.npz`. |

Les formats produits sont décrits dans le
[README des sorties](../sorties/README.md).
