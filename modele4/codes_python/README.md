# Codes Python — modèle 4

Ce dossier regroupe les deux moteurs de surface du modèle 4 : une version de
référence qui rappelle la colonne radiative et une version rapide vectorisée.

## Lancer

Depuis la racine du dépôt :

```bash
# Diagnostic mensuel ou simulation temporelle de référence
python -m modele4.codes_python.modele4

# Moteur rapide, sortie toutes les quatre heures par défaut
python -m modele4.codes_python.rapide

# Lanceur interactif de scénarios
python -m modele4.codes_python.lancer
```

Les deux moteurs acceptent notamment `--jours`, `--dt`, `--co2`,
`--max-latitudes`, `--max-longitudes`, `--convection` et `--output`. Le moteur
de référence propose aussi `--mode mensuel|temporel`.

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele4.py` | Moteur de référence et interface en ligne de commande. |
| `rapide.py` | Moteur vectorisé pour les simulations courantes. |
| `surface.py` | Capacité thermique, flux latent et convection. |
| `lancer.py` | Interface interactive qui compose et lance des scénarios. |

Les fichiers produits sont décrits dans le
[README des sorties](../sorties/README.md).
