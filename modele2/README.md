# Modèle 2 - colonne de CO₂ à 6 couches

Le modèle 2 étend le modèle 1 avec une colonne verticale de 6 couches
atmosphériques. Il calcule les transmissions, émissivités et flux infrarouges
à partir d'un profil pression-température-CO₂. Les températures restent
imposées et il n'intègre pas d'évolution climatique dans le temps.

## Lancer

Depuis la racine du dépôt :

```bash

python modele2/codes_python/modele2.py
```
Pour régénérer le profil de référence sans ouvrir de fenêtre graphique :

```bash
python modele2/ressources/profil_vertical_atmosphere_co2.py --no-plot
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Noyau radiatif de la colonne à six couches. |
| `ressources/` | Générateur du profil vertical et ses données produites. |

Chaque sous-dossier possède son propre README pour les détails d'utilisation.
