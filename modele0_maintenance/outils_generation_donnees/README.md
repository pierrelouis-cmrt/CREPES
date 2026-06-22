# Génération des données

Ce dossier produit les grilles de température, les fichiers mensuels du viewer
3D et les albédo de surface. Les sources externes locales ne sont pas recréées
ici.

## Script principal

`generer_donnees.py` centralise les cibles et protège les fichiers existants.

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --list
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --run tout-rapide --dry-run --yes
```

Pour remplacer une sortie existante, ajouter `--force`. `--yes` supprime les
questions interactives et `--output-dir` permet d'écrire une sortie de test
hors de `ressources/`.

## Cibles

| Cible | Sortie | Usage |
| --- | --- | --- |
| `grille-lowres-rapide` / `grille-hires-rapide` | `grid_*_fast.npy` | Essais visuels courts. |
| `grille-lowres-1an` / `grille-hires-1an` | `grid_*_1yr.npy` | Visualisations annuelles. |
| `grille-lowres-stabilisee` / `grille-hires-stabilisee` | `grid_*_stabilized.npy` | Seconde année calculée. |
| `temperatures-12mois` | `ressources/12_mois/*.csv` | Globe mensuel rapide. |
| `albedo-surface-nasa` | `ressources/albedo/albedo01.csv` à `albedo12.csv` | Albédo de surface. |

Les groupes `tout-rapide`, `tout-standard`, `tout-complet`,
`grilles-rapides`, `grilles-standard` et `grilles-stabilisees` lancent plusieurs
cibles. Chaque grille reçoit un fichier compagnon `.npy.json`.

## Paramètres et entrées

`--fast-days` règle la durée des sorties rapides et `--dtype float32` réduit la
taille des fichiers. Les options de convection sont détaillées dans le
[README des codes](../codes_python/README.md). Les emplacements et formats des
entrées sont documentés dans le [README des ressources](../ressources/README.md).

La cible `albedo-surface-nasa` appelle le script du sous-dossier
[`albedo/`](albedo/README.md) et nécessite le réseau.
