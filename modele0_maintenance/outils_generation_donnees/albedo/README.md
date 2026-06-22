# Générateur d'albédo — modèle 0

Ce dossier contient l'outil appelé par la cible `albedo-surface-nasa` du
générateur principal. Il interroge NASA POWER et produit les douze CSV
d'albédo attendus par le modèle 0.

## Lancer

Depuis la racine du dépôt :

```bash
python modele0_maintenance/outils_generation_donnees/albedo/generer_albedo_surface.py --year 2023 --dry-run
```

Utiliser `--output-dir` pour écrire ailleurs, `--force` pour autoriser le
remplacement et `--sleep` ou `--timeout` pour régler les appels réseau.
L'exécution complète est longue car elle contacte le service pour chaque point
de grille.

## Structure

| Fichier | Rôle |
| --- | --- |
| `generer_albedo_surface.py` | Télécharge, convertit et écrit les CSV mensuels d'albédo. |

Le lancement groupé est expliqué dans le
[README du générateur parent](../README.md).
