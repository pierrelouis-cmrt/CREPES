# Générateur d'albédo de surface

`generer_albedo_surface.py` est appelé par la cible `albedo-surface-nasa` du générateur principal. Il consulte l'API NASA POWER et écrit les douze fichiers `albedo01.csv` à `albedo12.csv` dans le format actif du modèle.

## Utilisation directe

```bash
python modele0_maintenance/outils_generation_donnees/albedo/generer_albedo_surface.py --year 2023 --dry-run
```

| Option | Rôle |
| --- | --- |
| `--year` | Année interrogée auprès de NASA POWER. |
| `--template-dir` | Dossier contenant le gabarit `albedo01.csv`. |
| `--output-dir` | Dossier de destination des douze CSV. |
| `--force` | Autorise le remplacement de CSV existants. |
| `--sleep`, `--timeout` | Réglages des appels réseau. |

Le script effectue un appel par point de grille ; il peut être long. Le format produit est décrit dans le [README des ressources albédo](../../ressources/albedo/README.md).
