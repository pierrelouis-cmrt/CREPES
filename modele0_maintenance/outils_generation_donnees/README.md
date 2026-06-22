# Génération des données — modèle 0

Ce dossier contient les outils qui sont necessaires ou produisent les sorties du
modèle 0. Les fichiers existants sont protégés par défaut.

## Lancer

Depuis la racine du dépôt :

```bash
# Appeler l'ensemble des ressources sans rien écrire
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status

# Afficher les cibles disponibles
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --list

# Préparer une génération rapide sans écrire
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --run tout-rapide --dry-run --yes
```

Ajouter `--force` pour remplacer une sortie existante. `--output-dir` permet
d'écrire une sortie d'essai ailleurs que dans les ressources actives.

## Structure

| Élément | Rôle |
| --- | --- |
| `generer_donnees.py` | Lance les cibles de grilles, séries mensuelles et albédo. |
| `albedo/` | Générateur des CSV d'albédo depuis NASA POWER. |

Les ressources de destination sont décrites dans le
[README parent](../ressources/README.md).
