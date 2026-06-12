# CREPES

Dépôt de travail pour comparer et développer plusieurs modèles climat.

## Organisation

| Dossier | Rôle |
| --- | --- |
| `modele0_maintenance/` | Ancien modèle combiné, conservé comme référence stable. |
| `modele1/` | Emplacement réservé pour un nouveau modèle. |
| `plan d'attaque/` | Plan de travail CO2 multicouche simplifié. |

Chaque nouveau modèle peut maintenant avoir son propre dossier à la racine,
avec son code, ses ressources, sa documentation et ses dépendances locales.

## Modèle 0

Lancer une simulation courte depuis la racine :

```bash
python3 modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Inventorier les données du modèle 0 :

```bash
python3 modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

La documentation complète du modèle 0 est dans
`modele0_maintenance/README.md`.
