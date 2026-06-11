# CREPES

Depot de travail pour comparer et developper plusieurs modeles climat.

## Organisation

| Dossier | Role |
| --- | --- |
| `modele0_maintenance/` | Ancien modele combine, conserve comme reference stable. |
| `modele1/` | Emplacement reserve pour un nouveau modele. |
| `plan d'attaque/` | Plan de travail CO2 multicouche simplifie. |

Chaque nouveau modele peut maintenant avoir son propre dossier a la racine,
avec son code, ses ressources, sa documentation et ses dependances locales.

## Modele 0

Lancer une simulation courte depuis la racine :

```bash
python3 modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Inventorier les donnees du modele 0 :

```bash
python3 modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

La documentation complete du modele 0 est dans
`modele0_maintenance/README.md`.
