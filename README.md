# CREPES

Dépôt de travail pour comparer et développer plusieurs modèles climat.

## Organisation

| Dossier | Rôle |
| --- | --- |
| `modele0_maintenance/` | Ancien modèle combiné, conservé comme référence stable. |
| `modele1/` | Emplacement réservé pour un nouveau modèle. |
| `modele2/` | Colonne atmosphérique CO2 à 6 couches avec noyau radiatif infrarouge simplifié. |
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

## Modèle 2

Lancer le noyau radiatif du modèle 2 :

```bash
./.venv/bin/python modele2/modele2.py
```

Régénérer le profil vertical de pression et de CO2 :

```bash
./.venv/bin/python modele2/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/profil_vertical_atmosphere_co2.png --csv modele2/profil_vertical_atmosphere_co2.csv --no-plot
```

La documentation détaillée du modèle 2 est dans `modele2/README.md`.
