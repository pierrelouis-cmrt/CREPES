# Codes Python — modèle 1

Ce dossier contient le calcul de colonne radiative simplifiée du modèle 1.

## Lancer

Depuis la racine du dépôt :

```bash
python modele1/codes_python/modele1.py
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele1.py` | Calcule les flux infrarouges pour une surface et trois couches atmosphériques. |
| `modele1-squelette.py` | Version pédagogique incomplète pour tester le code sans les valeurs.|
| `absorbance CO2.py` | Trace le spectre CO2 RADIS/HITRAN et calcule les absorbances moyennes utilisées par le modèle. |

Le script d'absorbance écrit ses PNG dans `../sorties/` par défaut en
environnement non interactif :

```bash
python "modele1/codes_python/absorbance CO2.py" --output "modele1/sorties/absorbance_CO2.png" --no-plot
```

Pour le périmètre et les hypothèses du modèle, voir le
[README parent](../README.md).
