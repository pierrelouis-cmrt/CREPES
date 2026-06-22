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

Le script d'absorbance a été déplacé dans `../ressources/`, car il produit une
ressource de données plutôt qu'un moteur du modèle :

```bash
python "modele1/ressources/absorbance CO2.py" --no-plot
```

Pour le périmètre et les hypothèses du modèle, voir le
[README parent](../README.md).
