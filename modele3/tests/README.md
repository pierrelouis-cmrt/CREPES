# Tests — modèle 3

Ce dossier vérifie le chargement du paquet compact, le calcul radiatif, les
coefficients d'opacité CO₂/H₂O et les invariants numériques principaux du
modèle 3. La visualisation qualitative `Absorbance_H2O.py` n'est pas un test de
calibrage et ne fait pas partie du runtime.

## Lancer

Depuis la racine du dépôt :

```bash
python modele3/tests/tester_modele3.py
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `tester_modele3.py` | Exécute les tests du paquet et du noyau radiatif. |

Les commandes de calcul de colonne sont dans le [README parent](../README.md).
