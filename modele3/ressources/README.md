# Ressources — modèle 3

Ce dossier contient l'outil qui prépare le paquet compact consommé par le
modèle 3 et les modèles de surface suivants. Le paquet lui-même est un dossier
de données détaillées, conservé séparément.

## Générer le paquet

Depuis la racine du dépôt :

```bash
python -m modele3.ressources.generer_donnees --overwrite
```

Les options `--resolution`, `--annee`, `--output`, `--dry-run` et
`--allow-fallbacks` permettent d'adapter la génération. Sans `--overwrite`,
une sortie existante n'est pas remplacée.

## Structure

| Élément | Rôle |
| --- | --- |
| `generer_donnees.py` | Prépare le paquet compact à partir des données sources. |
| `calibrage_opacite_co2/` | Coefficients effectifs CO₂ et rapport produits par le calibrage. |
| `calibrage_opacite_h2o/` | Coefficients effectifs H₂O et rapport produits par le calibrage. |
| `calibrage_opacite_nuages/` | Coefficients effectifs de nuages lus par le runtime. |
| `donnees_precalculees/` | Paquets prêts à être lus ; données détaillées non réécrites ici. |

Pour utiliser le paquet dans une colonne, voir le
[README des codes](../codes_python/README.md).
