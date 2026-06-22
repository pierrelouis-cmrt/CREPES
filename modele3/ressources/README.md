# Ressources — modèle 3

Ce dossier contient le générateur du paquet compact et les données précalculées
utilisées par le modèle 3 et le modèle 4.

## Générer le paquet

Depuis la racine du dépôt :

```bash
python -m modele3.ressources.generer_donnees --overwrite
```

Le paquet produit est stocké dans `donnees_precalculees/grille_5deg_2024/`.
Son format et ses sources sont décrits dans son
[README](donnees_precalculees/grille_5deg_2024/README.md).

## Structure

| Élément | Rôle |
| --- | --- |
| `generer_donnees.py` | Prépare le paquet compact à partir des données sources. |
| `donnees_precalculees/` | Paquets prêts à être lus par le modèle. |
