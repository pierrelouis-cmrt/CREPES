# Ressources — modèle 1

Ce dossier contient les outils et données auxiliaires du modèle 1.

## Lancer le script d'absorbance

Depuis la racine du dépôt :

```bash
python "modele1/ressources/absorbance CO2.py"
```

En environnement non interactif, ou avec `--no-plot`, le script écrit son image
par défaut dans `modele1/ressources/données/absorbance_CO2.png`.

Pour forcer l'écriture du fichier sans ouvrir de fenêtre graphique :

```bash
python "modele1/ressources/absorbance CO2.py" --no-plot
```

## Structure

| Élément | Rôle |
| --- | --- |
| `absorbance CO2.py` | Trace le spectre CO2 RADIS/HITRAN et calcule les absorbances moyennes utilisées par les modèles 1 et 2. |
| `données/` | Images générées par le script d'absorbance. |
