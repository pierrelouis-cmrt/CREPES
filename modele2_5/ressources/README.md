# Ressources — modèle 2.5

Ce dossier contient les scripts auxiliaires du modèle 2.5, les tests numériques et les fichiers générés du profil atmosphérique.

## Lancer les scripts

Depuis la racine du dépôt :

```bash
# Générer le profil global pression-température-CO₂
python modele2_5/ressources/profil_vertical_atmosphere_co2.py --no-plot

# Lancer les tests numériques
python modele2_5/ressources/tester_modele2_5.py

# Recalculer le facteur d’échelle des coefficients optiques
python modele2_5/ressources/calibrer_coefficients_optiques.py
```

Le premier script écrit par défaut ses fichiers dans `données/` :

```text
données/profil_vertical_atmosphere_co2.csv
données/profil_vertical_atmosphere_co2.png
```

Les options `--csv` et `--output` permettent de choisir d’autres emplacements.

## Structure

| Élément | Rôle |
| --- | --- |
| `profil_vertical_atmosphere_co2.py` | Génère le profil complet de pression, température et CO₂. |
| `tester_modele2_5.py` | Vérifie les cas limites, la grille verticale et la réponse au CO₂. |
| `calibrer_coefficients_optiques.py` | Calibre les coefficients optiques effectifs sur la cible de forçage. |
| `données/` | CSV et PNG générés par le profil atmosphérique. |
| `.cache/` | Cache local de Matplotlib créé à l’exécution. |

Pour lancer le modèle radiatif principal, voir le [README parent](../README.md).
