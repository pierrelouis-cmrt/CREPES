# Ressources — modèle 3

Ce dossier regroupe les fichiers lus par le modèle et les scripts directement
liés à leur génération ou à leur visualisation.

| Élément | Rôle |
| --- | --- |
| `coefficients_opacite_modele3.npz` | Coefficients CO₂, H₂O et nuages du calcul long-onde. |
| `donnees_precalculees/grille_5deg_2024/donnees_colonnes_5deg_2024.npz` | Paquet compact de grille, avec sa metadata intégrée. |
| `generer_donnees.py` | Génération du paquet compact, puis lancement des calibrages CO₂/H₂O. |
| `Absorbance_H2O.py` | Visualisation qualitative des bandes et du continuum d'absorption H₂O. |

Pour régénérer le paquet de grille depuis les sources locales :

```bash
python -m modele3.ressources.generer_donnees --overwrite
```

Cette commande relance aussi les calibrages CO₂ et H₂O, qui mettent à jour
`coefficients_opacite_modele3.npz`. Le troisième élément d'opacité, côté
nuages, est simplement le paramètre gris `tau_lw_par_fraction_nuage`; il est
stocké dans le même NPZ et peut être fixé avec `--tau-lw-nuage`.

Pour visualiser l'absorption qualitative de la vapeur d'eau :

```bash
python modele3/ressources/Absorbance_H2O.py
```

Cette visualisation n'est pas utilisée par `modele3.py` et ne modifie pas les
coefficients d'opacité.
