# Ressources — modèle 3

Ce dossier ne contient pas de script. Il garde seulement les fichiers lus par
le modèle.

| Élément | Rôle |
| --- | --- |
| `coefficients_opacite_modele3.npz` | Coefficients CO₂, H₂O et nuages du calcul long-onde. |
| `donnees_precalculees/grille_5deg_2024/donnees_colonnes_5deg_2024.npz` | Paquet compact de grille, avec sa metadata intégrée. |

Pour régénérer le paquet de grille depuis les sources locales :

```bash
python -m modele3.codes_python.generer_donnees --overwrite
```

Cette commande relance aussi les calibrages CO₂ et H₂O, qui mettent à jour
`coefficients_opacite_modele3.npz`. Le troisième élément d'opacité, côté
nuages, est simplement le paramètre gris `tau_lw_par_fraction_nuage`; il est
stocké dans le même NPZ et peut être fixé avec `--tau-lw-nuage`.
