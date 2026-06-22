# Codes Python — modèle 0

Ce dossier contient le moteur ponctuel maintenu du modèle 0 et les modules
dont il dépend. Le moteur intègre un bilan de surface simplifié avec un pas de
temps par défaut de 1 800 secondes.

## Lancer une simulation

Depuis la racine du dépôt :

```bash
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Exemples utiles :

```bash
# Affichage de deux jours pour Paris
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2

# Sans convection
python modele0_maintenance/codes_python/modele_courbe.py --sans-convection --no-plot

# Convection forcée avec vent constant
python modele0_maintenance/codes_python/modele_courbe.py --convection forcee --vent 3 --temperature-air 288 --no-plot
```

Les principales options sont `--lat`, `--lon`, `--days`, `--jour-affiche`,
`--no-plot`, `--convection`, `--vent`, `--temperature-air` et `--vent-api`.

## Structure

| Élément | Rôle |
| --- | --- |
| `modele_courbe.py` | Point d'entrée de la simulation ponctuelle. |
| `fonctions.py` | Préparation des séries d'entrée du bilan de surface. |
| `bibliotheque.py` | Constantes et compatibilité avec les scripts historiques. |
| `chemins.py` | Résolution centralisée des chemins de ressources. |
| `physique/` | Modules de calcul des termes physiques. |
| `visualisation/` | Affichage des résultats et grilles existantes. |

Les ressources requises sont recensées dans le
[README des ressources](../ressources/README.md).
