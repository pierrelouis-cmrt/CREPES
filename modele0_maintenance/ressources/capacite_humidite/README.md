# Capacité thermique et humidité du sol

`average_rzsm_tout.csv` est une entrée locale d'humidité de la zone racinaire (RZSM). Le module `physique/capacite_surface.py` la moyenne sur une grille de 1° puis en déduit la capacité thermique du sol.

## Format attendu

| Colonne | Contenu |
| --- | --- |
| `lat` | Latitude en degrés. |
| `lon` | Longitude en degrés, dans `[-180, 180]` ou `[0, 360]`. |
| `RZSM` | Humidité relative de la zone racinaire. |

Cette source externe locale n'est pas générée par le projet. Si elle est absente ou illisible, le moteur utilise une capacité sèche constante.
