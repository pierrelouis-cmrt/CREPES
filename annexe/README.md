# Annexes

Ce dossier contient des outils de préparation ou d'exploration qui ne font pas
partie de l'exécution normale des modèles.

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Scripts annexes, indépendants des moteurs de calcul. |
| `README.md` | Présentation, prérequis et précautions d'utilisation. |

## Télécharger et préparer le CO₂ CAMS

`codes_python/Fraction_massique_CO2.py` télécharge une prévision de CO₂ CAMS
via l'API Copernicus Atmosphere Data Store, extrait le NetCDF et construit un
CSV mondial sur une grille de `5° × 5°` aux niveaux de pression `1`, `10`,
`300`, `500` et `1000 hPa`.

Les valeurs du CSV restent des fractions massiques, en `kg de CO₂ / kg d'air`.
Pour les convertir approximativement en ppm :

```text
ppm = fraction_massique × 10⁶ × (28,97 / 44,01)
```

### Prérequis

- Un compte Copernicus ADS et une clé API valide.
- Les dépendances `cdsapi`, `xarray`, `pandas` et `numpy`.
- Renseigner la clé dans la constante `CLE_API` du script avant l'exécution.

### Lancer

Depuis la racine du dépôt :

```bash
python annexe/codes_python/Fraction_massique_CO2.py
```

Le script crée `co2_monde_5deg_final.csv` dans `annexe/codes_python/`. Le ZIP
et le NetCDF téléchargés sont temporaires et sont supprimés en fin de traitement
réussi. Aucun modèle ne lit actuellement ce CSV automatiquement.
