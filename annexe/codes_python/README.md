# Codes Python — annexes

Ce dossier contient quatre scripts autonomes : un outil de préparation du CO₂
CAMS et trois prototypes consacrés au CH₄. Ils ne font pas partie du runtime
des modèles 0 à 5.

## Prérequis

Selon le script lancé, installer `numpy`, `matplotlib`, `radis`, `cdsapi`,
`xarray` et `pandas`. Les scripts RADIS demandent un accès Internet au premier
lancement pour télécharger les données HITRAN.

## Structure

| Fichier | Rôle |
| --- | --- |
| `Fraction_massique_CO2.py` | Télécharge le CO₂ CAMS et produit un CSV mondial sur une grille de 5°. |
| `modele_ch4.py` | Colonne radiative pédagogique à six couches avec une seule bande CH₄. |
| `profil_atmosphere_ch4.py` | Génère un profil vertical standard de pression, température et CH₄. |
| `spectre_absorbance_ch4.py` | Calcule et affiche un spectre CH₄ détaillé avec RADIS/HITRAN. |

## CO₂ CAMS : fraction massique sur une grille de 5°

Avant de lancer `Fraction_massique_CO2.py`, créer un compte Copernicus
Atmosphere Data Store et remplacer la valeur de `CLE_API` dans le script par
votre clé personnelle. Ne pas versionner cette clé.

Depuis la racine du dépôt :

```bash
python annexe/codes_python/Fraction_massique_CO2.py
```

Le script demande la prévision CAMS du `2025-04-15`, aux niveaux `1`, `10`,
`300`, `500` et `1000 hPa`, puis échantillonne le fichier à 5°. Il produit :

```text
annexe/codes_python/co2_monde_5deg_final.csv
```

Les valeurs sont des fractions massiques `kg CO₂ / kg air`. Conversion
approximative en ppm :

```text
ppm = fraction_massique × 10⁶ × (28,97 / 44,01)
```

Le ZIP et le NetCDF intermédiaires sont supprimés après un traitement réussi.

## Prototype de colonne CH₄

```bash
python annexe/codes_python/modele_ch4.py
```

Le script affiche dans le terminal les six couches imposées, leurs opacités,
ainsi que le flux infrarouge sortant et le flux descendant à la surface. Il
utilise une unique bande CH₄ entre `7,3` et `8,0 µm` et des paramètres fixés :
son résultat est pédagogique, pas un calcul climatique validé.

## Profil vertical CH₄

```bash
python annexe/codes_python/profil_atmosphere_ch4.py --no-plot --csv annexe/codes_python/profil_ch4.csv
```

Les options utiles sont `--max-altitude-km`, `--step-m`,
`--surface-ch4-ppm`, `--ch4-gradient-ppm-per-km`, `--csv` et `--output`.
Sans `--no-plot`, le script affiche le profil ; avec `--output`, il exporte le
graphique demandé.

## Spectre d'absorbance CH₄

```bash
python annexe/codes_python/spectre_absorbance_ch4.py --no-plot --output annexe/codes_python/spectre_ch4.png --csv annexe/codes_python/spectre_ch4.csv
```

Par défaut, le calcul porte sur `1,90 ppm` de CH₄, `1,01325 bar`, `288,15 K`
et un trajet de `1000 m`. Les options `--ch4-ppm`, `--pressure-bar`,
`--temperature-k` et `--path-length-m` permettent de modifier ces paramètres.

Pour le périmètre général des outils, voir le [README parent](../README.md).
