# CREPES - modèle 0 maintenance

Projet climat combinant les apports des quatre groupes sources. Le moteur
stable par défaut reprend le modèle 4 Carcajous. Les autres briques sont soit
branchées de façon explicite, soit conservées comme sources pour reprise
ultérieure.

## Lancer une simulation

Depuis `CREPES/modele0_maintenance/` :

```bash
python3 codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 730
```

Sans fenêtre graphique :

```bash
python3 codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

## Convection

Par défaut, les deux convections sont actives :

- convection forcée Chevreaux ;
- convection naturelle Ornithorynquietant.

Le lancement standard utilise un vent constant de `2.5 m/s`, sans appel réseau.
Pour désactiver très simplement les deux convections :

```bash
python3 codes_python/modele_courbe.py --sans-convection --no-plot
```

Équivalent explicite :

```bash
python3 codes_python/modele_courbe.py --convection aucune --no-plot
```

Ne garder que la convection forcée Chevreaux :

```bash
python3 codes_python/modele_courbe.py --convection forcee --vent 2.5 --temperature-air 288 --no-plot
```

Ne garder que la convection naturelle Ornithorynquietant :

```bash
python3 codes_python/modele_courbe.py --convection naturelle --temperature-air 288 --no-plot
```

Pour utiliser le vent journalier NASA/cache au lieu du vent constant :

```bash
python3 codes_python/modele_courbe.py --vent-api --no-plot
```

## Génération des données

Le script global recommandé est :

```bash
python3 outils_generation_donnees/generer_donnees.py
```

La documentation détaillée est dans `outils_generation_donnees/README.md`.

Il affiche un menu console pour :

- voir l'état des ressources ;
- régénérer les grilles rapides ;
- régénérer les grilles annuelles ;
- choisir une cible précise ;
- régénérer les CSV `12_mois` dérivés d'une grille annuelle ;
- lancer explicitement la régénération d'albédo via NASA POWER.

Commandes non interactives utiles :

```bash
python3 outils_generation_donnees/generer_donnees.py --status
python3 outils_generation_donnees/generer_donnees.py --list
python3 outils_generation_donnees/generer_donnees.py --run tout-rapide --force --yes
python3 outils_generation_donnees/generer_donnees.py --run grille-hires-rapide --force --yes
python3 outils_generation_donnees/generer_donnees.py --run tout-standard --dry-run --yes
python3 outils_generation_donnees/generer_donnees.py --run temperatures-12mois --force --yes
python3 outils_generation_donnees/generer_donnees.py --run albedo-surface-nasa --dry-run --yes
```

Les grilles rapides générées par défaut couvrent 7 jours avec les deux
convections actives :

- `ressources/grilles/grid_lowres_fast.npy` ;
- `ressources/grilles/grid_hires_fast.npy`.

Les fichiers `.npy.json` associés indiquent la durée, la résolution, le mode de
convection, le vent et le moteur utilisé.

Les données actives générées par code restent générables par ces cibles. Les
autres entrées (`CERES`, RZSM, shapefiles, PDF sources) sont des sources locales
à fournir ou remplacer manuellement.

## Modules conservés

Non branchés volontairement :

- `codes_python/physique/diffusion.py` : conservé, mais le flux de surface reste ambigu.
- gaz à effet de serre : non intégré pour l'instant.

## Visualisations

```bash
python3 codes_python/visualisation/modele_planisphere_haute_res.py --grille rapide
python3 codes_python/visualisation/modele_sphere_haute_res.py --grille rapide
python3 codes_python/visualisation/affichage_3D_rapide.py --month janvier --hour 12
python3 codes_python/visualisation/interface_carte_courbe.py
```

Options de grilles pour les planisphères et sphères : `--grille auto`,
`--grille rapide`, `--grille 1an`, `--grille stabilisee`.

## Structure

| Dossier | Rôle |
| --- | --- |
| `codes_python/` | moteur, modules physiques et visualisations |
| `ressources/` | données finales utilisées par le moteur |
| `outils_generation_donnees/` | générateur principal, générateur albédo NASA et README de génération |
| `documents_sources/` | PDF de synthèse copiés depuis les groupes |
| `PROVENANCE.md` | traçabilité courte des apports |
| `THEORIE.md` | résumé théorique utile |

## Dépendances

Installer les dépendances du fichier `requirements.txt`, ou utiliser un
environnement Python déjà compatible avec `numpy`, `pandas`, `scipy`,
`matplotlib`, `xarray`, `geopandas`, `cartopy`, `pyshp` et `requests`.
