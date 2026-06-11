# CREPES - modele 0 maintenance

Projet climat combinant les apports des quatre groupes sources. Le moteur
stable par defaut reprend le modele 4 Carcajous. Les autres briques sont soit
branchees de facon explicite, soit conservees comme sources pour reprise
ulterieure.

## Lancer une simulation

Depuis `CREPES/modele0_"maintenance"/` :

```bash
python3 codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 730
```

Sans fenetre graphique :

```bash
python3 codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

## Convection

Par defaut, les deux convections sont actives :

- convection forcee Chevreaux ;
- convection naturelle Ornithorynquietant.

Le lancement standard utilise un vent constant de `2.5 m/s`, sans appel reseau.
Pour desactiver tres simplement les deux convections :

```bash
python3 codes_python/modele_courbe.py --sans-convection --no-plot
```

Equivalent explicite :

```bash
python3 codes_python/modele_courbe.py --convection aucune --no-plot
```

Ne garder que la convection forcee Chevreaux :

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

## Generation des donnees

Le script global recommande est :

```bash
python3 outils_generation_donnees/generer_donnees.py
```

La documentation detaillee est dans `outils_generation_donnees/README.md`.

Il affiche un menu console pour :

- voir l'etat des ressources ;
- regenerer les grilles rapides ;
- regenerer les grilles annuelles ;
- choisir une cible precise ;
- regenerer les CSV `12_mois` derives d'une grille annuelle ;
- lancer explicitement la regeneration d'albedo via NASA POWER.

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

Les grilles rapides generees par defaut couvrent 7 jours avec les deux
convections actives :

- `ressources/grilles/grid_lowres_fast.npy` ;
- `ressources/grilles/grid_hires_fast.npy`.

Les fichiers `.npy.json` associes indiquent la duree, la resolution, le mode de
convection, le vent et le moteur utilise.

Les donnees actives generees par code restent generables par ces cibles. Les
autres entrees (`CERES`, RZSM, shapefiles, PDF sources) sont des sources locales
a fournir ou remplacer manuellement.

## Modules conserves

Non branches volontairement :

- `codes_python/physique/diffusion.py` : conserve, mais le flux de surface reste ambigu.
- gaz a effet de serre : non integre pour l'instant.

## Visualisations

```bash
python3 codes_python/visualisation/modele_planisphere_haute_res.py --grille rapide
python3 codes_python/visualisation/modele_sphere_haute_res.py --grille rapide
python3 codes_python/visualisation/affichage_3D_rapide.py --month janvier --hour 12
python3 codes_python/visualisation/interface_carte_courbe.py
```

Options de grilles pour les planispheres et spheres : `--grille auto`,
`--grille rapide`, `--grille 1an`, `--grille stabilisee`.

## Structure

| Dossier | Role |
| --- | --- |
| `codes_python/` | moteur, modules physiques et visualisations |
| `ressources/` | donnees finales utilisees par le moteur |
| `outils_generation_donnees/` | generateur principal, generateur albedo NASA et README de generation |
| `documents_sources/` | PDF de synthese copies depuis les groupes |
| `PROVENANCE.md` | tracabilite courte des apports |
| `THEORIE.md` | resume theorique utile |

## Dependances

Installer les dependances du fichier `requirements.txt`, ou utiliser un
environnement Python deja compatible avec `numpy`, `pandas`, `scipy`,
`matplotlib`, `xarray`, `geopandas`, `cartopy`, `pyshp` et `requests`.
