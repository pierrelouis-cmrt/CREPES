# Modèle 3 - colonne radiative locale

Le modèle 3 calcule les flux radiatifs d'une colonne locale pour un point
`(latitude, longitude)` et une température de surface imposée. Il ne fait pas
évoluer `T_surface(t)` : cette boucle reste réservée au modèle 4.

Cas de référence :

```text
Paris : lat = 48.8566, lon = 2.3522
```

## Fichiers

La première version avait été découpée en beaucoup de petits modules. Cette
version garde un compromis plus lisible : un fichier principal et un petit
dossier `physique/` pour les formules indépendantes.

| Fichier | Rôle |
| --- | --- |
| `modele3.py` | Construction de la colonne, propagation des flux, diagnostics et CLI. |
| `physique/calculs.py` | Formules physiques élémentaires : solaire, Planck, opacités, albédo nuage, masses colonne. |
| `physique/README.md` | Rôle du sous-dossier physique. |
| `donnees.py` | Chargement d'un extrait JSON, lecture ERA5 optionnelle et secours simple. |
| `preparer_point.py` | Extrait un point compact versionnable depuis `/ressources`. |
| `donnees_exemple/paris_2024_m07.json` | Extrait léger pour exécuter Paris sans les gros fichiers locaux. |
| `plan.md` | Plan initial du modèle 3. |
| `THEORIE.md` | Récapitulatif théorique et sources scientifiques. |
| `tests/tester_modele3.py` | Tests numériques minimaux. |

Les anciens fichiers `constantes.py`, `types.py`, `solaire.py`, `spectre.py`,
`colonne.py` et `radiatif.py` ont été supprimés. Leur contenu utile est
maintenant réparti entre :

- `modele3.py` pour la logique du modèle ;
- `physique/calculs.py` pour les calculs physiques réutilisables.

Le but est de retrouver une séparation visuelle proche du modèle 0, sans
recréer trop de scripts.

## Comparaison avec le plan

Ce qui est implémenté :

- colonne locale pour un point `(lat, lon)` ;
- pression de surface locale utilisée comme base de la colonne ;
- couches en pression avec les niveaux `850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa` ;
- profil local `T(p)` et `q(p)` depuis JSON/ERA5 quand disponible ;
- CO2 uniforme configurable, par défaut `420 ppm` ;
- opacité H2O simple ajoutée à l'opacité CO2 avant la transmission ;
- nuages simples en court-onde et long-onde ;
- albédo et émissivité de surface ;
- sorties `SW`, `LW`, `OLR`, flux net et diagnostics par couche/bande ;
- validation qualitative sur les flux ERA5 disponibles.

Ce qui reste volontairement simplifié :

- pas d'évolution de la température de surface ;
- pas de grille globale ;
- pas de RADIS/HITRAN, pas de correlated-k ;
- pas d'ozone, CH4, N2O, aérosols ou microphysique nuageuse ;
- pas de transfert court-onde atmosphérique complet, donc le `SW_absorbe_surface`
  reste trop élevé sur Paris ;
- l'émissivité MODIS n'est plus lue directement depuis HDF4 dans le modèle :
  on utilise la valeur de l'extrait JSON ou un secours simple (`0.98`, `0.985`
  sur océan).

La simplification principale est informatique, pas physique : le calcul reste
celui prévu par le plan, mais avec moins d'abstractions Python. Quand une
formule physique doit être modifiée, il faut commencer par regarder
`physique/calculs.py`. Quand on veut comprendre le déroulement complet du
modèle, il faut lire `modele3.py`.

## Lancer

Depuis la racine du dépôt :

```bash
./.venv/bin/python -m modele3.modele3 \
  --lat 48.8566 \
  --lon 2.3522 \
  --mois 7 \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw
```

Sans les gros fichiers de `ressources/`, utiliser l'extrait versionné :

```bash
./.venv/bin/python -m modele3.modele3 \
  --donnees-extraites modele3/donnees_exemple/paris_2024_m07.json \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw
```

Sortie JSON complète :

```bash
./.venv/bin/python -m modele3.modele3 \
  --donnees-extraites modele3/donnees_exemple/paris_2024_m07.json \
  --temperature-surface 293.0 \
  --json
```

Tests :

```bash
./.venv/bin/python modele3/tests/tester_modele3.py
```

## Données volumineuses et Git

Le dossier racine `ressources/` est ignoré par Git, ce qui est correct : il
peut contenir des NetCDF de plusieurs centaines de Mo à plusieurs Go.

La solution retenue est en deux niveaux :

1. En local, si `ressources/` est présent et que `xarray` peut lire les NetCDF,
   le modèle essaie de récupérer les champs ERA5 utiles.
2. Pour GitHub, on utilise un extrait JSON compact avec les profils et flux
   utiles. Ce JSON est petit, versionnable et suffisant pour reproduire le cas
   prédéfini.

Créer ou remplacer un extrait :

```bash
./.venv/bin/python -m modele3.preparer_point \
  --lat 48.8566 \
  --lon 2.3522 \
  --mois 7 \
  --output modele3/donnees_exemple/paris_2024_m07.json
```

Un autre membre du groupe peut donc cloner le dépôt et lancer le modèle 3 sur
Paris même sans posséder les fichiers de `ressources/`.

## Validation Paris juillet

Commande utilisée :

```bash
./.venv/bin/python -m modele3.modele3 \
  --donnees-extraites modele3/donnees_exemple/paris_2024_m07.json \
  --temperature-surface 293.0 \
  --moyenne-journaliere-sw
```

| Flux | Modèle 3 W/m² | ERA5 W/m² | Écart W/m² |
| --- | ---: | ---: | ---: |
| `LW_down_surface` | 350.38 | 361.32 | -10.94 |
| `OLR` | 244.78 | 244.88 | -0.10 |
| `SW_absorbe_surface` | 268.53 | 178.70 | +89.83 |

Le long-onde est dans l'ordre de grandeur attendu pour ce cas. Le court-onde
reste surestimé et il faut le signaler clairement dans les comptes rendus :
`SW` signifie `ShortWave`, donc rayonnement solaire court-onde. Ici le modèle 3
obtient `SW_absorbe_surface = 268.53 W/m²`, alors qu'ERA5 donne
`178.70 W/m²`, soit `+89.83 W/m²`.

Cet écart ne vient pas d'une erreur de signe : le modèle 3 applique seulement
la géométrie solaire moyenne journalière, l'albédo de surface et un albédo
nuage effectif. ERA5 donne un flux court-onde net à la surface après transfert
atmosphérique complet. Pour le modèle 4, il faudra soit ajouter une
transmission atmosphérique SW simple, soit assumer une correction/calibration
court-onde avant d'intégrer `T_surface(t)`.

## Limites assumées

- Pas d'évolution de température de surface.
- Pas de grille globale.
- Pas de dynamique atmosphérique ni d'échanges horizontaux.
- Pas de rétroaction de la surface sur `T(p)`.
- Pas de RADIS/HITRAN ligne par ligne.
- Pas d'ozone, CH4, N2O ni microphysique nuageuse détaillée.
- Les coefficients H2O et nuages sont effectifs et calibrés seulement pour
  obtenir des ordres de grandeur cohérents sur le cas Paris.
