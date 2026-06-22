# Codes Python — modèle 0 (maintenance)

Ce dossier contient le moteur de simulation ponctuelle du **modèle 0** de
CREPES, ses briques physiques et ses outils de visualisation. Il constitue une
version maintenue du modèle historique : les chemins sont centralisés, les
entrées sont préparées avant le calcul et les modules restent importables
séparément.

Le moteur calcule l'évolution temporelle de la température de surface d'un
point géographique à partir du rayonnement solaire, des albédo, de la capacité
thermique du sol, de la chaleur latente et, selon la configuration, de la
convection.

> Le modèle 0 est conservé comme référence historique. Les modèles plus récents
> du dépôt sont documentés à la racine du projet.

## Installation

Depuis la racine du dépôt, créer puis activer un environnement virtuel si
nécessaire :

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

python -m pip install -r modele0_maintenance/requirements.txt
```

Les dépendances principales sont `numpy`, `pandas`, `scipy` et `matplotlib`.
Certaines fonctionnalités demandent en plus `xarray`/`netCDF4` (nuages CERES),
`geopandas`/`shapely` (détection des continents), `pyshp` (côtes), `cartopy`
(interface cartographique) ou `requests` (données NASA POWER).

## Lancer une simulation ponctuelle

Lancer cette commande depuis la racine du dépôt :

```bash
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Sans `--no-plot`, une fenêtre affiche la température de surface, les albédo,
le flux latent et la capacité thermique. Par défaut, la simulation dure deux
ans et les deux formes de convection sont actives.

Exemples utiles :

```bash
# Simulation d'un an avec affichage
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 365

# Bilan radiatif sans convection
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --sans-convection --no-plot

# Convection forcée uniquement, vent constant de 3 m/s
python modele0_maintenance/codes_python/modele_courbe.py --convection forcee --vent 3 --temperature-air 288 --no-plot

# Vent journalier NASA POWER, ou cache local s'il existe
python modele0_maintenance/codes_python/modele_courbe.py --vent-api --no-plot
```

Options principales :

| Option | Rôle |
| --- | --- |
| `--lat`, `--lon` | Latitude et longitude du point en degrés. |
| `--days` | Durée de la simulation en jours. |
| `--jour-affiche` | Jour mis en évidence sur le graphique. |
| `--no-plot` | Désactive l'affichage graphique. |
| `--convection` | `toutes` (défaut), `aucune`, `forcee` ou `naturelle`. |
| `--sans-convection` | Raccourci pour `--convection aucune`. |
| `--temperature-air` | Température de l'air en kelvins pour la convection. |
| `--vent` | Vitesse de vent constante en m/s pour la convection forcée. |
| `--vent-api` | Utilise une série journalière NASA POWER, avec cache et repli à 2,5 m/s. |

## Structure du code

| Élément | Rôle |
| --- | --- |
| `modele_courbe.py` | Point d'entrée principal. Prépare et intègre une simulation ponctuelle avec Backward Euler. Expose aussi `run_point_simulation()`. |
| `fonctions.py` | Prépare les séries d'entrée : capacité thermique, albédo du sol et des nuages, flux latent. |
| `bibliotheque.py` | Constantes et fonctions historiques réexportées pour préserver la compatibilité des anciens scripts. |
| `chemins.py` | Définit tous les chemins vers les ressources et le cache à partir de l'emplacement du fichier. |
| `physique/` | Modules spécialisés pour les grandeurs physiques. |
| `visualisation/` | Lecteurs et interfaces pour les grilles de température précalculées. |

### Modules physiques

| Module | Fonction |
| --- | --- |
| `solaire.py` | Géométrie solaire, déclinaison et flux solaire net absorbé. |
| `albedo.py` | Albédo de surface mensuel, albédo des nuages CERES et repli NASA POWER mis en cache. |
| `capacite_surface.py` | Capacité thermique du sol à partir de l'humidité RZSM ; capacité sèche de repli si la donnée manque. |
| `chaleur_latente.py` | Flux latent moyen par continent ou océan. |
| `convection.py` | Convection forcée (vent) et naturelle ; récupération facultative du vent NASA POWER. |
| `diffusion.py` | Diffusion thermique radiale dans le sol, conservée mais non branchée au moteur principal. |
| `co2.py` | Prototype séparé de transfert radiatif CO₂ ; il demande notamment `radis` et n'est pas intégré au moteur. |

## Données attendues

Les ressources sont cherchées dans `modele0_maintenance/ressources/` :

| Ressource | Utilisation | Comportement si absente |
| --- | --- | --- |
| `albedo/albedo01.csv` à `albedo12.csv` | Albédo de surface mensuel. | Requis pour une simulation ponctuelle. |
| `albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | Albédo des nuages. | Requis si `xarray` est disponible ; sans `xarray`, l'albédo nuageux vaut zéro. |
| `capacite_humidite/average_rzsm_tout.csv` | Capacité thermique du sol. | Capacité sèche constante de repli. |
| `carte/ne_110m_admin_0_countries.shp` | Attribution continent/océan. | Le point est traité comme océan. |
| `cotes/ne_10m_coastline.shp` | Contours des continents dans les visualisations. | Les visualisations restent disponibles sans contours. |
| `grilles/*.npy` | Cartes de températures précalculées. | Nécessaires aux planisphères et sphères. |
| `12_mois/*.csv` | Températures horaires mensuelles. | Nécessaires au viewer 3D rapide. |

Les données et grilles peuvent être inventoriées ou régénérées avec
`modele0_maintenance/outils_generation_donnees/generer_donnees.py`. Voir le
[README de génération](../outils_generation_donnees/README.md).

## Visualisations

Ces commandes se lancent depuis la racine du dépôt. Les options de grille sont
`auto`, `rapide`, `1an` et `stabilisee`.

```bash
# Planisphères interactifs
python modele0_maintenance/codes_python/visualisation/modele_planisphere_basse_res.py --grille rapide
python modele0_maintenance/codes_python/visualisation/modele_planisphere_haute_res.py --grille rapide

# Globes 3D interactifs
python modele0_maintenance/codes_python/visualisation/modele_sphere_basse_res.py --grille rapide
python modele0_maintenance/codes_python/visualisation/modele_sphere_haute_res.py --grille rapide

# Globe 3D à partir des CSV mensuels
python modele0_maintenance/codes_python/visualisation/affichage_3D_rapide.py --month janvier --hour 12

# Interface : clic sur la carte puis courbe de température du point
python modele0_maintenance/codes_python/visualisation/interface_carte_courbe.py
```

Les planisphères et les globes utilisent des curseurs pour changer le jour et
l'heure. Le choix `auto` privilégie une grille annuelle, puis stabilisée, puis
rapide, en choisissant d'abord la résolution demandée.

## Bilan physique et limites

Le bilan de surface intégré par le moteur est, de façon simplifiée :

```text
C · dT/dt = flux solaire absorbé − flux latent − flux convectif
          + rayonnement atmosphérique − rayonnement thermique de surface
```

La résolution temporelle par défaut est de 1 800 s. L'intégration est implicite
(Backward Euler) et résolue par quelques itérations de Newton à chaque pas.

Ce n'est pas un modèle climatique complet : la diffusion du sol et le prototype
CO₂ ne sont pas couplés à la simulation principale, et les échanges
atmosphériques sont volontairement simplifiés. Pour le contexte théorique et
la provenance des briques conservées, consulter
[`../THEORIE.md`](../THEORIE.md) et [`../PROVENANCE.md`](../PROVENANCE.md).
