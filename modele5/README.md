# Modèle 5 — grille de température avec échanges horizontaux

Le modèle 5 est la version couplée du modèle 4. Il calcule la température de
surface globale `T_surface(t, lat, lon)` avec le même bilan de surface que le
modèle 4, puis ajoute les échanges radiatifs horizontaux entre les couches
atmosphériques des colonnes voisines.

Il remplace donc le diagnostic du dossier `modele5 (temporaire)` : cette version
ne se contente plus de donner ce qu'une couche émet latéralement. Elle échange
effectivement cette émission entre les cellules de la grille et en répercute la
part transmise vers la surface dans le bilan de température.

## Bilan de surface

La base est strictement celle du moteur rapide du modèle 4 :

```text
C_surface dT_surface/dt =
    SW_absorbé
  + LW_descendant_absorbé
  - LW_émis_surface
  - Q_latent
  - Q_convection
  + Q_horizontal
```

`Q_horizontal` est le nouveau terme du modèle 5, en `W m⁻²` de surface au sol.
Le code garde l'intégration semi-implicite du modèle 4 pour le rayonnement
sortant et la convection ; le transport horizontal est évalué explicitement à
chaque pas de temps.

## Échange entre deux colonnes

Pour chaque couche `k`, bande infrarouge `b` et face commune entre deux cellules
`A` et `B`, le modèle reprend l'émission du prototype :

```text
E(k, b) = ε(k, b) × B_b(T(k))
```

où `ε` vient des mêmes opacités CO2 + H2O que le modèle 3 et `B_b` est le flux
de Planck intégré dans la bande. L'échange net sur une interface est :

```text
P(A ← B) = A_face × [E_B(k, b) - E_A(k, b)]
```

Chaque interface n'est calculée qu'une fois : le gain d'une colonne est la
perte de l'autre, en watts. Les surfaces des mailles et des faces tiennent
compte de la géométrie sphérique ; les longitudes sont périodiques sur la grille
globale. Sur une sous-grille de développement, les bords sont fermés afin de ne
pas relier artificiellement les deux extrémités du sous-domaine.

La convergence de chaque couche est ensuite pondérée par la transmission des
couches situées en dessous. Cette part atteint le sol et devient
`Q_horizontal`.

Les profils ERA5 du modèle 3 sont mensuels et servent de référence pour les
émissions latérales des couches.

## Lancer une simulation

Depuis la racine du dépôt :

```bash
.\.venv\Scripts\python.exe -m modele5.modele5
```

Par défaut : grille globale 5°, un jour simulé, pas de 1800 s et sortie toutes
les quatre heures dans `modele5/sorties/simulation_modele5.npz`.

Pour une petite grille de développement :

```bash
.\.venv\Scripts\python.exe -m modele5.modele5 --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele5/sorties/simulation_dev.npz
```

Pour isoler le bilan du modèle 4 sans l'échange horizontal :

```bash
.\.venv\Scripts\python.exe -m modele5.modele5 --max-latitudes 4 --max-longitudes 8 --facteur-horizontal 0 --output modele5/sorties/simulation_sans_horizontal.npz
```

## Options importantes

| Option | Défaut | Rôle |
| --- | ---: | --- |
| `--jours` | `1` | Durée simulée, en jours. |
| `--dt` | `1800` | Pas interne, en secondes. |
| `--sortie-heures` | `4` | Fréquence de sauvegarde. |
| `--facteur-horizontal` | `1` | Intensité de l'échange horizontal ; `0` le désactive. |
| `--max-latitudes`, `--max-longitudes` | — | Sous-grille de développement. |
| `--facteur-latent`, `--convection`, `--vent` | mêmes valeurs que modèle 4 | Termes de surface hérités du modèle 4. |

## Sortie NPZ

Le fichier produit conserve les champs principaux du modèle 4 :

- `temperature_surface_k[temps, lat, lon]` ;
- `temps_s`, `jours`, `heures`, `lat_deg`, `lon_deg` ;
- `capacite_surface_j_m2_k` ;
- flux moyens `sw_absorbe_surface`, `lw_down_absorbe_surface`, `lw_up_surface`,
  `flux_latent`, `flux_convection` et `flux_net_surface` ;
- `metadata_json`.

Deux diagnostics sont ajoutés :

- `flux_horizontal_net_surface_moyen_w_m2` : terme ajouté au bilan de surface ;
- `flux_horizontal_atmosphere_moyen_w_m2` : convergence brute de toutes les
  couches, avant transmission vers le sol. Sur la grille globale, son bilan en
  watts s'annule à l'arrondi numérique près.

## Tests

```bash
.\.venv\Scripts\python.exe modele5/tests/tester_modele5.py
```

Les tests vérifient la conservation de l'échange sur les interfaces, le cas
uniforme, une simulation courte et l'écriture du fichier NPZ.

## Planisphère du bilan total

Après une simulation, le script suivant affiche trois cartes : température
finale, évolution de température et flux horizontal moyen reçu par la surface.
Le titre donne aussi la puissance horizontale nette intégrée sur le globe.

```bash
.\.venv\Scripts\python.exe modele5/planisphere.py --fichier modele5/sorties/simulation_modele5.npz
```

Pour écrire directement un PNG sans ouvrir de fenêtre :

```bash
.\.venv\Scripts\python.exe modele5/planisphere.py --fichier modele5/sorties/simulation_modele5.npz --save modele5/sorties/planisphere_total.png --no-show
```

## Limites assumées

- Les profils de pression, humidité et température de référence sont mensuels
  et issus du paquet ERA5 du modèle 3.
- Les couches n'ont pas encore leur propre capacité thermique ni équation de
  mouvement.
- L'échange est radiatif infrarouge seulement ; il n'y a pas encore d'advection
  de masse, d'océan dynamique, de vent horizontal ni de diffusion turbulente.
- La transmission de la convergence atmosphérique vers le sol est une
  paramétrisation. Les résultats servent à étudier le couplage entre colonnes,
  pas à produire une prévision climatique validée.
