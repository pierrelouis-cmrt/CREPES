# Provenance

Cette fiche indique ce qui provient de chaque groupe et ce qui a été écrit pour la
compatibilité du projet.

## Synthèse

| Élément | Statut |
| --- | --- |
| Données et PDF | Copies exactes des archives sources. |
| Moteur par défaut | Carcajous, modèle 4. |
| Convection | Active par défaut, depuis Chevreaux et Ornithorynquietant. |
| Diffusion | Conservée, non activée. |
| Gaz à effet de serre | Non intégré pour l'instant. |
| Code de compatibilité | Chemins, CLI, wrappers, caches, conversions d'unités. |
| Génération de grilles | `outils_generation_donnees/generer_donnees.py`, cohérent avec le moteur courant. |

## Par groupe

### Carcajous Callipyges

Utilisé comme socle principal :

- `codes_python/modele_courbe.py` : intégration Backward Euler.
- `outils_generation_donnees/generer_donnees.py` : génération globale des grilles avec le moteur courant.
- `codes_python/physique/solaire.py` : géométrie solaire.
- `codes_python/physique/albedo.py` : albédo de surface et nuages CERES.
- `codes_python/physique/capacite_surface.py` : capacité depuis RZSM.
- `codes_python/physique/chaleur_latente.py` : chaleur latente par continent.
- `ressources/grilles/*.npy`, `ressources/albedo/`, `ressources/capacite_humidite/`.

### Chevreaux brillants

Conservé et activé par défaut dans le moteur principal :

- `codes_python/physique/convection.py` : convection forcée par vent.
- `documents_sources/Chevreaux_*.pdf` : théorie et modèle 6.

### Bernard l'hermite

Conservé comme interface et inspiration :

- `codes_python/visualisation/interface_carte_courbe.py` : clic carte vers courbe.
- `documents_sources/Bernard_Synthese.pdf`.

La GUI finale est simplifiée et utilise le moteur principal, pas le moteur
Bernard d'origine.

### Ornithorynquietant

Conservé et activé par défaut pour la convection naturelle :

- `codes_python/physique/convection.py` : convection naturelle.
- `codes_python/physique/diffusion.py` : diffusion radiale conservée, non branchée.
- `ressources/12_mois/*.csv` : format mensuel utilisé par le viewer 3D rapide,
désormais régénérable depuis la grille annuelle du moteur courant.

## Données et génération

Le détail opérationnel se trouve dans `outils_generation_donnees/README.md`.

| Donnée finale | Statut de génération |
| --- | --- |
| `grilles/grid_*_fast.npy` | Générable proprement par `outils_generation_donnees/generer_donnees.py`. |
| `grilles/grid_*_1yr.npy` | Générable proprement par `outils_generation_donnees/generer_donnees.py`; calcul long. |
| `grilles/grid_*_stabilized.npy` | Générable proprement par `outils_generation_donnees/generer_donnees.py`; calcul très long. |
| `albedo/*.csv` | Générable par `outils_generation_donnees/generer_donnees.py --run albedo-surface-nasa`; appelle NASA POWER. |
| `12_mois/*.csv` | Générable par `outils_generation_donnees/generer_donnees.py --run temperatures-12mois` depuis `grilles/grid_lowres_1yr.npy`. |
| `capacite_humidite/average_rzsm_tout.csv` | Source externe locale conservée comme entrée du moteur. |
| `CERES_EBAF-TOA_*.nc`, shapefiles, PDF sources | Sources externes locales à remplacer manuellement si besoin. |

## Choix d'activation

Activé par défaut :

- bilan radiatif Carcajous ;
- albédo sol et nuages ;
- chaleur latente ;
- capacité RZSM avec fallback sec constant si la source manque ;
- convection forcée Chevreaux ;
- convection naturelle Ornithorynquietant.

Désactivation ou sélection explicite :

- couper les deux convections : `--sans-convection` ou `--convection aucune` ;
- ne garder que la convection forcée : `--convection forcee` ;
- ne garder que la convection naturelle : `--convection naturelle`.

Non activé :

- diffusion, car le flux de surface est ambigu dans le script source ;
- gaz à effet de serre, volontairement non intégré pour reprise ultérieure.
