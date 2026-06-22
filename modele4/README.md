# Modèle 4 — grille de température de surface

Le modèle 4 fait évoluer la température de surface sur la grille mondiale de
5° du modèle 3. Il combine les flux radiatifs avec la capacité thermique, le
flux latent et la convection. Les cellules sont indépendantes : il n'y a pas
de transport horizontal ; cette extension est le rôle du modèle 5.

## Lancer

Depuis la racine du dépôt :

```bash
python -m pip install -r modele4/requirements.txt

# Diagnostic mensuel (moteur de référence)
python -m modele4.codes_python.modele4

# Simulation rapide vectorisée
python -m modele4.codes_python.rapide

# Tests
python modele4/tests/tester_modele4.py
python modele4/tests/tester_rapide.py
```

Pour une expérience courte, limiter la grille :

```bash
python -m modele4.codes_python.rapide --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele4/sorties/simulation_dev.npz
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Moteur de référence, moteur rapide, termes de surface et lanceur. |
| `documentation/` | Hypothèses, bilan de surface et notes de conception. |
| `tests/` | Tests du moteur de référence et du moteur rapide. |
| `sorties/` | Résultats `.npz` produits par les simulations. |
| `visualisation/` | Emplacement des outils de visualisation du modèle. |
| `requirements.txt` | Dépendances Python. |

Les sorties et les options propres aux moteurs sont documentées dans les README
des sous-dossiers concernés.
