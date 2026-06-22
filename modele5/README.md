# Modèle 5 — grille avec échanges radiatifs horizontaux

Le modèle 5 prolonge le moteur rapide du modèle 4. Il intègre la température de
surface sur la grille globale et ajoute un terme d'échange infrarouge entre les
colonnes atmosphériques voisines. Cet échange est conservatif à l'échelle de la
grille ; il ne remplace ni l'advection atmosphérique ni un océan dynamique.

## Lancer

Depuis la racine du dépôt :

```bash


# Simulation par défaut : un jour, grille globale, sortie toutes les 4 h
python -m modele5.codes_python.modele5

# Petite grille de développement
python -m modele5.codes_python.modele5 --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele5/sorties/simulation_dev.npz

# Comparaison sans échange horizontal
python -m modele5.codes_python.modele5 --max-latitudes 4 --max-longitudes 8 --facteur-horizontal 0 --output modele5/sorties/simulation_sans_horizontal.npz

# Tests
python modele5/tests/tester_modele5.py
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Moteur couplé et planisphère des résultats. |
| `sorties/` | Résultats `.npz` des simulations. |
| `tests/` | Vérifications de conservation et de simulation. |

Les sous-dossiers détaillent leur propre contenu et leurs conventions.
