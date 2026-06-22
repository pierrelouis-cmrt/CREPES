# Modèle 1 — colonne radiative simplifiée

Le modèle 1 est une colonne atmosphérique moyenne à trois couches. Les
températures sont imposées : le script calcule les flux infrarouges ascendant
au sommet de l'atmosphère et descendant vers la surface, avec deux bandes CO₂.
Il s'agit d'un premier noyau pédagogique, sans latitude ni évolution temporelle.

## Lancer le modèle

Depuis la racine du dépôt :

```bash
python -m pip install -r modele1/requirements.txt
python modele1/codes_python/modele1.py
```

Le script affiche notamment `flux_infrarouge_sortant_sommet_atmosphere_W_m2`
et `flux_infrarouge_descendant_surface_W_m2`.

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Script du modèle et squelette pédagogique. |
| `requirements.txt` | Dépendances du modèle (bibliothèque standard uniquement). |
| `README.md` | Présentation et commande de lancement. |

Les scripts du dossier `codes_python/` sont décrits dans son
[README](codes_python/README.md).
