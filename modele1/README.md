# Modèle 1

Colonne radiative CO₂ simplifiée à trois couches. Le script calcule le flux infrarouge sortant au sommet de l’atmosphère et le flux infrarouge descendant reçu par la surface.

## Lancer le modèle

Depuis la racine du dépôt :

```bash
python modele1/modele1.py
```

La sortie affiche :

```text
flux_infrarouge_sortant_sommet_atmosphere_W_m2 = ...
flux_infrarouge_descendant_surface_W_m2 = ...
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `modele1.py` | Code exécutable du modèle. |
| `modele1-squelette.py` | Version de travail ou support pédagogique. |
| `THEORIE.md` | Hypothèses, équations, paramètres, sources et limites. |
