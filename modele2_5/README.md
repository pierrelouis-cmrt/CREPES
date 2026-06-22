# Modèle 2.5 — colonne CO₂ à dix couches

Le modèle 2.5 est une itération du modèle 2 : il utilise dix couches définies
en pression, le profil standard de température 1976, des sous-bandes CO₂ et un
facteur diffusif. Il reste un calcul radiatif de colonne à températures
imposées.

## Lancer

Depuis la racine du dépôt :

```bash
python -m pip install -r modele2_5/requirements.txt
python modele2_5/codes_python/modele2_5.py
python modele2_5/ressources/tester_modele2_5.py
```

Pour régénérer le profil atmosphérique :

```bash
python modele2_5/ressources/profil_vertical_atmosphere_co2.py --no-plot
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Noyau de colonne radiative à dix couches. |
| `ressources/` | Profil vertical, calibration et tests numériques. |
| `requirements.txt` | Dépendances Python du modèle. |
| `README.md` | Présentation et procédures de lancement. |

Les consignes propres à chaque sous-dossier sont dans leurs README respectifs.
