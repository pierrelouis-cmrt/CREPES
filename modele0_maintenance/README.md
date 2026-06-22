# Modèle 0 — maintenance

Le modèle 0 est la version historique maintenue du modèle de surface CREPES.
Il simule la température de surface en un point ou sur une grille à partir du
rayonnement solaire, de l'albédo, de la capacité thermique, du flux latent et
de la convection. Il est conservé comme référence pour les modèles ultérieurs.

## Démarrer

Depuis la racine du dépôt :

```bash
python -m pip install -r modele0_maintenance/requirements.txt
python modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Pour inventorier les ressources sans les modifier :

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Moteur ponctuel, briques physiques et visualisations. |
| `outils_generation_donnees/` | Génération et inventaire contrôlés des sorties. |
| `ressources/` | Ressources lues par le modèle et fichiers générés. |
| `documents_sources/` | PDF de référence historiques. |
| `PROVENANCE.md` | Traçabilité des briques conservées. |

Chaque sous-dossier explique son propre contenu dans son README.
