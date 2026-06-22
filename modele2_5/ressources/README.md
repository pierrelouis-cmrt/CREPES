# Ressources — modèle 2.5

Ce dossier rassemble les outils auxiliaires du modèle 2.5 : génération du
profil atmosphérique, calibration des coefficients optiques et tests du noyau
radiatif. Les données détaillées générées restent dans `données/`.

## Lancer les outils

Depuis la racine du dépôt :

```bash
# Générer le profil pression-température-CO₂
python modele2_5/ressources/profil_vertical_atmosphere_co2.py --no-plot

# Vérifier les cas limites et la réponse au CO₂
python modele2_5/ressources/tester_modele2_5.py

# Recalculer le facteur d'échelle optique
python modele2_5/ressources/calibrer_coefficients_optiques.py
```

Les options `--csv` et `--output` du générateur de profil permettent de
déplacer les sorties.

## Structure

| Élément | Rôle |
| --- | --- |
| `profil_vertical_atmosphere_co2.py` | Génère le profil vertical standard. |
| `calibrer_coefficients_optiques.py` | Calibre les coefficients optiques effectifs. |
| `tester_modele2_5.py` | Lance les vérifications numériques du modèle. |
| `données/` | Fichiers produits par le générateur ; contenu détaillé exclu de cette documentation. |

Le moteur principal est décrit dans le [README parent](../README.md).
