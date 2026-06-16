# Donnees precalculees 3.1

Paquet compact genere depuis les ressources racine du depot. Ce dossier
est la source normale du calcul 3.1 et la future entree grille du modele 4.

- Fichier: `donnees_colonnes_5deg_2024.npz`
- Resolution: 5 degres
- Annee: 2024
- Grille: 36 latitudes x 72 longitudes x 12 mois
- Usage normal: `modele3_1.donnees.charger_paquet_grille`.

## Provenance

| Champ | Source active | Transformation |
| --- | --- | --- |
| Profils `T`, `q`, `cc` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3.1. |
| Surface et nuages | ERA5 single levels, `ressources/**/*.nc` | Selection au plus proche sur grille 5 degres. |
| Flux de validation | ERA5 flux moyens | Stockes pour comparaison, jamais pour recalibrer. |
| Albedo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Selection mensuelle au plus proche. |
| Albedo nuages | `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | `(toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon`. |

Les fichiers `ressources/albedo/*` sont des copies racine des donnees utiles
historiquement presentes dans le modele 0. Le code 3.1 ne lit pas le dossier
`modele0_maintenance/`.

## Contenu

Le `.npz` contient seulement les champs necessaires au calcul normal :
coordonnees, poids de surface, pression de surface, albedos, diagnostics
surface, flux ERA5 de validation et couches pretraitees. Les facteurs de
quantification, unites et sources sont dans `metadata.json`.
