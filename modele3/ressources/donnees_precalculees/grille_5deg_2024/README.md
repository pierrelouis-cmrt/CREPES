# Donnees precalculees 3.1

Paquet compact genere depuis les ressources racine du depot. Ce dossier
est la source normale du calcul 3.1 et l'entree grille du modele 4.

- Fichier: `donnees_colonnes_5deg_2024.npz`
- Resolution: 5 degres
- Annee: 2024
- Grille: 36 latitudes x 72 longitudes x 12 mois
- Usage normal: `modele3_1.donnees.charger_paquet_grille`.

## Provenance

| Champ | Source active | Transformation |
| --- | --- | --- |
| Profils `T`, `q` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3.1. |
| Surface | ERA5 single levels, `ressources/**/*.nc` | Selection au plus proche sur grille 5 degres. |
| Flux de validation | ERA5 flux moyens | Stockes pour comparaison, jamais pour recalibrer. |
| Transmissivite SW | Geometrie solaire 3.1 + ERA5 `avg_sdswrf` | `ERA5 SW_down / moyenne_mensuelle(S0*cos(i))`, borne `[0, 1]`. |
| Albedo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Selection mensuelle au plus proche. |

Les fichiers `ressources/albedo/*` sont des copies racine des donnees utiles
historiquement presentes dans le modele 0. Le code 3.1 ne lit pas le dossier
`modele0_maintenance/`.

## Contenu

Le `.npz` contient seulement les champs necessaires au calcul normal :
coordonnees, poids de surface, pression de surface, albedo, transmissivite
court-onde mensuelle, champs surface utiles, flux ERA5 de validation et
couches pretraitees. Les facteurs de quantification, unites et sources
sont dans `metadata.json`.
