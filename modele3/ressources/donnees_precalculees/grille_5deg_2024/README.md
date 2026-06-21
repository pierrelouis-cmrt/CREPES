# Données pré-calculées 3

Paquet compact généré depuis les ressources racine du dépôt. Ce dossier
est la source normale du calcul 3 et l'entrée grille du modèle 4.

- Fichier: `donnees_colonnes_5deg_2024.npz`
- Résolution: 5 degrés
- Année: 2024
- Grille: 36 latitudes x 72 longitudes x 12 mois
- Usage normal: `modele3.donnees.charger_paquet_grille`.

## Provenance

| Champ | Source active | Transformation |
| --- | --- | --- |
| Profils `T`, `q` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3. |
| Surface | ERA5 single levels, `ressources/**/*.nc` | Sélection au plus proche sur grille 5 degrés. |
| Flux de validation | ERA5 flux moyens | Stockés pour comparaison, jamais pour recalibrer. |
| Transmissivité SW | Géométrie solaire 3 + ERA5 `avg_sdswrf` | `ERA5 SW_down / moyenne_mensuelle(S0*cos(i))`, borne `[0, 1]`. |
| Albédo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Longitudes normalisées -180..180, sélection mensuelle au plus proche. |

Les fichiers `ressources/albedo/*` sont des copies racine des données utiles
historiquement présentes dans le modèle 0. Le code 3 ne lit pas le dossier
`modele0_maintenance/`.

Les couches verticales nulles ne sont pas considérées comme normales. Dans ce
paquet, 4 couches nulles issues de la quantification initiale ont été marquées
manquantes; le chargeur expose aussi un diagnostic si une couche non positive
reste présente dans une source future.

## Contenu

Le `.npz` contient seulement les champs nécessaires au calcul normal :
coordonnées, poids de surface, pression de surface, albédo, transmissivité
shortwave mensuelle, champs surface utiles, flux ERA5 de validation et couches
prétraitées. Les facteurs de quantification, unités et sources sont dans
`metadata.json`.
