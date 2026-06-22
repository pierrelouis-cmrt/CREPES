# Donnees precalculees 3

Paquet compact genere depuis les ressources racine du depot. Ce dossier
est la source normale du calcul 3 et l'entree grille du modele 4.

- Fichier: `donnees_colonnes_5deg_2024.npz`
- Resolution: 5 degres
- Annee: 2024
- Grille: 36 latitudes x 72 longitudes x 12 mois
- Usage normal: `modele3.codes_python.donnees.charger_paquet_grille`.

## Provenance

| Champ | Source active | Transformation |
| --- | --- | --- |
| Profils `T`, `q` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3. |
| Surface | ERA5 single levels, `ressources/**/*.nc` | Selection au plus proche sur grille 5 degres. |
| Flux de validation | ERA5 flux moyens | Stockes pour comparaison, jamais pour recalibrer. |
| Transmissivite SW | Geometrie solaire 3 + ERA5 `avg_sdswrf` | `ERA5 SW_down / moyenne_mensuelle(S0*cos(i))`, borne `[0, 1]`. |
| Albedo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Longitudes normalisees -180..180, selection mensuelle au plus proche, puis correction des zeros sur neige/glace. |
| Nuages | ERA5 `cc/lcc/mcc/hcc/tcc` + CERES EBAF TOA | Fractions nuageuses bornees `[0, 1]`; albedo nuageux CERES stocke comme diagnostic. |

Les fichiers `ressources/albedo/*` sont des copies racine des donnees utiles
historiquement presentes dans le modele 0. Le code 3 ne lit pas le dossier
`modele0_maintenance/`. Les valeurs d'albedo nulles sur des mailles
neige/glace viennent surtout de mois polaires ou le rapport source
`SW_UP / SW_DOWN` n'est pas observable ; elles sont remplacees par un
melange simple entre `0.30` et `0.65` selon la fraction neige/glace.

Les couches verticales dont l'epaisseur serait inferieure a 0.1 hPa sont
ignorees avant stockage pour eviter des couches nulles apres quantification.

## Contenu

Le `.npz` contient seulement les champs necessaires au calcul normal :
coordonnees, poids de surface, pression de surface, albedo, transmissivite
court-onde mensuelle, champs surface utiles, flux ERA5 de validation et
couches pretraitees, dont la fraction nuageuse par couche. Les facteurs de
quantification, unites et sources sont stockes dans le meme `.npz`.
