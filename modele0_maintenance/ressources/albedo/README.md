# Albédo

Ce dossier contient l'albédo du sol par mois et le jeu CERES utilisé pour la contribution nuageuse.

## Albédo de surface

`albedo01.csv` à `albedo12.csv` représentent janvier à décembre. La première colonne, `Latitude/Longitude`, contient les latitudes ; la première ligne contient les longitudes ; les cellules contiennent un albédo compris entre 0 et 1. Les douze fichiers doivent partager les mêmes axes.

## Albédo nuageux

`CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` est une source externe locale. Le moteur y lit `toa_sw_all_mon`, `toa_sw_clr_c_mon` et `solar_mon`.

Les CSV de surface peuvent être régénérés via NASA POWER : consulter le [README du générateur](../../outils_generation_donnees/albedo/README.md). Le fichier CERES n'est pas généré par le projet.
