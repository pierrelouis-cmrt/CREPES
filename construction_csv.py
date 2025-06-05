###Fichier servant à créer la structure des fichiers CSV albedo.
#Un dossier albedo est créé
#12 fichier albedo01.csv ... albedo12.csv correspondent à janvier ... decembre
#Chaque fichier a sur la première ligne l'ensemble des latitudes et sur la première colonne l'ensemble des longitudes
#Le remplissage de ces fichier ne ce fait pas avec ce code (l'exécution serait trop longue, nous preferons séparer les taches)

import os
import numpy as np
import csv

# Constantes
rayon_astre = 6371  # km, par exemple le rayon de la Terre
rayon_astre_m = rayon_astre * 1000

# Grille sphérique pour représenter la surface de l'astre
phi = np.linspace(0, 2 * np.pi, 60)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)

x = rayon_astre_m * np.sin(theta) * np.cos(phi)
y = rayon_astre_m * np.sin(theta) * np.sin(phi)
z = rayon_astre_m * np.cos(theta)

# Fonction pour extraire les coordonnées en latitude et longitude
def extract_coordinates_long_lat(phi, theta, rayon_astre_m):
    coordinates = []

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            x_central = rayon_astre_m * np.sin(theta[i, j]) * np.cos(phi[i, j])
            y_central = rayon_astre_m * np.sin(theta[i, j]) * np.sin(phi[i, j])
            z_central = rayon_astre_m * np.cos(theta[i, j])

            latitude = np.arcsin(z_central / rayon_astre_m) * 180 / np.pi
            longitude = np.arctan2(y_central, x_central) * 180 / np.pi

            coordinates.append((latitude, longitude))

    return coordinates

# Fonction principale
def main():
    all_coordinates = extract_coordinates_long_lat(phi, theta, rayon_astre_m)

    # Créer le dossier albedo s'il n'existe pas
    if not os.path.exists("albedo"):
        os.makedirs("albedo")

    # Générer des fichiers CSV pour chaque mois sans les appels API
    for i in range(12):
        csv_filename = f"albedo/albedo{i+1:02d}.csv"
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Écrire l'en-tête des longitudes
            longitudes = sorted({coord[1] for coord in all_coordinates})
            writer.writerow(["Latitude/Longitude"] + longitudes)

            # Écrire les lignes pour chaque latitude avec des valeurs vides ou "N/A"
            latitudes = sorted({coord[0] for coord in all_coordinates})
            for latitude in latitudes:
                row = [latitude] + ["N/A"] * len(longitudes)
                writer.writerow(row)

        print(f"Fichier CSV pour le mois {i+1} généré : {csv_filename}")

if __name__ == "__main__":
    main()
