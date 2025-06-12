
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import shapefile
import pandas as pd
from fonctions_modele1_coquille import project_to_sphere, get_shape, calc_power_temp, update_plot, slider_update, set_mois


# Charger les données SHP
sf = shapefile.Reader("ressources/data/ne_10m_coastline.shp")
shapes = sf.shapes()
Tmax = 388

list_albedo = []
for i in range(1, 10):
    # Charger les données d'albédo
    csv_file_path = f'ressources/albedo/albedo0{i}.csv'
    albedo_data = pd.read_csv(csv_file_path)
    if i == 1:
        latitudes = albedo_data['Latitude/Longitude'].values
        longitudes = albedo_data.columns[1:].astype(float)
    # Convertir les données CSV en grille d'albédo
    albedo_grid = albedo_data.set_index('Latitude/Longitude').to_numpy()
    list_albedo.append(albedo_grid)
for i in range(10, 13):
    csv_file_path = f'ressources/albedo/albedo{i}.csv'
    albedo_data = pd.read_csv(csv_file_path)
    # Convertir les données CSV en grille d'albédo
    albedo_grid = albedo_data.set_index('Latitude/Longitude').to_numpy()
    list_albedo.append(albedo_grid)


# Constantes
constante_solaire = 1361  # W/m^2, valeur moyenne au niveau de la Terre
rayon_astre = 6371  # km, par exemple le rayon de la Terre
sigma = 5.670e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4
epaisseur_atmosphere = 600  # km, approximative thickness of Earth's atmosphere

rayon_astre_m = rayon_astre * 1000
epaisseur_atmosphere_m = epaisseur_atmosphere * 1000

C_CO2_moy = 400 #ppm
C_H2O_moy = 25000 #ppm


# Grille sphérique pour représenter la surface de l'astre
phi = np.linspace(0, 2 * np.pi, 60)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)

x = rayon_astre_m * np.sin(theta) * np.cos(phi)
y = rayon_astre_m * np.sin(theta) * np.sin(phi)
z = rayon_astre_m * np.cos(theta)

x_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.cos(phi)
y_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.sin(phi)
z_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.cos(theta)

# Création de la figure et de l'axe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialisation du graphique
current_month = [3]
update_plot(0, current_month[0], ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes, Tmax)

# Création du slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=1)

# Liaison du slider à la fonction de mise à jour
time_slider.on_changed(lambda val: slider_update(val, current_month, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes))

# Création des axes et boutons pour chaque mois
mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
btn_mois = []

for i, mois in enumerate(mois_labels):
    ax_mois = plt.axes([0.1, 0.95 - i * 0.07, 0.1, 0.04])
    btn = Button(ax_mois, mois)
    btn.on_clicked(lambda event, m=i + 1: set_mois(m, current_month, time_slider, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes))
    btn_mois.append(btn)

# Affichage de la figure
plt.show()