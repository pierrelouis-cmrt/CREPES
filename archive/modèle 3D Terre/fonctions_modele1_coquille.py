import numpy as np
import shapefile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys
import subprocess


try:
    import sys
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", sys])
    import sys

try:
    import numpy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", numpy])
    import numpy


try:
    import shapefile
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pyshp])
    import shapefile

try:
    import matplotlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", matplotlib])
    import matplotlib



Tmax = 388
sigma = 5.67e-8        # Constante de Stefan-Boltzmann (W·m⁻²·K⁻⁴)
R = 6.371e6            # Rayon terrestre (m)
C = 4.31e20            # Capacité thermique de la Terre (J/K)



def project_to_sphere(lon, lat, radius=1):
    """
    Prend en entrée la longitude, la latitude et le rayon (optionnel, 1 par défaut)
    Fonction qui convertit des coordonnées géographiques (longitude et latitude) en coordonnées cartésiennes
    Sort les valeurs x, y, z de la position en coordonnées cartésiennes

    """
    lon = np.radians(lon)
    lat = np.radians(lat)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def get_shape(shape):
    """
    Extrait et projette les points d'une forme géographique sur une sphère.

    Paramètres:
    shape: Objet de forme géographique contenant des points (lon, lat).

    Retours:
    tuple: Coordonnées (x, y, z) projetées sur une sphère, ou None si les points sont insuffisants.
    """
    points = np.array(shape.points)
    points = points[::300]
    lon = points[:, 0]
    lat = points[:, 1]
    if len(lon) < 2 or len(lat) < 2:
        return None
    x_coast, y_coast, z_coast = project_to_sphere(lon, lat, 6371 * 1000 + 100000)
    return x_coast, y_coast, z_coast





def calc_power_temp(time, mois, sun_vector, x, y, z, phi, theta, constante_solaire, sigma, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Calcule la température globale à un instant donné en supposant un refroidissement radiatif
    selon la loi de Stefan-Boltzmann.
    """
    Ti = 288
    t_sec = time * 3600

    if t_sec == 0:
        Tglobale = Ti
    else:
        dT_dt = -(4 * np.pi * R**2 * sigma * Ti**4) / C
        Tglobale = Ti + dT_dt * t_sec

    temperature = Tglobale * np.ones_like(x) 

    puissance_recue = np.zeros_like(x) 
    return puissance_recue, temperature



def update_plot(time, mois, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes, Tmax):
    """
    Fonction prend en entrée l'heure de la journée et le mois (par défaut, Mars : sera modifié quand on clique sur les boutons à gauche de la modélisation), l'axe, la figure, shapes, les coordonnées (x,y,z), les constantes :sigma, phi, theta, rayon_astre_m, la liste d'albedo, la latitude et la longitude 
    Elle calcule la puissance emise par la terre avec la fonction calc_power_temp puis effet_de_serre. Puis elle met à jour le modèle : les lignes de côte sont tracées, puis la surface de la sphère est représentée en utilisant les valeurs de puissance calculées, avec des couleurs déterminées par une colormap (turbo).
    """
    sun_vector = np.array([1, 0, 0])
    _, temperature = calc_power_temp(time, mois, sun_vector, x, y, z, phi, theta, constante_solaire, sigma, rayon_astre_m, list_albedo, latitudes, longitudes)
    ax.clear()

    for shape in shapes:
        result = get_shape(shape)
        if result is not None:  # Vérifiez si get_shape a retourné des coordonnées valides
            x_coast, y_coast, z_coast = result
            ax.plot(x_coast, y_coast, z_coast, color='black', zorder=5)


    norm = Normalize(vmin=253, vmax=323)
    cmap = plt.cm.turbo
    facecolors = cmap(norm(temperature))
    surf = ax.plot_surface(x, y, z, facecolors=facecolors, rstride=1, cstride=1, linewidth=1, shade=False)

    if not hasattr(fig, "colorbar_added"):
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])

        
        cbar_ax = fig.add_axes([0.9, 0.25, 0.03, 0.5])  
        fig.colorbar(mappable, cax=cbar_ax, label="Température (K)")
        fig.colorbar_added = True  


    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Distribution de la puissance radiative reçue par l\'astre à t = {time:.1f} h (mois : {mois})')
    fig.canvas.draw_idle()




def slider_update(val, current_month, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Prend en entrée l'heure de la journée, le mois, l'axe, la figure, shapes, les coordonnées (x,y,z), les constantes :sigma, phi, theta, rayon_astre_m, la liste d'albedo, la latitude et la longitude  
    Fonction qui update le modèle lorsque l'on fait varier la valeur de temps.
    """
    update_plot(val, current_month[0], ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes, Tmax)

def set_mois(mois, current_month, time_slider, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Prend en entrée le mois sélectionné dans la sidebar : Janvier, Février, Mars, etc.
    Fonction qui met à jour le modèle lorsque l'on clique sur le bouton mois (boutons radio)
    """
    current_month[0] = mois
    slider_update(time_slider.val, current_month, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes)


