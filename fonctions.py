import numpy as np
import shapefile
import matplotlib.pyplot as plt

def update_sun_vector(mois, sun_vector):
    """
    Met à jour le vecteur solaire en fonction du mois pour tenir compte de l'inclinaison saisonnière de l'axe de la Terre.
    
    Paramètres:
    mois (int): Le mois de l'année (1 pour janvier, 12 pour décembre).
    
    Retours:
    numpy.ndarray: Le vecteur solaire mis à jour après application de la rotation saisonnière.

    La matrice de rotation saisonnière est utilisée pour faire pivoter le vecteur solaire autour de l'axe y en fonction de la saison .
    """
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    sun_vector_rotated = np.dot(rotation_matrix_saison, sun_vector)
    return sun_vector_rotated

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

def get_albedo(lat, lon, mois, list_albedo, latitudes, longitudes):
    """
    Prend en entrée la latitude, la longitude et le mois
    Fonction qui va chercher l'albedo de ce point pour ce mois particulier dans la listes créée précédemment list_albedo
    Sort la valeur de l'albedo
    """
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return list_albedo[mois-1][lat_idx, lon_idx]

def calc_power_temp(time, mois, sun_vector, x, y, z, phi, theta, constante_solaire, sigma, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Calcule la puissance solaire reçue et la température en fonction de l'heure et du mois.

    Paramètres:
    time (float): Heure de la journée (0-24).
    mois (int): Mois de l'année (1-12).
    sun_vector (numpy.ndarray): Vecteur solaire initial.
    x, y, z (numpy.ndarray): Coordonnées de la grille sphérique.
    phi, theta (numpy.ndarray): Coordonnées angulaires de la grille sphérique.
    constante_solaire (float): Constante solaire (W/m^2).
    sigma (float): Constante de Stefan-Boltzmann (W/m^2/K^4).
    rayon_astre_m (float): Rayon de l'astre en mètres.
    list_albedo (list): Grilles d'albédo pour chaque mois.
    latitudes, longitudes (numpy.ndarray): Latitudes et longitudes des données d'albédo.

    Retours:
    tuple: Puissance reçue (numpy.ndarray) et température (numpy.ndarray).

    La matrice de rotation fait pivoter le vecteur solaire autour de l'axe z. L'angle d'incidence est calculé,
    puis l'albédo est mappé sur la grille pour ajuster la puissance reçue. La température est déterminée
    par la loi de Stefan-Boltzmann.
    """
    angle_rotation = (time / 24) * 2 * np.pi  # Conversion du temps en angle
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation), np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])

    sun_vector_rotated = np.dot(rotation_matrix, update_sun_vector(mois, sun_vector))

    normal = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T

    albedo_grid_mapped = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            lon, lat = np.degrees(phi[i, j]), 90 - np.degrees(theta[i, j])
            if lon > 180:
                lon -= 360
            albedo_grid_mapped[i, j] = get_albedo(lat, lon, mois, list_albedo, latitudes, longitudes)

    coef_reflexion = albedo_grid_mapped
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)

    temperature = (puissance_recue / sigma) ** 0.25

    return puissance_recue, temperature

def update_plot(time, mois, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Fonction prend en entrée l'heure de la journée et le mois (par défaut, Mars : sera modifié quand on clique sur les boutons à gauche de la modélisation), l'axe, la figure, shapes, les coordonnées (x,y,z), les constantes :sigma, phi, theta, rayon_astre_m, la liste d'albedo, la latitude et la longitude 
    Elle calcule la puissance emise par la terre avec la fonction calc_power_temp puis effet_de_serre. Puis elle met à jour le modèle : les lignes de côte sont tracées, puis la surface de la sphère est représentée en utilisant les valeurs de puissance calculées, avec des couleurs déterminées par une colormap (viridis).
    """
    sun_vector = np.array([1, 0, 0])
    puissance_recue, _ = calc_power_temp(time, mois, sun_vector, x, y, z, phi, theta, constante_solaire, sigma, rayon_astre_m, list_albedo, latitudes, longitudes)
    ax.clear()

    for shape in shapes:
        result = get_shape(shape)
        if result is not None:  # Vérifiez si get_shape a retourné des coordonnées valides
            x_coast, y_coast, z_coast = result
            ax.plot(x_coast, y_coast, z_coast, color='black', zorder=5)

    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(puissance_recue / np.max(puissance_recue)), rstride=1, cstride=1, linewidth=1)

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
    update_plot(val, current_month[0], ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes)

def set_mois(mois, current_month, time_slider, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes):
    """
    Prend en entrée le mois sélectionné dans la sidebar : Janvier, Février, Mars, etc.
    Fonction qui met à jour le modèle lorsque l'on clique sur le bouton mois (boutons radio)
    """
    current_month[0] = mois
    slider_update(time_slider.val, current_month, ax, fig, shapes, x, y, z, constante_solaire, sigma, phi, theta, rayon_astre_m, list_albedo, latitudes, longitudes)