# modele_sphere_hires.py
# ==============================================================================
# MODÈLE 0-D DE TEMPÉRATURE DE SURFACE - SPHÈRE 3D HAUTE RÉSOLUTION
#
# DESCRIPTION :
# Ce script exécute le modèle thermique sur une grille globale HAUTE RÉSOLUTION
# et visualise les résultats sur une sphère 3D interactive.
# Il combine la structure modulaire des scripts basse résolution avec une
# optimisation par pré-calcul des paramètres pour gérer la haute résolution.
#
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import pathlib
import os
import sys

# --- Import des modules locaux ---
import fonctions as f
import lib

# --- Dépendances optionnelles ---
try:
    import cartopy.feature as cfeature
    USE_CARTOPY = True
    print("Cartopy détecté. Les lignes de côte seront affichées en 3D.")
except ImportError:
    USE_CARTOPY = False
    print("AVERTISSEMENT: Cartopy non trouvé. Les côtes ne seront pas affichées.")

# --- Chemins des fichiers de résultats HAUTE RÉSOLUTION ---
NPY_DIR = pathlib.Path("ressources/npy")
HIRES_STABILIZED_FILE = NPY_DIR / "grid_hires_stabilized.npy"
HIRES_ONEYEAR_FILE = NPY_DIR / "grid_hires_1yr.npy"


# ────────────────────────────────────────────────
# CHARGEMENT ET SURÉCHANTILLONNAGE DES DONNÉES (UPSCALE)
# ────────────────────────────────────────────────
print("\n--- Étape 1: Chargement et Suréchantillonnage des Données ---")
try:
    # Chargement des données sources en basse résolution
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_lowres, LAT_lowres, LON_lowres = f.load_albedo_series(ALBEDO_DIR)
    RZSM_GRID_lowres, RZSM_LAT_lowres, RZSM_LON_lowres = f.load_and_grid_rzsm_data(
        pathlib.Path("ressources/Cp_humidity/average_rzsm_tout.csv")
    )
    ceres_clim_lowres = f.load_monthly_cloud_albedo_from_ceres(
        lat_deg=None, lon_deg=None, return_full_map=True
    )
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR CRITIQUE: Un fichier de ressources est introuvable : {e}")
    sys.exit(1)

# Définition de la grille HAUTE RÉSOLUTION
NLAT_HI, NLON_HI = 70, 140
LAT_HI = np.linspace(LAT_lowres.min(), LAT_lowres.max(), NLAT_HI)
LON_HI = np.linspace(LON_lowres.min(), LON_lowres.max(), NLON_HI)
print(f"Grille haute résolution définie : {NLAT_HI}x{NLON_HI} points.")

# --- Pré-calcul et interpolation des grilles de paramètres ---
print("Pré-calcul des paramètres sur la grille haute résolution...")

def upscale_grid(data_lowres, lat_low, lon_low, lat_hi, lon_hi):
    """Interpole une grille 2D de basse à haute résolution."""
    interp = RegularGridInterpolator(
        (lat_low, lon_low), data_lowres, bounds_error=False, fill_value=None
    )
    points_hi = np.array(np.meshgrid(lat_hi, lon_hi, indexing="ij"))
    return interp(np.moveaxis(points_hi, 0, -1))

monthly_albedo_hi = np.array([
    upscale_grid(monthly_albedo_lowres[m], LAT_lowres, LON_lowres, LAT_HI, LON_HI)
    for m in range(12)
])
RZSM_GRID_hi = upscale_grid(
    RZSM_GRID_lowres, RZSM_LAT_lowres[:-1], RZSM_LON_lowres[:-1], LAT_HI, LON_HI
)
Q_GRID_HI = np.array([[lib.P_em_surf_evap(lat, lon, verbose=False) for lon in LON_HI] for lat in LAT_HI])
C_GRID_HI = (f.compute_cp_from_rzsm(RZSM_GRID_hi) * 1000.0) * f.RHO_BULK * lib.EPAISSEUR_ACTIVE

# Utilisation de la fonction corrigée lisser_donnees_annuelles
albedo_sol_daily_grid = f.lisser_donnees_annuelles(monthly_albedo_hi, sigma=15.0)
monthly_cloud_albedo_hi = ceres_clim_lowres.sel(
    lat=LAT_HI, lon=LON_HI, method="nearest"
).to_numpy()
albedo_nuages_daily_grid = f.lisser_donnees_annuelles(monthly_cloud_albedo_hi, sigma=15.0)

print("--- Pré-calcul terminé ---")


# ────────────────────────────────────────────────
# FONCTIONS DE SIMULATION (Code inchangé)
# ────────────────────────────────────────────────
def f_rhs(T, phinet, C, q_latent):
    return (
        phinet
        - q_latent
        + lib.P_em_atm_thermal(lib.Tatm)
        - lib.P_em_surf_thermal(T)
    ) / C

def integrate_point_temperature(
    days, lat_rad, lon_deg, alb_sol_daily, alb_nuages_daily, C_const, q_base, T0
):
    from scipy.ndimage import gaussian_filter1d
    N = int(days * 24 * 3600 / lib.dt)
    T = np.empty(N + 1)
    T[0] = T0
    sign_daynight = np.empty(N)
    for k in range(N):
        t_sec = k * lib.dt
        jour_sim = int(t_sec // 86400) + 1
        _, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        sign_daynight[k] = 1.0 if f.cos_incidence(lat_rad, jour_sim, heure_solaire) > 0 else -1.0
    q_latent_smoothed = gaussian_filter1d(q_base * sign_daynight, sigma=3.0, mode="wrap")

    for k in range(N):
        day_of_year, heure_solaire = f.get_time_variables(k * lib.dt, lon_deg)
        jour_sim = int(k * lib.dt // 86400) + 1
        phi_n = lib.P_inc_solar(
            lat_rad, jour_sim, heure_solaire,
            alb_sol_daily[day_of_year], alb_nuages_daily[day_of_year]
        )
        X = T[k]
        for _ in range(8):
            F = X - T[k] - lib.dt * f_rhs(X, phi_n, C_const, q_latent_smoothed[k])
            dF = 1.0 - lib.dt * (-4.0 * lib.sigma * X**3 / C_const)
            if abs(dF) < 1e-9: break
            X -= F / dF
            if abs(F) < 1e-6: break
        T[k + 1] = X
    return T

def run_full_hires_simulation(days, stabilize=False):
    result_file = HIRES_STABILIZED_FILE if stabilize else HIRES_ONEYEAR_FILE
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    sim_type = "stabilisée (2 ans)" if stabilize else "rapide (1 an)"
    print(f"\nLancement de la simulation HAUTE RÉSOLUTION {sim_type}...")
    print(f"Les résultats seront sauvegardés dans '{result_file}'")

    N_steps = int(days * 24 * 3600 / lib.dt) + 1
    T_grid = np.zeros((N_steps, NLAT_HI, NLON_HI))

    for i in tqdm(range(NLAT_HI), desc="Progression (latitude)"):
        for j in range(NLON_HI):
            lat, lon = LAT_HI[i], LON_HI[j]
            T0 = 288.15 - 30 * np.sin(np.radians(lat)) ** 2
            T_series = integrate_point_temperature(
                days, np.radians(lat), lon,
                albedo_sol_daily_grid[:, i, j],
                albedo_nuages_daily_grid[:, i, j],
                C_GRID_HI[i, j],
                Q_GRID_HI[i, j],
                T0
            )
            T_grid[:, i, j] = T_series

    if stabilize:
        print("Simulation terminée. Extraction de la deuxième année...")
        steps_per_year = int(365 * 24 * 3600 / lib.dt)
        T_grid = T_grid[steps_per_year:, :, :]

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid


# ────────────────────────────────────────────────
# EXÉCUTION PRINCIPALE ET VISUALISATION 3D
# ────────────────────────────────────────────────
if __name__ == "__main__":
    target_file = None
    run_sim = False
    sim_days = 0
    stabilize = False

    if HIRES_STABILIZED_FILE.exists():
        print(f"Un fichier de simulation HI-RES stabilisée a été trouvé ('{HIRES_STABILIZED_FILE.name}').")
        choice = input("Voulez-vous le charger (c) ou relancer une simulation (r) ? [c/r]: ").lower()
        if choice == 'c': target_file = HIRES_STABILIZED_FILE
        else: run_sim = True
    elif HIRES_ONEYEAR_FILE.exists():
        print(f"Un fichier de simulation HI-RES sur 1 an a été trouvé ('{HIRES_ONEYEAR_FILE.name}').")
        choice = input("Charger ce fichier (c), ou lancer une nouvelle simulation stabilisée (r) ? [c/r]: ").lower()
        if choice == 'c': target_file = HIRES_ONEYEAR_FILE
        else: run_sim = True
    else:
        print("Aucun fichier de simulation HI-RES pré-calculé n'a été trouvé.")
        run_sim = True

    if run_sim and target_file is None:
        choice = input("Quelle simulation lancer ?\n  1. Rapide (1 an)\n  2. Stabilisée (2 ans, recommandé)\nVotre choix [1/2]: ").lower()
        if choice == '2':
            sim_days = 365 * 2
            stabilize = True
        else:
            sim_days = 365
            stabilize = False

    if target_file:
        print(f"Chargement des résultats depuis '{target_file}'...")
        T_grid_all_times = np.load(target_file)
    else:
        T_grid_all_times = run_full_hires_simulation(sim_days, stabilize)

    # --- Configuration de la visualisation 3D (identique, mais avec les données HI-RES) ---
    SIM_DAYS_DISPLAY = T_grid_all_times.shape[0] // int(24 * 3600 / lib.dt)
    plt.close("all")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    lon_sphere_coords = np.append(LON_HI, LON_HI[0] + 360)
    T_grid_sphere = np.concatenate((T_grid_all_times, T_grid_all_times[:, :, 0:1]), axis=2)
    lon_rad = np.radians(lon_sphere_coords)
    lat_rad = np.radians(90 - LAT_HI)
    lon_mesh, lat_mesh = np.meshgrid(lon_rad, lat_rad)
    R = 1.0
    X = R * np.sin(lat_mesh) * np.cos(lon_mesh)
    Y = R * np.sin(lat_mesh) * np.sin(lon_mesh)
    Z = R * np.cos(lat_mesh)

    vmin, vmax = 220, 320
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.inferno
    T_slice = T_grid_sphere[0, :, :]
    face_colors = cmap(norm(T_slice))
    surf = ax.plot_surface(
        X, Y, Z, facecolors=face_colors, rstride=1, cstride=1,
        antialiased=False, shade=False, edgecolor='none'
    )
    ax.set_axis_off()

    if USE_CARTOPY:
        R_coast = 1.01
        coastline_feature = cfeature.COASTLINE
        for geometry in coastline_feature.geometries():
            for line in (geometry if hasattr(geometry, 'geoms') else [geometry]):
                lons, lats = line.xy
                lon_c_rad = np.radians(np.array(lons))
                lat_c_rad = np.radians(90 - np.array(lats))
                Xc = R_coast * np.sin(lat_c_rad) * np.cos(lon_c_rad)
                Yc = R_coast * np.sin(lat_c_rad) * np.sin(lon_c_rad)
                Zc = R_coast * np.cos(lat_c_rad)
                ax.plot(Xc, Yc, Zc, color='black', linewidth=0.5)

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.01)
    cb.set_label("Température de surface (K)")
    plt.subplots_adjust(bottom=0.2)
    title = fig.suptitle("Jour 0, Heure 0", fontsize=14)
    ax_slider_day = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_day = Slider(ax_slider_day, "Jour", 0, SIM_DAYS_DISPLAY - 1, valinit=0, valstep=1)
    ax_slider_hour = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_hour = Slider(ax_slider_hour, "Heure", 0, 23, valinit=0, valstep=1)

    def _refresh(val):
        day = int(slider_day.val)
        hour = int(slider_hour.val)
        steps_per_day = int(24 * 3600 / lib.dt)
        steps_per_hour = int(3600 / lib.dt)
        time_idx = min(day * steps_per_day + hour * steps_per_hour, T_grid_sphere.shape[0] - 1)
        T_slice = T_grid_sphere[time_idx, :, :]
        new_colors_3d = cmap(norm(T_slice))
        colors_for_faces = new_colors_3d[:-1, :-1, :]
        surf.set_facecolors(colors_for_faces.reshape(-1, 4))
        title.set_text(f"Jour {day}, Heure {hour}")
        fig.canvas.draw_idle()

    slider_day.on_changed(_refresh)
    slider_hour.on_changed(_refresh)
    _refresh(0)
    print("\nFenêtre de visualisation 3D HAUTE RÉSOLUTION ouverte.")
    plt.show()