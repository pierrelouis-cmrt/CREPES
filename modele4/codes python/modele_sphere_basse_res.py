# modele_sphere.py
# ==============================================================================
# MODÈLE 0-D DE TEMPÉRATURE DE SURFACE - VISUALISATION SPHÈRE 3D
#
# DESCRIPTION :
# Ce script exécute le modèle thermique sur une grille globale et visualise
# les résultats sur une sphère 3D interactive.
#
# PS :
# Il est très difficile de faire en sorte que les lignes des côtes ne soient
# pas visibles à travers la sphère
# 
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from matplotlib import cm
from tqdm import tqdm
import pathlib
import os
import sys

# --- Import des modules locaux ---
import fonctions as f
import lib

# --- Dépendances optionnelles pour la visualisation ---
try:
    import cartopy.feature as cfeature
    from cartopy.io import shapereader

    USE_CARTOPY = True
    print("Cartopy détecté. Les lignes de côte seront affichées en 3D.")
except ImportError:
    USE_CARTOPY = False
    print("AVERTISSEMENT: Cartopy non trouvé. Les côtes ne seront pas affichées.")

# --- Chemins des fichiers de résultats (identiques au planisphère) ---
NPY_DIR = pathlib.Path("ressources/npy")
STABILIZED_FILE = NPY_DIR / "grid_lowres_stabilized.npy"
ONEYEAR_FILE = NPY_DIR / "grid_lowres_1yr.npy"


# ────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES GLOBALES
# ────────────────────────────────────────────────
try:
    print("\n--- Chargement des données géospatiales ---")
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = f.load_albedo_series(ALBEDO_DIR)
    NLAT, NLON = len(LAT), len(LON)

    RZSM_CSV_PATH = pathlib.Path(
        "ressources/Cp_humidity/average_rzsm_tout.csv"
    )
    RZSM_GRID, lat_bins_rzsm, lon_bins_rzsm = f.load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )

    CERES_CLIM_DATA = f.load_monthly_cloud_albedo_from_ceres(
        lat_deg=None, lon_deg=None, return_full_map=True
    )
    print("--- Toutes les données ont été chargées avec succès ---\n")

except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR CRITIQUE: Un fichier de ressources est introuvable : {e}")
    sys.exit(1)


# ────────────────────────────────────────────────
# FONCTIONS DE SIMULATION
# ────────────────────────────────────────────────
def f_rhs(T, phinet, C, q_latent):
    """Côté droit de l'équation différentielle (dT/dt)."""
    return (
        phinet
        - q_latent
        + lib.P_em_atm_thermal(lib.Tatm)
        - lib.P_em_surf_thermal(T)
    ) / C


def integrate_point_temperature(
    days,
    lat_rad,
    lon_deg,
    alb_sol_daily,
    alb_nuages_daily,
    C_const,
    q_base,
    T0=288.0,
):
    """Intègre la température pour UN SEUL point géographique."""
    from scipy.ndimage import gaussian_filter1d

    N = int(days * 24 * 3600 / lib.dt)
    T = np.empty(N + 1)
    T[0] = T0

    sign_daynight = np.empty(N)
    for k in range(N):
        t_sec = k * lib.dt
        jour_sim = int(t_sec // 86400) + 1
        _, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        sign_daynight[k] = (
            1.0
            if f.cos_incidence(lat_rad, jour_sim, heure_solaire) > 0
            else -1.0
        )
    q_latent_smoothed = gaussian_filter1d(
        q_base * sign_daynight, sigma=3.0, mode="wrap"
    )

    for k in range(N):
        t_sec = k * lib.dt
        day_of_year, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        jour_sim = int(t_sec // 86400) + 1

        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[k]

        phi_n = lib.P_inc_solar(
            lat_rad, jour_sim, heure_solaire, albedo_sol, albedo_nuages
        )

        X = T[k]
        for _ in range(8):
            F = X - T[k] - lib.dt * f_rhs(X, phi_n, C_const, q_latent_step)
            dF = 1.0 - lib.dt * (-4.0 * lib.sigma * X**3 / C_const)
            if abs(dF) < 1e-9:
                break
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X
    return T


def run_full_simulation(days, stabilize=False):
    """Exécute la simulation pour toute la grille et sauvegarde le résultat."""
    result_file = STABILIZED_FILE if stabilize else ONEYEAR_FILE
    NPY_DIR.mkdir(parents=True, exist_ok=True)

    sim_type = "stabilisée (2 ans)" if stabilize else "rapide (1 an)"
    print(f"\nLancement de la simulation globale {sim_type}...")
    print(f"Les résultats seront sauvegardés dans '{result_file}'")

    N_steps = int(days * 24 * 3600 / lib.dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))

    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat, lon = LAT[i], LON[j]
            albedo_mensuel_loc = monthly_albedo_sol[:, i, j]
            alb_sol_daily = f.lisser_donnees_annuelles(
                albedo_mensuel_loc, sigma=15.0
            )
            alb_nuages_m = CERES_CLIM_DATA.sel(
                lat=lat, lon=lon, method="nearest"
            ).to_numpy()
            alb_nuages_daily = f.lisser_donnees_annuelles(
                alb_nuages_m, sigma=15.0
            )
            lat_idx_rzsm = min(
                np.abs(lat_bins_rzsm[:-1] - lat).argmin(),
                RZSM_GRID.shape[0] - 1,
            )
            lon_idx_rzsm = min(
                np.abs(lon_bins_rzsm[:-1] - lon).argmin(),
                RZSM_GRID.shape[1] - 1,
            )
            rzsm_val = RZSM_GRID[lat_idx_rzsm, lon_idx_rzsm]
            cp_kj = (
                f.compute_cp_from_rzsm(np.array([rzsm_val]))[0]
                if not np.isnan(rzsm_val)
                else f.CP_SEC
            )
            C_const = (cp_kj * 1000.0) * f.RHO_BULK * lib.EPAISSEUR_ACTIVE
            continent = f.continent_finder(lat, lon)
            q_base = lib.Q_LATENT_CONTINENT.get(
                continent, lib.Q_LATENT_CONTINENT["Océan"]
            )
            if lat > 75:
                q_base = 0.0
            T0 = 288.15 - 30 * np.sin(np.radians(lat)) ** 2
            T_series = integrate_point_temperature(
                days,
                np.radians(lat),
                lon,
                alb_sol_daily,
                alb_nuages_daily,
                C_const,
                q_base,
                T0,
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

    if STABILIZED_FILE.exists():
        print(f"Un fichier de simulation stabilisée a été trouvé ('{STABILIZED_FILE.name}').")
        choice = input("Voulez-vous le charger (c) ou relancer une simulation (r) ? [c/r]: ").lower()
        if choice == 'c':
            target_file = STABILIZED_FILE
        else:
            run_sim = True
    elif ONEYEAR_FILE.exists():
        print(f"Un fichier de simulation sur 1 an a été trouvé ('{ONEYEAR_FILE.name}').")
        choice = input("Charger ce fichier (c), ou lancer une nouvelle simulation stabilisée (r) ? [c/r]: ").lower()
        if choice == 'c':
            target_file = ONEYEAR_FILE
        else:
            run_sim = True
    else:
        print("Aucun fichier de simulation pré-calculé n'a été trouvé.")
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
        T_grid_all_times = run_full_simulation(sim_days, stabilize)

    SIM_DAYS_DISPLAY = T_grid_all_times.shape[0] // int(24 * 3600 / lib.dt)

    plt.close("all")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Forcer un aspect parfaitement sphérique en fixant les limites des axes
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_box_aspect([1, 1, 1])

    # Préparation des coordonnées pour la sphère
    lon_sphere = np.append(LON, LON[0] + 360)
    T_grid_sphere = np.concatenate(
        (T_grid_all_times, T_grid_all_times[:, :, 0:1]), axis=2
    )
    lon_rad = np.radians(lon_sphere)
    lat_rad = np.radians(90 - LAT)
    lon_mesh, lat_mesh = np.meshgrid(lon_rad, lat_rad)
    R = 1.0
    X = R * np.sin(lat_mesh) * np.cos(lon_mesh)
    Y = R * np.sin(lat_mesh) * np.sin(lon_mesh)
    Z = R * np.cos(lat_mesh)

    # Configuration de la colormap
    vmin, vmax = 220, 320
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.inferno

    # Tracé initial de la surface
    T_slice = T_grid_sphere[0, :, :]
    face_colors = cmap(norm(T_slice))
    surf = ax.plot_surface(
        X, Y, Z, facecolors=face_colors, rstride=1, cstride=1,
        antialiased=False, shade=False, edgecolor='none',
        zorder=1
    )
    ax.set_axis_off()

    # Ajout des lignes de côte
    if USE_CARTOPY:
        # Augmenter significativement le rayon pour que les côtes soient bien visibles
        R_coast = 1.08
        coastline_feature = cfeature.COASTLINE
        for geometry in coastline_feature.geometries():
            for line in (geometry if hasattr(geometry, 'geoms') else [geometry]):
                lons, lats = line.xy
                lon_c_rad = np.radians(np.array(lons))
                lat_c_rad = np.radians(90 - np.array(lats))
                Xc = R_coast * np.sin(lat_c_rad) * np.cos(lon_c_rad)
                Yc = R_coast * np.sin(lat_c_rad) * np.sin(lon_c_rad)
                Zc = R_coast * np.cos(lat_c_rad)
                ax.plot(Xc, Yc, Zc, color='black', linewidth=1.2, 
                       zorder=10, alpha=0.9)

    # Ajout de la barre de couleur et des sliders
    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
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
        time_idx = day * steps_per_day + hour * steps_per_hour
        time_idx = min(time_idx, T_grid_sphere.shape[0] - 1)
        T_slice = T_grid_sphere[time_idx, :, :]
        new_colors_3d = cmap(norm(T_slice))
        colors_for_faces = new_colors_3d[:-1, :-1, :]
        surf.set_facecolors(colors_for_faces.reshape(-1, 4))
        title.set_text(f"Jour {day}, Heure {hour}")
        fig.canvas.draw_idle()

    slider_day.on_changed(_refresh)
    slider_hour.on_changed(_refresh)
    _refresh(0)

    print("\nFenêtre de visualisation 3D ouverte. Tournez la sphère à la souris.")
    plt.show()