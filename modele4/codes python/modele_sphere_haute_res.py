# ---------------------------------------------------------------
# Modèle 0-D – Sphère 3D HAUTE RÉSOLUTION (Optimisé et Unifié)
#
# DESCRIPTION :
# - Script final qui combine la simulation optimisée et la visualisation 3D.
# - Pré-calcule toutes les grilles de paramètres pour une performance maximale.
# - Produit et utilise un unique fichier .npy (grid_hires_optimized_365d.npy).
# - Simule et affiche 1 année complète.
# - Affiche les côtes et assure un rendu lisse sur la sphère.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pathlib
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import os
import xarray as xr

# Import des modules contenant la logique du modèle
import fonctions as f
import lib

# Optionnel : utiliser cartopy pour le tracé des côtes
try:
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False
    print("Avertissement: Cartopy non trouvé. Les côtes ne seront pas affichées.")

# ---------- Constantes physiques et de simulation (depuis les modules) ----------
sigma = lib.sigma
Tatm = lib.Tatm
dt = lib.dt
EPAISSEUR_ACTIVE = lib.EPAISSEUR_ACTIVE

# --- Chargement des données et définition de la grille HAUTE RÉSOLUTION ---
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_lowres, LAT_lowres, LON_lowres = f.load_albedo_series(ALBEDO_DIR)
    RZSM_GRID_lowres, RZSM_LAT_lowres, RZSM_LON_lowres = f.load_and_grid_rzsm_data(
        pathlib.Path("ressources/Cp_humidity/average_rzsm_tout.csv")
    )
    ds_ceres = xr.open_dataset(f.CERES_FILE_PATH, decode_times=True).load()
except (FileNotFoundError, TypeError) as e:
    print(f"ERREUR: Fichier de ressources introuvable ou erreur de chargement : {e}")
    exit()

NLAT_HI, NLON_HI = 60, 120
LAT_HI = np.linspace(LAT_lowres.min(), LAT_lowres.max(), NLAT_HI)
LON_HI = np.linspace(LON_lowres.min(), LON_lowres.max(), NLON_HI)
print("Données sources chargées, grille haute résolution définie.")

# --- OPTIMISATION : Pré-calcul des grilles de paramètres ---
print("Optimisation: Pré-calcul des grilles de paramètres...")

def upscale_grid(data_lowres, lat_low, lon_low, lat_hi, lon_hi):
    points_hi = np.array(np.meshgrid(lat_hi, lon_hi, indexing="ij"))
    points_hi = np.moveaxis(points_hi, 0, -1)
    interp = RegularGridInterpolator(
        (lat_low, lon_low), data_lowres, bounds_error=False, fill_value=None
    )
    return interp(points_hi)

monthly_albedo_hi = np.array([
    upscale_grid(monthly_albedo_lowres[m], LAT_lowres, LON_lowres, LAT_HI, LON_HI)
    for m in range(12)
])
RZSM_GRID_hi = upscale_grid(
    RZSM_GRID_lowres, RZSM_LAT_lowres[:-1], RZSM_LON_lowres[:-1], LAT_HI, LON_HI
)

Q_GRID_HI = np.array([[lib.P_em_surf_evap(lat, lon, verbose=False) for lon in LON_HI] for lat in LAT_HI])
cp_kj_grid = f.compute_cp_from_rzsm(RZSM_GRID_hi)
C_GRID_HI = (cp_kj_grid * 1000.0) * f.RHO_BULK * EPAISSEUR_ACTIVE

jours_par_mois = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
albedo_sol_daily_grid = gaussian_filter1d(np.repeat(monthly_albedo_hi, jours_par_mois, axis=0), sigma=15.0, axis=0, mode="wrap")

ds_ceres_sorted = ds_ceres.assign_coords(lon=(((ds_ceres.lon + 180) % 360) - 180)).sortby("lon")
ceres_clim = ds_ceres_sorted["toa_sw_all_mon"].groupby("time.month").mean(dim="time", skipna=True)
monthly_cloud_albedo_hi = ceres_clim.sel(lat=xr.DataArray(LAT_HI, dims="lat"), lon=xr.DataArray(LON_HI, dims="lon"), method="nearest").to_numpy()
albedo_nuages_daily_grid = gaussian_filter1d(np.repeat(monthly_cloud_albedo_hi, jours_par_mois, axis=0), sigma=15.0, axis=0, mode="wrap")

print("Pré-calcul terminé.")

# ────────────────────────────────────────────────
# Fonctions de résolution et de simulation
# ────────────────────────────────────────────────

def f_rhs(T, phinet, C, q_latent):
    return (phinet - q_latent + sigma * Tatm**4 - sigma * T**4) / C

def integrate_point_temperature(days, lat_rad, lon_deg, alb_sol_daily, alb_nuages_daily, C_const, q_base, T0):
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0
    sign_daynight = np.array([1.0 if f.cos_incidence(lat_rad, int(k*dt//86400)+1, ((k*dt/3600)+lon_deg/15)%24) > 0 else -1.0 for k in range(N)])
    q_latent_smoothed = gaussian_filter1d(q_base * sign_daynight, sigma=3.0, mode="wrap")

    for k in range(N):
        day_of_year = int(k * dt // 86400) % 365
        heure_solaire = ((k * dt / 3600.0) + lon_deg / 15.0) % 24.0
        phi_n = lib.P_inc_solar(lat_rad, day_of_year + 1, heure_solaire, alb_sol_daily[day_of_year], alb_nuages_daily[day_of_year])
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_smoothed[k])
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6: break
        T[k + 1] = X
    return T

def run_full_simulation(days, result_file=None):
    if result_file is None:
        npy_dir = pathlib.Path("ressources/npy")
        npy_dir.mkdir(parents=True, exist_ok=True)
        result_file = npy_dir / f"grid_hires_optimized_{days}d.npy"

    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print(f"Lancement de la simulation HAUTE RÉSOLUTION pour {days} jours...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT_HI, NLON_HI))

    for i in tqdm(range(NLAT_HI), desc="Progression (latitude)"):
        for j in range(NLON_HI):
            lat, lon = LAT_HI[i], LON_HI[j]
            T0 = 288.15 - 30 * np.sin(np.radians(lat)) ** 2
            T_series = integrate_point_temperature(
                days, np.radians(lat), lon,
                albedo_sol_daily_grid[:, i, j], albedo_nuages_daily_grid[:, i, j],
                C_GRID_HI[i, j], Q_GRID_HI[i, j], T0
            )
            T_grid[:, i, j] = T_series

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid

# ────────────────────────────────────────────────
# Exécution principale et tracé interactif 3D
# ────────────────────────────────────────────────
if __name__ == "__main__":
    SIM_DAYS = 365
    T_grid_all_times = run_full_simulation(SIM_DAYS)

    plt.close("all")
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # Préparer les données pour une sphère continue
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
        R_coast = 1.005
        coastline_feature = cfeature.COASTLINE
        for geometry in coastline_feature.geometries():
            for line in (geometry if hasattr(geometry, 'geoms') else [geometry]):
                lons, lats = line.xy
                lon_c_rad = np.radians(np.array(lons))
                lat_c_rad = np.radians(90 - np.array(lats))
                Xc = R_coast * np.sin(lat_c_rad) * np.cos(lon_c_rad)
                Yc = R_coast * np.sin(lat_c_rad) * np.sin(lon_c_rad)
                Zc = R_coast * np.cos(lat_c_rad)
                ax.plot(Xc, Yc, Zc, color='black', linewidth=0.5, transform=ax.transData)

    m = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(m, ax=ax, shrink=0.5, aspect=10, pad=0.01)
    cb.set_label("Température de surface (K)")

    plt.subplots_adjust(bottom=0.2)
    ax_slider_day = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_day = Slider(ax_slider_day, "Jour", 0, SIM_DAYS - 1, valinit=0, valstep=1, color="0.5")
    ax_slider_hour = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_hour = Slider(ax_slider_hour, "Heure", 0, 23, valinit=12, valstep=1, color="0.5")

    def _refresh(val):
        day = int(slider_day.val)
        hour = int(slider_hour.val)
        steps_per_day = int(24 * 3600 / dt)
        steps_per_hour = int(3600 / dt)
        time_idx = min(day * steps_per_day + hour * steps_per_hour, T_grid_sphere.shape[0] - 1)
        
        T_slice = T_grid_sphere[time_idx, :, :]
        new_colors_3d = cmap(norm(T_slice))
        # Les couleurs sont appliquées aux faces, qui sont (N-1)x(M-1)
        face_colors_flat = new_colors_3d[:-1, :-1, :].reshape(-1, 4)
        surf.set_facecolors(face_colors_flat)
        
        ax.set_title(f"Jour {day}, Heure {hour}", y=0.95)
        fig.canvas.draw_idle()

    slider_day.on_changed(_refresh)
    slider_hour.on_changed(_refresh)
    _refresh(0)
    plt.show()