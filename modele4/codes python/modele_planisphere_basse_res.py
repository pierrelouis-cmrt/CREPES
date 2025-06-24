# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Planisphère (Version Modulaire)
#
# DESCRIPTION :
# - Utilise le modèle physique avancé en appelant les fonctions
#   des modules externes `fonctions.py` et `lib.py`.
# - Le modèle inclut :
#   - Albédo de surface (A2) et albédo des nuages (A1).
#   - Capacité thermique calculée depuis l'humidité du sol (RZSM).
#   - Flux de chaleur latent (Q) basé sur la géographie.
# - Le script principal orchestre la simulation globale et la
#   visualisation interactive.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pathlib
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binned_statistic_2d
import pandas as pd
from tqdm import tqdm
import os
import xarray as xr

# Import des fonctions de modélisation depuis les fichiers fournis
import fonctions as f
import lib

# Optionnel : utiliser cartopy pour un meilleur rendu des côtes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

# ---------- constantes physiques et de simulation ----------
# Les constantes sont importées depuis les modules pour la cohérence
sigma = lib.sigma
Tatm = f.Tatm  # Utilisation de la valeur de fonctions.py (-50°C)
dt = f.dt
EPAISSEUR_ACTIVE = f.EPAISSEUR_ACTIVE

# --- Chargement et préparation des données au démarrage ---
try:
    # 1. Données d'albédo de surface
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = f.load_albedo_series(ALBEDO_DIR)
    NLAT, NLON = len(LAT), len(LON)

    # 2. Données d'humidité du sol (RZSM) pour la capacité thermique
    RZSM_CSV_PATH = f.RZSM_CSV_PATH
    df_rzsm = pd.read_csv(RZSM_CSV_PATH)
    df_rzsm["lon"] = ((df_rzsm["lon"] + 180) % 360) - 180
    lon_bins = np.arange(-180, 180 + 1.0, 1.0)
    lat_bins = np.arange(-90, 90 + 1.0, 1.0)
    RZSM_GRID, _, _, _ = binned_statistic_2d(
        x=df_rzsm["lon"],
        y=df_rzsm["lat"],
        values=df_rzsm["RZSM"],
        statistic="mean",
        bins=[lon_bins, lat_bins],
    )
    RZSM_GRID = RZSM_GRID.T  # Transposer pour avoir (lat, lon)
    print("Données d'humidité du sol (RZSM) chargées et grillées.")

    # 3. Données d'albédo des nuages (CERES)
    # NOTE: Correction des noms de variables pour correspondre au fichier .nc
    ds_ceres = xr.open_dataset(f.CERES_FILE_PATH, decode_times=True)
    ds_ceres = ds_ceres.assign_coords(
        lon=(((ds_ceres.lon + 180) % 360) - 180)
    ).sortby("lon")
    toa_sw_all = ds_ceres["toa_sw_all_mon"]
    toa_sw_clr = ds_ceres["toa_sw_clr_c_mon"]
    solar_in = ds_ceres["solar_mon"]
    cloud_albedo_instant = xr.where(
        solar_in > 1e-6, (toa_sw_all - toa_sw_clr) / solar_in, 0.0
    )
    CERES_CLIM_DATA = cloud_albedo_instant.groupby("time.month").mean(
        dim="time", skipna=True
    )
    print("Données d'albédo des nuages (CERES) chargées.")

except FileNotFoundError as e:
    print(f"ERREUR: Un fichier de ressources est introuvable : {e}")
    exit()


# ────────────────────────────────────────────────
# Fonctions de résolution et de simulation
# ────────────────────────────────────────────────


def f_rhs(T, phinet, C, q_latent):
    """Côté droit de l'équation différentielle du modèle avancé."""
    return (phinet - q_latent + sigma * Tatm**4 - sigma * T**4) / C


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
    """Intègre la température pour UN SEUL point avec le modèle avancé."""
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0

    # Pré-calcul du signal jour/nuit lissé pour le flux latent
    sign_daynight = np.empty(N)
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        sign_daynight[k] = (
            1.0 if f.cos_incidence(lat_rad, jour, heure_solaire) > 0 else -1.0
        )
    q_latent_smoothed = gaussian_filter1d(
        q_base * sign_daynight, sigma=3.0, mode="wrap"
    )

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        day_of_year = (jour - 1) % 365

        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[k]

        # Appel aux fonctions de lib.py pour les flux physiques
        phi_n = lib.P_inc_solar(
            lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages
        )

        # Newton pour Backward-Euler
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_step)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X
    return T


def run_full_simulation(days, result_file=None):
    """Exécute la simulation pour toute la grille et sauvegarde le résultat."""
    if result_file is None:
        npy_dir = pathlib.Path("ressources/npy")
        npy_dir.mkdir(parents=True, exist_ok=True)
        result_file = npy_dir / "grid_advanced_model.npy"
    else:
        result_file = pathlib.Path(result_file)
        result_file.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print("Lancement de la simulation globale (modèle avancé)...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))

    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat, lon = LAT[i], LON[j]

            # 1. Albédo de surface (lissé)
            albedo_mensuel_loc = monthly_albedo_sol[:, i, j]
            alb_sol_daily = f.lisser_donnees_annuelles(
                albedo_mensuel_loc, sigma=15.0
            )

            # 2. Albédo des nuages (lissé)
            alb_nuages_m = CERES_CLIM_DATA.sel(
                lat=lat, lon=lon, method="nearest"
            ).to_numpy()
            alb_nuages_daily = f.lisser_donnees_annuelles(
                alb_nuages_m, sigma=15.0
            )

            # 3. Capacité thermique (via RZSM)
            lat_idx_rzsm = min(
                np.abs(lat_bins[:-1] - lat).argmin(), RZSM_GRID.shape[0] - 1
            )
            lon_idx_rzsm = min(
                np.abs(lon_bins[:-1] - lon).argmin(), RZSM_GRID.shape[1] - 1
            )
            rzsm_val = RZSM_GRID[lat_idx_rzsm, lon_idx_rzsm]
            cp_kj = (
                f.compute_cp_from_rzsm(np.array([rzsm_val]))[0]
                if not np.isnan(rzsm_val)
                else f.CP_SEC
            )
            C_const = (cp_kj * 1000.0) * f.RHO_BULK * EPAISSEUR_ACTIVE

            # 4. Flux de chaleur latent (Q) - Appel à lib.py
            q_base = lib.P_em_surf_evap(lat, lon)

            # Lancement de la simulation pour ce point
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

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid


# ────────────────────────────────────────────────
# Exécution principale et affichage du planisphère
# ────────────────────────────────────────────────
if __name__ == "__main__":
    SIM_DAYS = 365  # Simuler une année complète pour la stabilisation
    T_grid_all_times = run_full_simulation(SIM_DAYS)

    plt.close("all")
    if USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=proj)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        proj = None

    initial_T_grid = T_grid_all_times[0, :, :]
    _ORIGIN = "lower"

    if proj is not None:
        im = ax.imshow(
            initial_T_grid,
            origin=_ORIGIN,
            extent=[-180, 180, -90, 90],
            transform=proj,
            cmap="inferno",
            vmin=220,
            vmax=320,
        )
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="k")
            ax.set_global()
    else:
        im = ax.imshow(
            initial_T_grid,
            origin=_ORIGIN,
            extent=[-180, 180, -90, 90],
            cmap="inferno",
            interpolation="nearest",
            vmin=220,
            vmax=320,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cb.set_label("Température de surface (K)")

    # --- Curseurs interactifs pour le jour et l'heure ---
    plt.subplots_adjust(bottom=0.25)
    ax_slider_day = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_day = Slider(
        ax_slider_day, "Jour", 0, SIM_DAYS - 1, valinit=0, valstep=1, color="0.5"
    )

    ax_slider_hour = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_hour = Slider(
        ax_slider_hour, "Heure", 0, 23, valinit=12, valstep=1, color="0.5"
    )

    def _refresh(val):
        day = int(slider_day.val)
        hour = int(slider_hour.val)

        steps_per_day = int(24 * 3600 / dt)
        steps_per_hour = int(3600 / dt)
        time_step_index = day * steps_per_day + hour * steps_per_hour
        time_step_index = min(time_step_index, T_grid_all_times.shape[0] - 1)

        T_slice = T_grid_all_times[time_step_index, :, :]
        im.set_data(T_slice)
        ax.set_title(f"Température de surface (K) - Jour {day}, Heure {hour}")
        fig.canvas.draw_idle()

    slider_day.on_changed(_refresh)
    slider_hour.on_changed(_refresh)
    _refresh(0)  # Initialiser la vue

    plt.show()