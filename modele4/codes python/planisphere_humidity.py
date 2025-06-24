# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Planisphère (Version Complète)
#
# DESCRIPTION :
# - Intègre le modèle avancé (albédo lissé, capacité statique) sur une grille globale.
# - Le flux solaire est calculé avec les fonctions astronomiques précises.
# - L'intégration temporelle utilise un schéma Backward-Euler implicite.
# - La simulation est effectuée pour une grille de points sur tout le globe.
# - La visualisation est interactive avec des curseurs pour le jour et l'heure.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os
import subprocess


try:
    import numpy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", numpy])
    import numpy

try:
    import matplotlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", matplotlib])
    import matplotlib

try:
    import math
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", math])
    import math

try:
    import pathlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pathlib])
    import pathlib

try:
    import pandas
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pandas])
    import pandas

try:
    import scipy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", scipy])
    import scipy

try:
    import os
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", os])
    import os

try:
    import tqdm
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", tqdm])
    import tqdm

# Optionnel : utiliser cartopy pour un meilleur rendu des côtes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

# ---------- constantes physiques et de simulation ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (-50 °C)
dt = 1800.0  # pas de temps : 30 min
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# --- Constantes pour le calcul de la capacité thermique via RZSM ---
RHO_W = 1000.0  # Masse volumique de l'eau (kg/m^3)
RHO_B = 1300.0  # Masse volumique du sol sec (kg/m^3)
CP_SEC = 0.8  # Capacité thermique massique du sol sec (kJ/kg/K)
CP_WATER = 4.187  # Capacité thermique massique de l'eau (kJ/kg/K)
CP_ICE = 2.09  # Capacité thermique massique de la glace (kJ/kg/K)


# ────────────────────────────────────────────────
# DATA – Chargement des données
# ────────────────────────────────────────────────


def load_monthly_grid_data(
    csv_dir: str | pathlib.Path, pattern: str = "data_{:02d}.csv"
):
    """Charge 12 fichiers CSV de données mensuelles au format grille."""
    csv_dir = pathlib.Path(csv_dir)
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    cubes: list[np.ndarray] = []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    return np.stack(cubes, axis=0), latitudes, longitudes


def load_and_grid_static_data(
    csv_path: pathlib.Path, lat_grid: np.ndarray, lon_grid: np.ndarray
):
    """Charge un CSV de points (lat,lon,val) et le projette sur une grille."""
    print(f"Chargement et griddage des données depuis '{csv_path}'...")
    df = pd.read_csv(csv_path)
    grid = np.full((len(lat_grid), len(lon_grid)), np.nan)

    # Création de fonctions d'indexation locales
    _lat_idx = lambda lat: np.abs(lat_grid - lat).argmin()
    _lon_idx = lambda lon: np.abs(lon_grid - lon).argmin()

    for _, row in df.iterrows():
        idx_lat = _lat_idx(row["lat"])
        idx_lon = _lon_idx(row["lon"])
        grid[idx_lat, idx_lon] = row["RZSM"]

    # Remplir les potentiels NaN restants (si le CSV ne couvre pas tout)
    grid = pd.DataFrame(grid).interpolate(method="linear", axis=0).bfill().values
    print("Griddage des données statiques terminé.")
    return grid


# --- Chargement des données au démarrage ---
try:
    # Chargement de l'albédo mensuel
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo, LAT, LON = load_monthly_grid_data(
        ALBEDO_DIR, pattern="albedo{:02d}.csv"
    )
    print("Données d'albédo mensuel chargées.")
    NLAT, NLON = len(LAT), len(LON)

    # Chargement de l'humidité du sol (RZSM) statique
    RZSM_CSV = pathlib.Path("temp/average_rzsm_tout.csv")
    static_rzsm_grid = load_and_grid_static_data(RZSM_CSV, LAT, LON)

except FileNotFoundError as e:
    print(f"ERREUR: Un fichier ou dossier de ressources est introuvable : {e}")
    exit()


# ────────────────────────────────────────────────
# Capacité thermique basée sur l'humidité du sol (RZSM)
# ────────────────────────────────────────────────


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    """
    Calcule la capacité calorifique (kJ/kg/K) de manière vectorielle.
    Gère la valeur spéciale pour la glace (RZSM=0.9).
    """
    is_ice = np.isclose(rzsm, 0.9)
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_B * (1 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp = CP_SEC + w * (CP_WATER - CP_SEC)
    return np.where(is_ice, CP_ICE, cp)


# ────────────────────────────────────────────────
# Lissage des données annuelles par convolution gaussienne
# ────────────────────────────────────────────────


def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """Lisse 12 valeurs mensuelles en un profil journalier continu (365 j)."""
    jours_par_mois = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    valeurs_journalieres_discontinues = np.repeat(
        valeurs_mensuelles, jours_par_mois
    )
    return gaussian_filter1d(
        valeurs_journalieres_discontinues, sigma=sigma, mode="wrap"
    )


# ────────────────────────────────────────────────
# Fonctions astronomiques et bilan énergétique
# ────────────────────────────────────────────────


def declination(day):
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)


def cos_incidence(lat_rad, day, hour):
    δ = declination(day)
    H = np.radians(15 * (hour - 12))
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


def phi_net(lat_rad, day, hour, albedo):
    return constante_solaire * cos_incidence(lat_rad, day, hour) * (1 - albedo)


def f_rhs(T, phinet, C):
    return (phinet + sigma * Tatm**4 - sigma * T**4) / C


# ────────────────────────────────────────────────
# Intégrateur et simulation globale
# ────────────────────────────────────────────────


def integrate_point_temperature(
    days, lat_rad, lon_deg, albedo_profile, C_profile, T0=288.0
):
    """Intègre la température pour UN SEUL point avec des profils déjà lissés."""
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        jour_dans_annee = (jour - 1) % 365

        albedo = albedo_profile[jour_dans_annee]
        C = C_profile[jour_dans_annee]  # C est constant mais on garde l'indexation
        phi_n = phi_net(lat_rad, jour, heure_solaire, albedo)

        # Newton pour Backward-Euler
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X
    return T


def run_full_simulation(days, result_file="temp_grid_full.npy"):
    """Exécute la simulation pour toute la grille et sauvegarde le résultat."""
    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print("Lancement de la simulation globale (cela peut prendre du temps)...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))

    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat, lon = LAT[i], LON[j]

            # Lissage de l'albédo (variable) pour ce point
            albedo_mensuel_loc = monthly_albedo[:, i, j]
            albedo_journalier = lisser_donnees_annuelles(
                albedo_mensuel_loc, sigma=15.0
            )

            # MODIFIÉ : Calcul de la capacité thermique à partir du RZSM statique
            rzsm_loc = static_rzsm_grid[i, j]
            cap_massique = compute_cp_from_rzsm(rzsm_loc) * 1000.0  # J/kg/K
            cap_surfacique = cap_massique * MASSE_SURFACIQUE_ACTIVE  # J/m^2/K

            # Créer un profil annuel constant pour la capacité thermique
            C_journalier_constant = np.full(365, cap_surfacique)

            # Simulation pour ce point
            T0 = 288.15 - 30 * np.sin(np.radians(lat)) ** 2
            T_series = integrate_point_temperature(
                days,
                np.radians(lat),
                lon,
                albedo_journalier,
                C_journalier_constant,
                T0,
            )
            T_grid[:, i, j] = T_series

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid


# ────────────────────────────────────────────────
# Exécution principale et tracé interactif
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
    _refresh(0)

    plt.show()