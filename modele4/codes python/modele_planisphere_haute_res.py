# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Planisphère HAUTE RÉSOLUTION
#
# DESCRIPTION :
# - Interpole les données d'entrée sur une grille plus fine (120x60).
# - La simulation est plus longue mais le rendu est plus détaillé.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import RegularGridInterpolator # NOUVEAU
from tqdm import tqdm
import os

# --- (Le code jusqu'à la simulation est presque identique,
#      sauf pour la gestion des grilles haute résolution) ---
# ... (copier/coller le début du Script 1) ...
# ---------- constantes physiques et de simulation ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (-50 °C)
dt = 1800.0  # pas de temps : 30 min
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo mensuel
# ────────────────────────────────────────────────


def load_albedo_series(
    csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    """Charge les 12 fichiers CSV d'albédo mensuel."""
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
    print("Données d'albédo chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes


# --- Chargement des données au démarrage ---
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_lowres, LAT_lowres, LON_lowres = load_albedo_series(ALBEDO_DIR)
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    exit()

# NOUVEAU : Définition de la grille haute résolution
NLAT_HI, NLON_HI = 60, 120
LAT = np.linspace(LAT_lowres.min(), LAT_lowres.max(), NLAT_HI)
LON = np.linspace(LON_lowres.min(), LON_lowres.max(), NLON_HI)
NLAT, NLON = NLAT_HI, NLON_HI

# NOUVEAU : Fonction pour sur-échantillonner les données
def upscale_data(data_lowres, lat_low, lon_low, lat_hi, lon_hi):
    """Sur-échantillonne les données mensuelles sur la grille haute résolution."""
    print("Sur-échantillonnage des données d'albédo...")
    data_hires = np.zeros((12, len(lat_hi), len(lon_hi)))
    points_hi = np.array(np.meshgrid(lat_hi, lon_hi, indexing='ij'))
    points_hi = np.moveaxis(points_hi, 0, -1)

    for month in range(12):
        interp = RegularGridInterpolator(
            (lat_low, lon_low), data_lowres[month, :, :],
            bounds_error=False, fill_value=None
        )
        data_hires[month, :, :] = interp(points_hi)
    print("Sur-échantillonnage terminé.")
    return data_hires

monthly_albedo = upscale_data(monthly_albedo_lowres, LAT_lowres, LON_lowres, LAT, LON)


# ────────────────────────────────────────────────
# Capacité thermique basée sur l'albédo
# ────────────────────────────────────────────────

_REF_ALBEDO = {
    "ice": 0.60, "water": 0.10, "snow": 0.80, "desert": 0.35,
    "forest": 0.20, "land": 0.15,
}
_CAPACITY_BY_TYPE = {
    "ice": 2.0, "water": 4.18, "snow": 2.0, "desert": 0.8,
    "forest": 1.0, "land": 1.0,
}


def capacite_thermique_massique(albedo: float) -> float:
    """Retourne la capacité thermique massique (kJ kg-1 K-1) pour un albedo."""
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[surf]


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
        C = C_profile[jour_dans_annee]
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



def run_full_simulation(days, result_file=None):
    """Exécute la simulation pour toute la grille et sauvegarde le résultat."""
    # Définir le chemin par défaut dans 'ressources/npy'
    if result_file is None:
        npy_dir = pathlib.Path("ressources/npy")
        npy_dir.mkdir(parents=True, exist_ok=True)
        result_file = npy_dir / "grid_full_hires.npy"
    else:
        result_file = pathlib.Path(result_file)
        result_file.parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print("Lancement de la simulation globale HAUTE RÉSOLUTION...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))
    v_capacite = np.vectorize(capacite_thermique_massique)

    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat, lon = LAT[i], LON[j]

            # Lissage des données d'entrée pour ce point spécifique
            albedo_mensuel_loc = monthly_albedo[:, i, j]
            albedo_journalier = lisser_donnees_annuelles(
                albedo_mensuel_loc, sigma=15.0
            )
            cap_massique_mensuelle = v_capacite(albedo_mensuel_loc) * 1000.0
            cap_surfacique_mensuelle = (
                cap_massique_mensuelle * MASSE_SURFACIQUE_ACTIVE
            )
            C_journalier = lisser_donnees_annuelles(
                cap_surfacique_mensuelle, sigma=15.0
            )

            # Simulation pour ce point
            T0 = 288.15 - 30 * np.sin(np.radians(lat)) ** 2
            T_series = integrate_point_temperature(
                days, np.radians(lat), lon, albedo_journalier, C_journalier, T0
            )
            T_grid[:, i, j] = T_series

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid


# ────────────────────────────────────────────────
# Exécution principale et tracé interactif
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # Optionnel : utiliser cartopy pour un meilleur rendu des côtes
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        USE_CARTOPY = True
    except ImportError:
        USE_CARTOPY = False

    SIM_DAYS = 365
    T_grid_all_times = run_full_simulation(SIM_DAYS)

    # --- Le code de tracé est identique au Script 1 ---
    # ... (copier/coller la section de tracé du Script 1) ...
    plt.close("all")
    if USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 8))  # Augmenter la hauteur
        ax = plt.axes(projection=proj)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))  # Augmenter la hauteur
        proj = None

    initial_T_grid = T_grid_all_times[0, :, :]
    _ORIGIN = "lower"

    if proj is not None:
        im = ax.imshow(
            initial_T_grid, origin=_ORIGIN, extent=[-180, 180, -90, 90],
            transform=proj, cmap="inferno", vmin=220, vmax=320,
            interpolation='bilinear' # Lissage pour la haute résolution
        )
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="k")
            ax.set_global()
    else:
        im = ax.imshow(
            initial_T_grid, origin=_ORIGIN, extent=[-180, 180, -90, 90],
            cmap="inferno", interpolation="bilinear", vmin=220, vmax=320,
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