# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# VERSION MISE À JOUR :
# - Ajout de l'albédo des nuages (A1) en plus de l'albédo de
#   surface (A2).
# - Lissage des données mensuelles (albédo sol, albédo nuages)
#   par convolution gaussienne.
# - Utilisation de GeoPandas pour une détection précise
#   des continents.
# - Visualisation du flux de chaleur latente (Q) dans
#   le graphique de sortie.
# - CORRIGÉ : Gestion des géométries nulles dans le shapefile.
# - NOUVEAU : Remplacement des données mock de l'albédo des nuages
#   par un calcul basé sur les données CERES (fichier .nc).
# - NOUVEAU : La capacité thermique est calculée à partir des
#   données d'humidité du sol (RZSM).
# - NOUVEAU : Le flux de chaleur latente (Q) est calculé à partir
#   des taux d'évaporation annuels par continent.
# - NOUVEAU : La variation saisonnière de Q est
#   supprimée. Q est une valeur de base constante (positive le
#   jour, négative la nuit).
# - NOUVEAU : Correction manuelle pour la détection
#   de l'Arctique (Q=0).
# - NOUVEAU (votre demande) : Visualisation du flux de chaleur
#   latent alterné pour un seul jour, avec une transition lissée.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import sys
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

# NOUVEAU : Importations pour la partie géospatiale et NetCDF
try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# NOUVEAU : Importation pour le traitement des données NetCDF
try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# NOUVEAU : Importation pour le griddage des données d'humidité
try:
    from scipy.stats import binned_statistic_2d

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (‑50 °C)
dt = 1800.0  # pas de temps : 30 min
EPAISSEUR_ACTIVE = 0.5  # m (20 cm)

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo de surface (inchangé)
# ────────────────────────────────────────────────


def load_albedo_series(
    csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    """Charge les 12 fichiers CSV d'albédo de surface mensuel."""
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
    print("Données d'albédo de surface chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes


try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    exit()

# ────────────────────────────────────────────────
# Capacité thermique depuis l'humidité du sol (RZSM) (inchangé)
# ────────────────────────────────────────────────

RHO_W = 1000.0
RHO_BULK = 1300.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09
RZSM_CSV_PATH = pathlib.Path("ressources/Cp_humidity/average_rzsm_tout.csv")


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    is_ice = np.isclose(rzsm, 0.9)
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_BULK * (1 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp = CP_SEC + w * (CP_WATER - CP_SEC)
    return np.where(is_ice, CP_ICE, cp)


def load_and_grid_rzsm_data(csv_path: pathlib.Path):
    if not SCIPY_AVAILABLE:
        return None, None, None
    df = pd.read_csv(csv_path)
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    grid_res = 1.0
    lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    lat_bins = np.arange(-90, 90 + grid_res, grid_res)
    statistic, _, _, _ = binned_statistic_2d(
        x=df["lon"],
        y=df["lat"],
        values=df["RZSM"],
        statistic="mean",
        bins=[lon_bins, lat_bins],
    )
    return statistic.T, lat_bins, lon_bins


try:
    RZSM_GRID, RZSM_LAT_BINS, RZSM_LON_BINS = load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )
    if RZSM_GRID is None:
        raise RuntimeError("Scipy manquant ou échec du griddage RZSM.")
    _rzsm_lat_idx = lambda lat: min(
        np.abs(RZSM_LAT_BINS - lat).argmin(), RZSM_GRID.shape[0] - 1
    )
    _rzsm_lon_idx = lambda lon: min(
        np.abs(RZSM_LON_BINS - lon).argmin(), RZSM_GRID.shape[1] - 1
    )
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR: Impossible de charger les données d'humidité du sol : {e}")
    exit()

# ────────────────────────────────────────────────
# Données de chaleur latente (Q) via évaporation (inchangé)
# ────────────────────────────────────────────────

Delta_hvap = 2453000
rho_eau = 1000
Delta_t = 31557600

evap_Eur = 0.49 / Delta_t
evap_Am_Nord = 0.47 / Delta_t
evap_Am_sud = 0.94 / Delta_t
evap_oceanie = 0.41 / Delta_t
evap_Afr = 0.58 / Delta_t
evap_Asi = 0.37 / Delta_t
evap_ocean = 1.40 / Delta_t

phi_Eur = Delta_hvap * rho_eau * evap_Eur
phi_Am_Nord = Delta_hvap * rho_eau * evap_Am_Nord
phi_Am_sud = Delta_hvap * rho_eau * evap_Am_sud
phi_oceanie = Delta_hvap * rho_eau * evap_oceanie
phi_Afr = Delta_hvap * rho_eau * evap_Afr
phi_Asi = Delta_hvap * rho_eau * evap_Asi
phi_ocean = Delta_hvap * rho_eau * evap_ocean

Q_LATENT_CONTINENT = {
    "Europe": phi_Eur,
    "North America": phi_Am_Nord,
    "South America": phi_Am_sud,
    "Oceania": phi_oceanie,
    "Africa": phi_Afr,
    "Asia": phi_Asi,
    "Océan": phi_ocean,
    "Antarctica": 0.0,
}

SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)


def create_continent_finder(shapefile_path: pathlib.Path):
    if not GEOPANDAS_AVAILABLE:
        return lambda lat, lon: "Océan"
    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger le shapefile: {e}")
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        for _, row in world.iterrows():
            if row["geometry"] is not None and row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point


continent_finder = create_continent_finder(SHAPEFILE_PATH)


def get_q_latent_base(lat: float, lon: float) -> float:
    """Récupère la valeur de Q (W m-2) de base pour un point géographique."""
    continent = continent_finder(lat, lon)

    if continent == "Océan" and lat > 75.0:
        print(
            f"Point ({lat:.2f}, {lon:.2f}) dans l'océan Arctique, "
            "correction manuelle vers Q=0."
        )
        continent = "Antarctica"

    q_val = Q_LATENT_CONTINENT.get(continent, Q_LATENT_CONTINENT["Océan"])
    print(
        f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur : "
        f"{continent} (Q base = {q_val:.2f} W m⁻²)"
    )
    return q_val


# ────────────────────────────────────────────────
# Données d'albédo des nuages depuis CERES (inchangé)
# ────────────────────────────────────────────────

CERES_FILE_PATH = (
    pathlib.Path("ressources/albedo")
    / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
)


def load_monthly_cloud_albedo_from_ceres(
    lat_deg: float, lon_deg: float
) -> np.ndarray:
    if not XARRAY_AVAILABLE:
        exit("ERREUR: xarray non installé.")
    try:
        ds = xr.open_dataset(CERES_FILE_PATH, decode_times=True)
    except FileNotFoundError:
        exit(f"ERREUR: Fichier CERES introuvable : {CERES_FILE_PATH}")

    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    toa_sw_all = ds["toa_sw_all_mon"]
    toa_sw_clr = ds["toa_sw_clr_c_mon"]
    solar_in = ds["solar_mon"]
    cloud_albedo_instant = xr.where(
        solar_in > 1e-6, (toa_sw_all - toa_sw_clr) / solar_in, 0.0
    )
    cloud_albedo_monthly_clim = cloud_albedo_instant.groupby(
        "time.month"
    ).mean(dim="time", skipna=True)
    monthly_values = cloud_albedo_monthly_clim.sel(
        lat=lat_deg, lon=lon_deg, method="nearest"
    ).to_numpy()

    if len(monthly_values) != 12:
        monthly_values = np.pad(
            monthly_values, (0, 12 - len(monthly_values)), mode="edge"
        )
    print("Données d'albédo des nuages chargées.")
    return monthly_values


# ────────────────────────────────────────────────
# Lissage et fonctions physiques (inchangés)
# ────────────────────────────────────────────────


def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    jours_par_mois = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    valeurs_journalieres_discontinues = np.repeat(
        valeurs_mensuelles, jours_par_mois
    )
    return gaussian_filter1d(
        valeurs_journalieres_discontinues, sigma=sigma, mode="wrap"
    )


def declination(day):
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)


def cos_incidence(lat_rad, day, hour):
    δ = declination(day)
    H = np.radians(15 * (hour - 12))
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


def phi_net(lat_rad, day, hour, albedo_sol, albedo_nuages):
    phi_entrant = constante_solaire * cos_incidence(lat_rad, day, hour)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)


def f_rhs(T, phinet, C, q_latent):
    return (phinet - q_latent + sigma * Tatm**4 - sigma * T**4) / C


# ────────────────────────────────────────────────
# Intégrateur Backward‑Euler (inchangé)
# ────────────────────────────────────────────────

def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0, sigma_q=3.0):
    """Intégrateur Backward‑Euler avec flux latent journalier lissé.

    - Pré‑calcul du signal jour/nuit sur l'ensemble de la simulation.
    - Convolution gaussienne (sigma_q en *pas de temps*) appliquée au flux
      latent afin que la version lissée soit directement injectée dans le
      modèle thermique.
    """

    N = int(days * 24 * 3600 / dt)  # nombre total de pas
    T = np.empty(N + 1)
    (
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
    ) = (np.empty(N + 1) for _ in range(5))

    T[0] = T0
    lat_rad = np.radians(lat_deg)
    lat_idx, lon_idx = _lat_idx(lat_deg), _lon_idx(lon_deg)

    # Valeur de base de Q pour la zone étudiée
    q_base = get_q_latent_base(lat_deg, lon_deg)

    # ── Pré‑calcul jour/nuit pour toute la simulation ────────────────────
    sign_daynight = np.empty(N)
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        sign_daynight[k] = 1.0 if cos_incidence(lat_rad, jour, heure_solaire) > 0 else -1.0

    q_latent_raw = q_base * sign_daynight
    q_latent_smoothed = gaussian_filter1d(q_latent_raw, sigma=sigma_q, mode="wrap")

    # ── Données d'albédo (sol & nuages) et capacité thermique ────────────
    albedo_sol_m_loc = monthly_albedo_sol[:, lat_idx, lon_idx]
    alb_sol_daily = lisser_donnees_annuelles(albedo_sol_m_loc, sigma=15.0)

    alb_nuages_m = load_monthly_cloud_albedo_from_ceres(lat_deg, lon_deg)
    alb_nuages_daily = lisser_donnees_annuelles(alb_nuages_m, sigma=15.0)

    rzsm_lat_idx = _rzsm_lat_idx(lat_deg)
    rzsm_lon_idx = _rzsm_lon_idx(lon_deg)
    rzsm_val = RZSM_GRID[rzsm_lat_idx, rzsm_lon_idx]
    cp_kj = compute_cp_from_rzsm(np.array([rzsm_val]))[0] if not np.isnan(rzsm_val) else CP_SEC
    C_const = (cp_kj * 1000.0) * RHO_BULK * EPAISSEUR_ACTIVE

    (
        albedo_sol_hist[0],
        albedo_nuages_hist[0],
        C_hist[0],
        q_latent_hist[0],
        q_latent_step_hist[0],
    ) = (
        alb_sol_daily[0],
        alb_nuages_daily[0],
        C_const,
        q_base,
        q_latent_smoothed[0],
    )

    # ── Boucle d'intégration ─────────────────────────────────────────────
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        day_of_year = (jour - 1) % 365

        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[k]

        phi_n = phi_net(lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages)

        # Newton–Raphson implicite
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_step)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

        # Historique pour diagnostics
        albedo_sol_hist[k + 1] = albedo_sol
        albedo_nuages_hist[k + 1] = albedo_nuages
        C_hist[k + 1] = C_const
        q_latent_hist[k + 1] = q_base
        q_latent_step_hist[k + 1] = q_latent_step

    return (
        T,
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
    )

# ────────────────────────────────────────────────
# Fonctions de tracé et exécution principale (MODIFIÉ)
# ────────────────────────────────────────────────


def tracer_comparaison(
    times,
    T,
    albedo_sol_hist,
    albedo_nuages_hist,
    C_hist,
    q_latent_hist,
    q_latent_step_hist,
    titre,
    jour_a_afficher,
    sigma_plot=3.0,
):
    """Trace la température, les albédos, la capacité thermique et le flux latent.

    - Le flux latent affiché est déjà lissé (issu du modèle). Pour encore plus
      de lisibilité on applique à l'affichage une petite convolution gaussienne
      supplémentaire (sigma_plot).
    """

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, height_ratios=[3, 2, 2])
    days_axis = times / 86400
    steps_per_day = int(24 * 3600 / dt)

    # ── Température ───────────────────────────────────────────────────────
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)

    axs[0].plot(days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8, label="Simulation Année 2")
    axs[0].plot(days_axis[start_idx:end_idx + 1], T[start_idx:end_idx + 1] - 273.15, lw=2.5, color="firebrick", label=f"Jour n°{jour_a_afficher}")
    axs[0].set_ylabel("Température surface (°C)")
    axs[0].set_title(titre)
    axs[0].grid(ls=":")
    axs[0].legend()
    axs[0].set_xlim(0, 365)

    # ── Albédos ───────────────────────────────────────────────────────────
    axs[1].set_ylabel("Albédo (sans unité)")
    axs[1].plot(days_axis, albedo_sol_hist, color="tab:blue", lw=2.0, label="Albédo Sol (A2)")
    axs[1].plot(days_axis, albedo_nuages_hist, color="cyan", lw=2.0, ls=":", label="Albédo Nuages (A1)")
    axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5))
    axs[1].legend(loc="upper left")
    axs[1].grid(ls=":")

    # ── Flux latent & capacité thermique ──────────────────────────────────
    q_plot = gaussian_filter1d(q_latent_step_hist, sigma=sigma_plot, mode="wrap")
    axs[2].plot(days_axis, q_plot, color="tab:green", lw=1.5, alpha=0.6, label="Flux Latent lissé (Q)")
    axs[2].set_ylabel("Flux Chaleur Latente (W m⁻²)")
    axs[2].legend(loc="upper left")

    ax3 = axs[2].twinx()
    ax3.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color="tab:red")
    ax3.plot(days_axis, C_hist, color="tab:red", lw=2.0, ls="--", label="Capacité thermique")
    ax3.tick_params(axis="y", labelcolor="tab:red")
    ax3.legend(loc="upper right")

    axs[2].set_xlabel("Jour de l'année (simulation stabilisée)")
    axs[2].grid(ls=":")

    fig.tight_layout()
    plt.show()




if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182  # Solstice d'été, bonne journée pour voir l'effet

    # Pour Paris (Europe)
    lat_sim, lon_sim = 48.5, 2.3
    # Pour l'Amazonie (Amérique du Sud, Q élevé)
    # lat_sim, lon_sim = -3.46, -62.21
    # Pour le Sahara (Afrique, Q modéré, Cp faible)
    # lat_sim, lon_sim = 25.0, 15.0
    # Pour l'Océan Arctique (Pôle Nord)
    # lat_sim, lon_sim = 82.0, 135.0
    # Pour l'Antarctique (Pôle Sud)
    # lat_sim, lon_sim = -76.0, 100.0

    print(
        f"Lancement de la simulation pour Lat={lat_sim}N, Lon={lon_sim}E..."
    )
    (
        T_full,
        alb_sol_full,
        alb_nuages_full,
        C_full,
        q_latent_full,
        q_latent_step_full,
    ) = backward_euler(jours_de_simulation, lat_sim, lon_sim)

    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt
    (
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_yr2,
        q_latent_step_yr2,
    ) = (
        arr[steps_per_year:]
        for arr in [
            T_full,
            alb_sol_full,
            alb_nuages_full,
            C_full,
            q_latent_full,
            q_latent_step_full,
        ]
    )

    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_yr2,
        q_latent_step_yr2,
        f"Simulation (Q constant) pour Lat={lat_sim}, Lon={lon_sim}",
        jour_a_afficher,
    )