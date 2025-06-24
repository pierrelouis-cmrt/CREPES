# fonctions.py
# ==============================================================================
# BIBLIOTHÈQUE DE FONCTIONS UTILITAIRES
# Rôle : Fournit des fonctions pour le chargement et le traitement des données
#        géospatiales (albédo, humidité du sol), les calculs de géométrie
#        solaire, et la préparation des paramètres pour la simulation.
# ==============================================================================

import numpy as np
import pandas as pd
import pathlib
import subprocess
import sys
from math import pi
from scipy.ndimage import gaussian_filter1d

# --- Import et gestion des dépendances optionnelles ---
try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_AVAILABLE = True
except ImportError:
    print(
        "AVERTISSEMENT: GeoPandas non trouvé. La détection des continents sera désactivée."
    )
    GEOPANDAS_AVAILABLE = False

try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    print("AVERTISSEMENT: xarray non trouvé. L'albédo des nuages sera désactivé.")
    XARRAY_AVAILABLE = False

try:
    from scipy.stats import binned_statistic_2d

    SCIPY_AVAILABLE = True
except ImportError:
    print(
        "AVERTISSEMENT: Scipy non trouvé. Le griddage RZSM sera désactivé."
    )
    SCIPY_AVAILABLE = False

# --- Import du module de constantes physiques ---
import lib as lib


# ────────────────────────────────────────────────
# FONCTIONS DE PRÉPARATION DE LA SIMULATION
# ────────────────────────────────────────────────
def prepare_simulation_inputs(lat_deg, lon_deg, total_days, dt, sigma_q=3.0):
    """
    Charge, traite et pré-calcule toutes les données nécessaires à la simulation.

    Cette fonction centralise la préparation des séries temporelles (albédo,
    flux latent) et des constantes (capacité thermique) pour un point donné.

    IN:
        lat_deg (float): Latitude du point de simulation [degrés].
        lon_deg (float): Longitude du point de simulation [degrés].
        total_days (int): Nombre total de jours de la simulation.
        dt (float): Pas de temps de la simulation [s].
        sigma_q (float): Écart-type pour le lissage gaussien du flux latent.

    OUT:
        dict: Un dictionnaire contenant tous les paramètres prêts à l'emploi:
              - 'C': Capacité thermique surfacique [J m⁻² K⁻¹]
              - 'q_base': Flux latent de base pour le lieu [W m⁻²]
              - 'albedo_sol_daily': Série de 365 jours d'albédo du sol
              - 'albedo_nuages_daily': Série de 365 jours d'albédo des nuages
              - 'q_latent_smoothed': Série temporelle du flux latent lissé
    """
    print("--- Préparation des paramètres de simulation ---")

    # --- 1. Capacité thermique (C) ---
    RZSM_CSV_PATH = pathlib.Path(
        "ressources/Cp_humidity/average_rzsm_tout.csv"
    )
    RZSM_GRID, RZSM_LAT_BINS, RZSM_LON_BINS = load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )
    if RZSM_GRID is None:
        raise RuntimeError("Échec du chargement ou griddage des données RZSM.")

    _rzsm_lat_idx = lambda lat: min(
        np.abs(RZSM_LAT_BINS - lat).argmin(), RZSM_GRID.shape[0] - 1
    )
    _rzsm_lon_idx = lambda lon: min(
        np.abs(RZSM_LON_BINS - lon).argmin(), RZSM_GRID.shape[1] - 1
    )
    rzsm_val = RZSM_GRID[_rzsm_lat_idx(lat_deg), _rzsm_lon_idx(lon_deg)]
    cp_kj = (
        compute_cp_from_rzsm(np.array([rzsm_val]))[0]
        if not np.isnan(rzsm_val)
        else CP_SEC
    )
    C_const = (cp_kj * 1000.0) * RHO_BULK * lib.EPAISSEUR_ACTIVE
    print(f"Capacité thermique calculée : {C_const:.2e} J m⁻² K⁻¹")

    # --- 2. Albédos (Sol et Nuages) ---
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
    albedo_sol_m_loc = monthly_albedo_sol[:, _lat_idx(lat_deg), _lon_idx(lon_deg)]
    alb_sol_daily = lisser_donnees_annuelles(albedo_sol_m_loc, sigma=15.0)

    alb_nuages_m = load_monthly_cloud_albedo_from_ceres(lat_deg, lon_deg)
    alb_nuages_daily = lisser_donnees_annuelles(alb_nuages_m, sigma=15.0)

    # --- 3. Flux de chaleur latente (Q) ---
    q_base = lib.P_em_surf_evap(lat_deg, lon_deg, verbose=True)
    N = int(total_days * 24 * 3600 / dt)
    sign_daynight = np.empty(N)
    lat_rad = np.radians(lat_deg)

    for k in range(N):
        t_sec = k * dt
        jour, heure_solaire = get_time_variables(t_sec, lon_deg)
        sign_daynight[k] = (
            1.0 if cos_incidence(lat_rad, jour + 1, heure_solaire) > 0 else -1.0
        )

    q_latent_raw = q_base * sign_daynight
    q_latent_smoothed = gaussian_filter1d(
        q_latent_raw, sigma=sigma_q, mode="wrap"
    )
    print("--- Préparation terminée ---")

    return {
        "C": C_const,
        "q_base": q_base,
        "albedo_sol_daily": alb_sol_daily,
        "albedo_nuages_daily": alb_nuages_daily,
        "q_latent_smoothed": q_latent_smoothed,
    }


# ────────────────────────────────────────────────
# GÉOMÉTRIE SOLAIRE ET TEMPS
# ────────────────────────────────────────────────
def get_time_variables(t_sec, lon_deg):
    """
    Calcule le jour de l'année et l'heure solaire locale.

    IN:
        t_sec (float): Temps écoulé depuis le début de la simulation [s].
        lon_deg (float): Longitude du lieu [degrés].

    OUT:
        tuple (int, float):
            - Jour de l'année (0-364).
            - Heure solaire locale (0-24).
    """
    jour_sim = int(t_sec // 86400)
    day_of_year = jour_sim % 365
    heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
    return day_of_year, heure_solaire


def declination(day):
    """
    Calcule la déclinaison solaire (en radians) pour un jour donné.

    IN:
        day (int): Jour de l'année (1-365).

    OUT:
        float: Déclinaison solaire [radians].
    """
    return np.radians(23.44) * np.sin(2 * pi * (284 + day) / 365)


def cos_incidence(lat_rad, day, hour):
    """
    Calcule le cosinus de l’angle d’incidence du rayonnement solaire.

    IN:
        lat_rad (float): Latitude [radians].
        day (int): Jour de l'année (1-365).
        hour (float): Heure solaire locale (0-24).

    OUT:
        float: Cosinus de l'angle d'incidence (valeur >= 0).
    """
    δ = declination(day)
    H = np.radians(15 * (hour - 12))  # Angle horaire en radians
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


# ────────────────────────────────────────────────
# CAPACITÉ THERMIQUE (via Humidité du Sol RZSM)
# ────────────────────────────────────────────────
RHO_W = 1000.0
RHO_BULK = 1300.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    """
    Calcule la capacité thermique massique (c_p) en kJ/kg/K depuis l'humidité du sol.

    IN:
        rzsm (np.ndarray): Teneur en eau volumique du sol (Root Zone Soil Moisture).

    OUT:
        np.ndarray: Capacité thermique massique [kJ kg⁻¹ K⁻¹].
    """
    is_ice = np.isclose(rzsm, 0.9)  # Heuristique pour la glace
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_BULK * (1 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp = CP_SEC + w * (CP_WATER - CP_SEC)
    return np.where(is_ice, CP_ICE, cp)


def load_and_grid_rzsm_data(csv_path: pathlib.Path):
    """
    Charge et grille les données RZSM depuis un CSV sur une grille régulière.

    IN:
        csv_path (pathlib.Path): Chemin vers le fichier CSV des données RZSM.

    OUT:
        tuple:
            - Grille 2D des valeurs RZSM moyennes.
            - Bords des bins en latitude.
            - Bords des bins en longitude.
    """
    if not SCIPY_AVAILABLE:
        return None, None, None
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier RZSM introuvable: {csv_path}")
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
    print("Données d'humidité du sol (RZSM) chargées et grillées.")
    return statistic.T, lat_bins, lon_bins


# ────────────────────────────────────────────────
# TRAITEMENT DE DONNÉES ANNUELLES
# ────────────────────────────────────────────────
def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """
    Lisse une série de 12 valeurs mensuelles en une série de 365 valeurs journalières.

    IN:
        valeurs_mensuelles (np.ndarray): Tableau de 12 valeurs.
        sigma (float): Écart-type pour le filtre gaussien (en jours).

    OUT:
        np.ndarray: Tableau de 365 valeurs journalières lissées.
    """
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
# CHARGEMENT DES DONNÉES D'ALBÉDO
# ────────────────────────────────────────────────
def load_albedo_series(
    csv_dir: pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    """
    Charge une série de 12 fichiers CSV d'albédo de surface.

    IN:
        csv_dir (pathlib.Path): Dossier contenant les 12 fichiers CSV.
        pattern (str): Modèle de nom de fichier avec un format pour le mois.

    OUT:
        tuple:
            - np.ndarray: Cube de données (12, lat, lon).
            - np.ndarray: Tableau des latitudes.
            - np.ndarray: Tableau des longitudes.
    """
    if not csv_dir.exists():
        raise FileNotFoundError(f"Dossier d'albédo introuvable: {csv_dir}")
    latitudes, longitudes, cubes = None, None, []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    print("Données d'albédo de surface chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes


CERES_FILE_PATH = (
    pathlib.Path("ressources/albedo")
    / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
)


def load_monthly_cloud_albedo_from_ceres(
    lat_deg: float, lon_deg: float
) -> np.ndarray:
    """
    Extrait l'albédo mensuel des nuages depuis un fichier NetCDF CERES.

    IN:
        lat_deg (float): Latitude du point d'intérêt [degrés].
        lon_deg (float): Longitude du point d'intérêt [degrés].

    OUT:
        np.ndarray: Tableau de 12 valeurs d'albédo mensuel des nuages.
    """
    if not XARRAY_AVAILABLE:
        print("AVERTISSEMENT: xarray non disponible. Retour d'un albédo nul.")
        return np.zeros(12)
    if not CERES_FILE_PATH.exists():
        raise FileNotFoundError(f"Fichier CERES introuvable: {CERES_FILE_PATH}")

    with xr.open_dataset(CERES_FILE_PATH, decode_times=True) as ds:
        ds.load()  # Force le chargement des données en mémoire
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
# DÉTECTION DE CONTINENT
# ────────────────────────────────────────────────
SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)


def create_continent_finder(shapefile_path: pathlib.Path):
    """
    Crée une fonction qui trouve le continent pour un point (lat, lon).

    IN:
        shapefile_path (pathlib.Path): Chemin vers le fichier shapefile des pays.

    OUT:
        function: Une fonction qui prend (lat, lon) et retourne un nom de continent.
    """
    if not GEOPANDAS_AVAILABLE:
        return lambda lat, lon: "Océan"
    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger le shapefile: {e}")
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        valid_world = world[world.geometry.notna()]
        for _, row in valid_world.iterrows():
            if row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point


# Instance globale de la fonction de détection, créée une seule fois.
continent_finder = create_continent_finder(SHAPEFILE_PATH)