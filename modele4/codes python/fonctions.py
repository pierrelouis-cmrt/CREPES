# fonctions.py
import numpy as np
import pandas as pd
import pathlib
import subprocess
import sys
from math import pi
from scipy.ndimage import gaussian_filter1d

# --- Blocs d'installation des dépendances (corrigés) ---
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
    print("xarray non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xarray"])
    import xarray as xr

try:
    from scipy.stats import binned_statistic_2d

    SCIPY_AVAILABLE = True
except ImportError:
    print(
        "AVERTISSEMENT: binned_statistic_2d de Scipy non trouvé. Le griddage RZSM sera désactivé."
    )
    SCIPY_AVAILABLE = False


# Calcul de la déclinaison solaire en fonction du jour de l'année
def declination(day):
    """Calcule la déclinaison solaire (en radians) pour un jour donné de l'année."""
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)


# Calcul du cosinus de l’angle d’incidence du rayonnement solaire
def cos_incidence(lat_rad, day, hour):
    """Calcule le cosinus de l’angle d’incidence du rayonnement solaire."""
    δ = declination(day)
    H = np.radians(15 * (hour - 12))  # Angle horaire en radians
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


# ────────────────────────────────────────────────
# Capacité thermique depuis l'humidité du sol (RZSM)
# ────────────────────────────────────────────────
RHO_W = 1000.0
RHO_BULK = 1300.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    """Calcule la capacité thermique massique (c_p) en kJ/kg/K depuis l'humidité du sol (RZSM)."""
    is_ice = np.isclose(rzsm, 0.9)
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_BULK * (1 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp = CP_SEC + w * (CP_WATER - CP_SEC)
    return np.where(is_ice, CP_ICE, cp)


def load_and_grid_rzsm_data(csv_path: pathlib.Path):
    """Charge et grille les données RZSM depuis un CSV."""
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


# Applique un lissage gaussien à des données mensuelles
def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """Lisse une série de 12 valeurs mensuelles en une série journalière."""
    jours_par_mois = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    valeurs_journalieres_discontinues = np.repeat(
        valeurs_mensuelles, jours_par_mois
    )
    return gaussian_filter1d(
        valeurs_journalieres_discontinues, sigma=sigma, mode="wrap"
    )


# Charge les données d’albédo mensuel depuis des fichiers CSV
def load_albedo_series(
    csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    """Charge une série de 12 fichiers CSV d'albédo de surface."""
    csv_dir = pathlib.Path(csv_dir)
    latitudes, longitudes, cubes = None, None, []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    print("Données d'albédo de surface chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes


# Crée une fonction qui associe un point géographique à un continent
def create_continent_finder(shapefile_path: pathlib.Path):
    """Crée une fonction pour trouver le continent d'un point (lat, lon)."""
    if not GEOPANDAS_AVAILABLE:
        return lambda lat, lon: "Océan"
    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception as e:
        print(f"AVERTISSEMENT: Impossible de charger le shapefile: {e}")
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        # Gérer les géométries nulles
        valid_world = world[world.geometry.notna()]
        for _, row in valid_world.iterrows():
            if row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point


SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)
continent_finder = create_continent_finder(SHAPEFILE_PATH)


# ────────────────────────────────────────────────
# Données d'albédo des nuages depuis CERES
# ────────────────────────────────────────────────
CERES_FILE_PATH = (
    pathlib.Path("ressources/albedo")
    / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
)


def load_monthly_cloud_albedo_from_ceres(
    lat_deg: float, lon_deg: float
) -> np.ndarray:
    """Extrait l'albédo mensuel des nuages depuis un fichier NetCDF CERES."""
    if not XARRAY_AVAILABLE:
        sys.exit("ERREUR: xarray non installé.")
    try:
        # LA CORRECTION EST ICI : .load() force le chargement des données en mémoire
        ds = xr.open_dataset(CERES_FILE_PATH, decode_times=True).load()
    except FileNotFoundError:
        sys.exit(f"ERREUR: Fichier CERES introuvable : {CERES_FILE_PATH}")

    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    toa_sw_all = ds["toa_sw_all_mon"]
    toa_sw_clr = ds["toa_sw_clr_c_mon"]
    solar_in = ds["solar_mon"]

    # Le calcul fonctionne maintenant car les données sont des tableaux numpy
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