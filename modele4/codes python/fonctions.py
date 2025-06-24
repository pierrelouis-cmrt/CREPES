"""
Ce script regroupe plusieurs fonctions liées à la modélisation climatique simplifiée :

- Calcul de l’angle d’incidence du rayonnement solaire en fonction de la latitude, du jour et de l’heure.
- Estimation de la capacité thermique massique à partir de l’albédo de surface.
- Chargement et lissage de séries temporelles d’albédo à partir de fichiers CSV mensuels.
- Détection du continent correspondant à des coordonnées géographiques via un shapefile (optionnel avec GeoPandas).
- Attribution d’une puissance latente (Q) en fonction du continent détecté.
- Simulation d’une série mensuelle d’albédo des nuages selon la latitude.
- Détermine la capacité thermique massique (c_p) et la masse volumique (rho) d'une surface en se basant sur son albédo comme proxy.


Ce module peut servir de base à une modélisation énergétique de la surface terrestre à l’échelle globale.
"""

# Import des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d

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
# SUPPRIMÉ : La masse surfacique est maintenant calculée dynamiquement.
# MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2
# NOUVEAU : Épaisseur de la couche de sol active pour le calcul de C.
EPAISSEUR_ACTIVE = 0.2  # m (20 cm)


# Bloc try/except pour vérifier si GeoPandas est installé
try:
    import geopandas as gpd
    from shapely.geometry import Point
    GEOPANDAS_AVAILABLE = True  # Indicateur que GeoPandas est disponible
except ImportError:
    GEOPANDAS_AVAILABLE = False  # Sinon, GeoPandas est indisponible

# Calcul de la déclinaison solaire en fonction du jour de l'année
def declination(day):
    """Retourne la déclinaison solaire (rad) pour le jour de l’année (1‑365)."""
    # Le jour est cyclique sur 365 jours
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)

# Calcul du cosinus de l’angle d’incidence du rayonnement solaire
def cos_incidence(lat_rad, day, hour):
    """Cosinus de l’angle d’incidence du rayonnement sur le plan local."""
    δ = declination(day)
    H = np.radians(15 * (hour - 12))  # Angle horaire en radians
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)  # On ne retourne pas de valeur négative

# Références d’albédo pour différents types de surfaces
_REF_ALBEDO = {
    "ice": 0.60,
    "water": 0.10,
    "snow": 0.80,
    "desert": 0.35,
    "forest": 0.20,
    "land": 0.15,
}

# Capacité thermique massique (kJ/kg/K) selon le type de surface
_CAPACITY_BY_TYPE = {
    "ice": 2.0,
    "water": 4.18,
    "snow": 2.0,
    "desert": 0.8,
    "forest": 1.0,
    "land": 1.0,
}

# Associe un albédo à une capacité thermique massique
def capacite_thermique_massique(albedo: float) -> float:
    """Retourne la capacité thermique massique (kJ kg-1 K-1) pour un albedo."""
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[surf]

# Applique un lissage gaussien à des données mensuelles étendues en journalières
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


# Charge les données d’albédo mensuel depuis une série de fichiers CSV
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

# Chemin vers le fichier shapefile de Natural Earth
SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)


# Crée une fonction qui associe un point géographique à un continent
def create_continent_finder(shapefile_path: pathlib.Path):
    if not GEOPANDAS_AVAILABLE:
        return lambda lat, lon: "Océan"
    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception as e:
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        for _, row in world.iterrows():
            if row["geometry"] is not None and row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point

# Création de la fonction de détection de continent au chargement
continent_finder = create_continent_finder(SHAPEFILE_PATH)


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

load_monthly_cloud_albedo_from_ceres(lat_deg=49.0, lon_deg=2.3)


# ────────────────────────────────────────────────
# MODIFIÉ - Capacité thermique et lissage
# ────────────────────────────────────────────────


# MODIFIÉ : La fonction retourne maintenant c_p et rho
def proprietes_thermiques_surface(
    albedo: float,
) -> tuple[float, float]:
    """
    Détermine la capacité thermique massique (c_p) et la masse volumique (rho)
    d'une surface en se basant sur son albédo comme proxy.

    Retourne:
        tuple[float, float]: (capacité massique [kJ kg-1 K-1], densité [kg m-3])
    """
    if np.isnan(albedo):
        return 1.0, 1500.0  # Valeurs par défaut pour la terre

    _REF_ALBEDO = {
        "ice": 0.60,
        "water": 0.10,
        "snow": 0.80,
        "desert": 0.35,
        "forest": 0.20,
        "land": 0.15,
    }
    # Capacité thermique massique en kJ kg-1 K-1
    _CAPACITY_BY_TYPE = {
        "ice": 2.0,
        "water": 4.18,
        "snow": 2.0,
        "desert": 0.8,
        "forest": 1.0,
        "land": 1.0,
    }
    # NOUVEAU : Masse volumique (densité) en kg m-3
    _DENSITY_BY_TYPE = {
        "ice": 917.0,
        "water": 1000.0,
        "snow": 300.0,  # Neige tassée
        "desert": 1600.0,  # Sable sec
        "forest": 1300.0,  # Sol forestier
        "land": 1500.0,  # Sol générique
    }

    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    c_p = _CAPACITY_BY_TYPE[surf]
    rho = _DENSITY_BY_TYPE[surf]
    return c_p, rho
