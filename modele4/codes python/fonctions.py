"""
Ce script regroupe plusieurs fonctions liées à la modélisation climatique simplifiée :

- Calcul de l’angle d’incidence du rayonnement solaire en fonction de la latitude, du jour et de l’heure.
- Estimation de la capacité thermique massique à partir de l’albédo de surface.
- Chargement et lissage de séries temporelles d’albédo à partir de fichiers CSV mensuels.
- Détection du continent correspondant à des coordonnées géographiques via un shapefile (optionnel avec GeoPandas).
- Attribution d’une puissance latente (Q) en fonction du continent détecté.
- Simulation d’une série mensuelle d’albédo des nuages selon la latitude.

Ce module peut servir de base à une modélisation énergétique de la surface terrestre à l’échelle globale.
"""

# Import des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
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
    jours_par_mois = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    valeurs_journalieres_discontinues = np.repeat(valeurs_mensuelles, jours_par_mois)
    return gaussian_filter1d(valeurs_journalieres_discontinues, sigma=sigma, mode="wrap")

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
    """
    Charge un shapefile et retourne une fonction capable de trouver
    le continent pour un point (lat, lon).
    """
    if not GEOPANDAS_AVAILABLE:
        print(
            "AVERTISSEMENT: GeoPandas n'est pas installé. "
            "La détection de continent sera désactivée (Q=0)."
        )
        return lambda lat, lon: "Océan"

    try:
        print(f"Chargement du shapefile depuis : {shapefile_path}")
        world = gpd.read_file(shapefile_path)
        world = world.to_crs(epsg=4326)  # Conversion en système de coordonnées standard (WGS84)
        print("Shapefile chargé avec succès.")
    except Exception as e:
        print(f"ERREUR: Impossible de charger le shapefile : {e}")
        print("La détection de continent sera désactivée (Q=0).")
        return lambda lat, lon: "Océan"

    # Fonction interne pour trouver le continent d’un point
    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)  # Shapely attend (lon, lat)
        for _, row in world.iterrows():
            if row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point

# Création de la fonction de détection de continent au chargement
continent_finder = create_continent_finder(SHAPEFILE_PATH)


# Simule une série mensuelle d'albédo des nuages pour un point donné
def load_monthly_cloud_albedo_mock(lat_deg: float, lon_deg: float):
    print(
        "NOTE : Utilisation de données simulées (mock) pour l'albédo des nuages."
    )
    amplitude = 0.15 * np.sin(np.radians(abs(lat_deg)))
    avg_cloud_albedo = 0.3
    mois = np.arange(12)
    variation_saisonniere = amplitude * np.cos(2 * pi * (mois - 0.5) / 12)
    return avg_cloud_albedo - variation_saisonniere
