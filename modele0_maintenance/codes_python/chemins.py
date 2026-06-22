"""Chemins centralises du projet CREPES.

Tous les scripts passent par ce module pour eviter les chemins relatifs
fragiles presents dans les dossiers d'origine.
"""

from pathlib import Path

# Toutes les racines sont déduites de ce fichier, pas du dossier de lancement.
CODES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODES_DIR.parent
RESSOURCES_DIR = PROJECT_ROOT / "ressources"

# Données physiques lues par les modules de calcul.
ALBEDO_DIR = RESSOURCES_DIR / "albedo"
CERES_FILE = ALBEDO_DIR / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
RZSM_CSV = RESSOURCES_DIR / "capacite_humidite" / "average_rzsm_tout.csv"
MAP_SHAPEFILE = RESSOURCES_DIR / "carte" / "ne_110m_admin_0_countries.shp"
COASTLINE_SHP = RESSOURCES_DIR / "cotes" / "ne_10m_coastline.shp"

# Grilles pré-calculées consommées par les visualisations.
LOWRES_NPY = RESSOURCES_DIR / "grilles" / "grid_lowres_1yr.npy"
HIRES_NPY = RESSOURCES_DIR / "grilles" / "grid_hires_1yr.npy"
LOWRES_FAST_NPY = RESSOURCES_DIR / "grilles" / "grid_lowres_fast.npy"
HIRES_FAST_NPY = RESSOURCES_DIR / "grilles" / "grid_hires_fast.npy"
LOWRES_STABILIZED_NPY = RESSOURCES_DIR / "grilles" / "grid_lowres_stabilized.npy"
HIRES_STABILIZED_NPY = RESSOURCES_DIR / "grilles" / "grid_hires_stabilized.npy"
MONTHLY_TEMPERATURE_DIR = RESSOURCES_DIR / "12_mois"
CACHE_DIR = RESSOURCES_DIR / "caches"


def ensure_cache_dir() -> Path:
    """Cree le dossier de cache local si un module API en a besoin."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR
