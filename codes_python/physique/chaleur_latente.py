"""Chaleur latente / evapotranspiration.

Implementation active retenue: fonction Carcajous par continent. Cote
Chevreaux, le modele 6 contient seulement `P_em_surf_evap(...) = 86`; le PDF
d'evapotranspiration est donc conserve comme documentation, pas comme code
detaille a brancher.
"""

from __future__ import annotations

from chemins import MAP_SHAPEFILE

try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

Delta_hvap = 2453000
rho_eau = 1000
Delta_t_an = 365.25 * 24 * 3600

evap_Eur = 0.49 / Delta_t_an
evap_Am_Nord = 0.47 / Delta_t_an
evap_Am_sud = 0.94 / Delta_t_an
evap_oceanie = 0.41 / Delta_t_an
evap_Afr = 0.58 / Delta_t_an
evap_Asi = 0.37 / Delta_t_an
evap_ocean = 1.40 / Delta_t_an

Q_LATENT_CONTINENT = {
    "Europe": Delta_hvap * rho_eau * evap_Eur,
    "North America": Delta_hvap * rho_eau * evap_Am_Nord,
    "South America": Delta_hvap * rho_eau * evap_Am_sud,
    "Oceania": Delta_hvap * rho_eau * evap_oceanie,
    "Africa": Delta_hvap * rho_eau * evap_Afr,
    "Asia": Delta_hvap * rho_eau * evap_Asi,
    "Océan": Delta_hvap * rho_eau * evap_ocean,
    "Antarctica": 0.0,
}


def create_continent_finder(shapefile_path=MAP_SHAPEFILE):
    """Cree une fonction qui associe un point geographique a un continent."""
    if not GEOPANDAS_AVAILABLE:
        return lambda lat, lon: "Océan"
    try:
        world = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception:
        return lambda lat, lon: "Océan"

    def find_continent_for_point(lat: float, lon: float) -> str:
        point = Point(lon, lat)
        valid_world = world[world.geometry.notna()]
        for _, row in valid_world.iterrows():
            if row["geometry"].contains(point):
                return row["CONTINENT"]
        return "Océan"

    return find_continent_for_point


continent_finder = create_continent_finder(MAP_SHAPEFILE)


def P_em_surf_evap(lat: float, lon: float, verbose: bool = False) -> float:
    """Flux latent de base pour un point geographique, en W m-2."""
    continent = continent_finder(lat, lon)
    q_val = Q_LATENT_CONTINENT.get(continent, Q_LATENT_CONTINENT["Océan"])
    if verbose:
        print(
            f"Coordonnees ({lat:.2f}, {lon:.2f}) detectees sur : "
            f"{continent} (Q base = {q_val:.2f} W m-2)"
        )
    if lat > 75:
        return 0.0
    return q_val
