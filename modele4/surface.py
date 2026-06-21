"""Briques de surface du modele 4.

Les constantes et formules viennent du modele 0 lorsque le modele 4 a besoin
d'un terme non radiatif : capacite thermique, chaleur latente et convection.
Le module reste autonome pour eviter les imports fragiles du dossier de
maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np


RHO_W = 1000.0
RHO_ICE = 917.0
RHO_BULK = 2600.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09
EPAISSEUR_ACTIVE_M = 0.5
EPAISSEUR_OCEAN_M = 1.0
EPAISSEUR_GLACE_NEIGE_M = 1.0

DELTA_HVAP = 2_453_000.0
RHO_EAU = 1000.0
DELTA_T_AN = 365.25 * 24.0 * 3600.0
RACINE_PROJET = Path(__file__).resolve().parents[1]
RZSM_MODELE0_DEFAUT = (
    RACINE_PROJET
    / "modele0_maintenance"
    / "ressources"
    / "capacite_humidite"
    / "average_rzsm_tout.csv"
)
SHAPEFILE_CONTINENTS_MODELE0 = (
    RACINE_PROJET
    / "modele0_maintenance"
    / "ressources"
    / "carte"
    / "ne_110m_admin_0_countries.shp"
)

Q_LATENT_CONTINENT_W_M2 = {
    "Europe": DELTA_HVAP * RHO_EAU * (0.49 / DELTA_T_AN),
    "North America": DELTA_HVAP * RHO_EAU * (0.47 / DELTA_T_AN),
    "South America": DELTA_HVAP * RHO_EAU * (0.94 / DELTA_T_AN),
    "Oceania": DELTA_HVAP * RHO_EAU * (0.41 / DELTA_T_AN),
    "Africa": DELTA_HVAP * RHO_EAU * (0.58 / DELTA_T_AN),
    "Asia": DELTA_HVAP * RHO_EAU * (0.37 / DELTA_T_AN),
    "Océan": DELTA_HVAP * RHO_EAU * (1.40 / DELTA_T_AN),
    "Ocean": DELTA_HVAP * RHO_EAU * (1.40 / DELTA_T_AN),
    "Antarctica": 0.0,
}

TEMPERATURE_AIR_DEFAUT_K = 288.0
VENT_DEFAUT_M_S = 2.5

try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_DISPONIBLE = True
except ImportError:  # pragma: no cover - dependance optionnelle
    gpd = None
    Point = None
    GEOPANDAS_DISPONIBLE = False

_CACHE_RZSM = {}
_CACHE_DETECTEURS_CONTINENT = {}


@dataclass(frozen=True)
class ConfigurationSurface:
    """Options des termes de surface herites du modele 0."""

    facteur_latent: float = 1.0
    mode_convection: str = "toutes"
    vent_m_s: float = VENT_DEFAUT_M_S
    temperature_air_defaut_k: float = TEMPERATURE_AIR_DEFAUT_K


def _float_fini(valeur, defaut=None):
    try:
        resultat = float(valeur)
    except (TypeError, ValueError):
        return defaut
    if not isfinite(resultat):
        return defaut
    return resultat


def fraction(valeur, defaut=0.0):
    valeur = _float_fini(valeur, defaut)
    if valeur is None:
        valeur = defaut
    return max(0.0, min(1.0, valeur))


def capacite_depuis_rzsm(rzsm):
    """Capacite surfacique Carcajous issue de l'humidite RZSM."""

    rzsm_array = np.asarray(rzsm, dtype=float)
    glace = np.isclose(rzsm_array, 0.9)
    rzsm_clipped = np.clip(rzsm_array, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_BULK * (1.0 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp_kj_kg_k = CP_SEC + w * (CP_WATER - CP_SEC)
    cp_kj_kg_k = np.where(glace, CP_ICE, cp_kj_kg_k)
    capacite = cp_kj_kg_k * 1000.0 * RHO_BULK * EPAISSEUR_ACTIVE_M
    if capacite.ndim == 0:
        return float(capacite)
    return capacite


def capacite_sol_sec():
    """Capacite minimale de sol sec en J m-2 K-1."""

    return CP_SEC * 1000.0 * RHO_BULK * EPAISSEUR_ACTIVE_M


def capacite_ocean_surface():
    """Capacite simple d'une couche oceanique active de surface."""

    return CP_WATER * 1000.0 * RHO_W * EPAISSEUR_OCEAN_M


def capacite_glace_neige_surface():
    """Capacite simple d'une couche active de glace ou neige compacte."""

    return CP_ICE * 1000.0 * RHO_ICE * EPAISSEUR_GLACE_NEIGE_M


def _moyenne_par_bins_modele0(latitudes, longitudes, valeurs):
    """Reproduit le regrillage 1 degre du modele 0 sans dependance SciPy."""

    grid_res = 1.0
    lon_bins = np.arange(-180.0, 180.0 + grid_res, grid_res)
    lat_bins = np.arange(-90.0, 90.0 + grid_res, grid_res)
    sommes = np.zeros((len(lat_bins) - 1, len(lon_bins) - 1), dtype=np.float64)
    comptes = np.zeros_like(sommes)

    masque = np.isfinite(latitudes) & np.isfinite(longitudes) & np.isfinite(valeurs)
    latitudes = latitudes[masque]
    longitudes = ((longitudes[masque] + 180.0) % 360.0) - 180.0
    valeurs = valeurs[masque]

    indices_lon = np.searchsorted(lon_bins, longitudes, side="right") - 1
    indices_lat = np.searchsorted(lat_bins, latitudes, side="right") - 1
    indices_lon = np.where(np.isclose(longitudes, lon_bins[-1]), len(lon_bins) - 2, indices_lon)
    indices_lat = np.where(np.isclose(latitudes, lat_bins[-1]), len(lat_bins) - 2, indices_lat)

    dans_grille = (
        (indices_lat >= 0)
        & (indices_lat < sommes.shape[0])
        & (indices_lon >= 0)
        & (indices_lon < sommes.shape[1])
    )
    np.add.at(sommes, (indices_lat[dans_grille], indices_lon[dans_grille]), valeurs[dans_grille])
    np.add.at(comptes, (indices_lat[dans_grille], indices_lon[dans_grille]), 1.0)

    grille = np.full_like(sommes, np.nan)
    np.divide(sommes, comptes, out=grille, where=comptes > 0.0)
    return grille, lat_bins, lon_bins


def charger_grille_rzsm(csv_path):
    """Charge et regrille le CSV RZSM comme le faisait le modele 0."""

    if csv_path is None:
        return None
    csv_path = Path(csv_path)
    cle_cache = str(csv_path.resolve()) if csv_path.exists() else str(csv_path)
    if cle_cache in _CACHE_RZSM:
        return _CACHE_RZSM[cle_cache]
    if not csv_path.exists():
        return None

    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    table = np.atleast_1d(table)
    rzsm, latitudes, longitudes = _moyenne_par_bins_modele0(
        np.asarray(table["lat"], dtype=np.float64),
        np.asarray(table["lon"], dtype=np.float64),
        np.asarray(table["RZSM"], dtype=np.float64),
    )
    grille = {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "rzsm": rzsm,
        "source": str(csv_path),
    }
    _CACHE_RZSM[cle_cache] = grille
    return grille


def rzsm_plus_proche(grille_rzsm, lat_deg, lon_deg):
    if grille_rzsm is None:
        return None
    lon_deg = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    latitudes = grille_rzsm["latitudes"]
    longitudes = grille_rzsm["longitudes"]
    indice_lat = min(
        int(np.nanargmin(np.abs(latitudes - float(lat_deg)))),
        grille_rzsm["rzsm"].shape[0] - 1,
    )
    indice_lon = min(
        int(np.nanargmin(np.abs(longitudes - lon_deg))),
        grille_rzsm["rzsm"].shape[1] - 1,
    )
    valeur = float(grille_rzsm["rzsm"][indice_lat, indice_lon])
    if not isfinite(valeur):
        return None
    return valeur


def capacite_surface(surface, rzsm=None):
    """Retourne C en J m-2 K-1 pour une cellule du modele 4.

    La terre non enneigee reprend la capacite RZSM du modele 0 quand elle est
    disponible. Si RZSM manque localement, on garde le fallback de sol sec du
    modele 0. Les fractions ocean et glace/neige du paquet modele 3 recoivent
    une inertie propre afin d'eviter qu'elles soient traitees comme du sol sec
    ou comme une humidite de sol.
    """

    fraction_terre = fraction(surface.get("land_fraction"), defaut=1.0)
    fraction_glace_neige = fraction(surface.get("snow_ice_fraction"), defaut=0.0)
    fraction_non_glace = 1.0 - fraction_glace_neige
    fraction_ocean = 1.0 - fraction_terre

    rzsm_valide = _float_fini(rzsm)
    if rzsm_valide is None:
        capacite_terre = capacite_sol_sec()
    else:
        capacite_terre = capacite_depuis_rzsm(rzsm_valide)

    return (
        fraction_glace_neige * capacite_glace_neige_surface()
        + fraction_non_glace * fraction_terre * capacite_terre
        + fraction_non_glace * fraction_ocean * capacite_ocean_surface()
    )


def source_capacite_surface(rzsm_csv=None):
    if rzsm_csv is not None and Path(rzsm_csv).exists():
        return (
            f"modele0 RZSM {rzsm_csv} pour terres non glacees; "
            "ocean 1 m eau; glace/neige 1 m glace; sol sec si RZSM manque"
        )
    return (
        "surface par fractions; sol sec pour terres sans RZSM; "
        "ocean/glace/neige explicites"
    )


def creer_detecteur_continent(shapefile_path=SHAPEFILE_CONTINENTS_MODELE0):
    """Cree le detecteur continent/ocean du modele 0."""

    shapefile_path = Path(shapefile_path)
    if not GEOPANDAS_DISPONIBLE or not shapefile_path.exists():
        return None
    try:
        monde = gpd.read_file(shapefile_path).to_crs(epsg=4326)
    except Exception:
        return None

    monde_valide = monde[monde.geometry.notna()]

    def detecter(lat, lon):
        point = Point(lon, lat)
        for _, ligne in monde_valide.iterrows():
            if ligne["geometry"].contains(point):
                return ligne["CONTINENT"]
        return "Océan"

    return detecter


def _detecteur_continent(shapefile_path=SHAPEFILE_CONTINENTS_MODELE0):
    cle = str(Path(shapefile_path).resolve()) if Path(shapefile_path).exists() else str(shapefile_path)
    if cle not in _CACHE_DETECTEURS_CONTINENT:
        _CACHE_DETECTEURS_CONTINENT[cle] = creer_detecteur_continent(shapefile_path)
    return _CACHE_DETECTEURS_CONTINENT[cle]


@lru_cache(maxsize=8192)
def _continent_point_cache(shapefile_key, lat_deg, lon_deg):
    detecteur = _detecteur_continent(Path(shapefile_key))
    if detecteur is None:
        return "Océan"
    return detecteur(lat_deg, lon_deg)


def continent_point(lat_deg, lon_deg, shapefile_path=SHAPEFILE_CONTINENTS_MODELE0):
    lon_deg = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    lat_deg = float(lat_deg)
    chemin = Path(shapefile_path)
    cle = str(chemin.resolve()) if chemin.exists() else str(chemin)
    return _continent_point_cache(cle, round(lat_deg, 6), round(lon_deg, 6))


def source_flux_latent(shapefile_path=SHAPEFILE_CONTINENTS_MODELE0):
    if GEOPANDAS_DISPONIBLE and Path(shapefile_path).exists():
        return f"modele0 continents {shapefile_path}"
    return "modele0 fallback ocean; detecteur continent indisponible"


def flux_latent_moyen(
    surface,
    facteur=1.0,
    detecteur_continent=None,
    shapefile_path=SHAPEFILE_CONTINENTS_MODELE0,
):
    """Flux latent annuel moyen du modele 0, positif en perte de surface."""

    facteur = max(0.0, _float_fini(facteur, 0.0))
    if facteur == 0.0:
        return 0.0
    latitude = _float_fini(surface.get("latitude_deg"), 0.0)
    if latitude is not None and latitude > 75.0:
        return 0.0
    longitude = _float_fini(surface.get("longitude_deg"), 0.0)
    if detecteur_continent is None:
        continent = continent_point(latitude, longitude, shapefile_path=shapefile_path)
    else:
        continent = detecteur_continent(latitude, longitude)

    q_base = Q_LATENT_CONTINENT_W_M2.get(continent, Q_LATENT_CONTINENT_W_M2["Océan"])
    return facteur * q_base


def coefficient_convection_forcee(vent_m_s):
    """Coefficient h de convection forcee repris du modele 0 Chevreaux."""

    rho = 1.2
    mu = 1.8e-5
    longueur = 1.0
    prandtl = 0.71
    lambda_air = 0.026

    re = rho * max(float(vent_m_s), 0.0) * longueur / mu
    if re < 5e5:
        coeff, power_re, power_pr = 0.664, 0.5, 1.0 / 3.0
    else:
        coeff, power_re, power_pr = 0.037, 0.8, 1.0 / 3.0
    nusselt = coeff * re**power_re * prandtl**power_pr
    return nusselt * lambda_air / longueur


def coefficient_convection_naturelle(temperature_surface_k, temperature_air_k):
    """Coefficient h de convection naturelle repris du modele 0."""

    lam = 0.026
    longueur = 0.05
    exposant = 1.0 / 4.0
    coeff = 0.54 if temperature_surface_k >= temperature_air_k else 0.27

    g = 9.81
    nu = 1.5e-5
    alpha = 2e-5
    beta = 1.0 / max(temperature_air_k, 1.0)
    grashof = g * beta * (temperature_surface_k - temperature_air_k) * longueur**3 / nu**2
    prandtl = nu / alpha
    rayleigh = grashof * prandtl
    nusselt = coeff * abs(rayleigh) ** exposant
    return nusselt * lam / longueur


def flux_convection_forcee(temperature_surface_k, temperature_air_k, vent_m_s):
    """Flux convectif force, positif si la surface perd de la chaleur."""

    return coefficient_convection_forcee(vent_m_s) * (temperature_surface_k - temperature_air_k)


def flux_convection_naturelle(temperature_surface_k, temperature_air_k):
    """Flux convectif naturel, positif si la surface perd de la chaleur."""

    return coefficient_convection_naturelle(temperature_surface_k, temperature_air_k) * (
        temperature_surface_k - temperature_air_k
    )


def flux_convection(temperature_surface_k, temperature_air_k, config):
    """Flux convectif total selon le mode choisi."""

    mode = config.mode_convection
    if mode == "aucune":
        return 0.0
    if mode not in {"forcee", "naturelle", "toutes"}:
        raise ValueError(f"Mode de convection inconnu: {mode}")

    flux = 0.0
    if mode in {"forcee", "toutes"}:
        flux += flux_convection_forcee(
            temperature_surface_k,
            temperature_air_k,
            config.vent_m_s,
        )
    if mode in {"naturelle", "toutes"}:
        flux += flux_convection_naturelle(temperature_surface_k, temperature_air_k)
    return flux


def derivee_flux_convection(temperature_surface_k, temperature_air_k, config):
    """Derivee numerique de Q_conv par rapport a T_surface."""

    if config.mode_convection == "aucune":
        return 0.0
    eps = max(1e-4, abs(temperature_surface_k) * 1e-6)
    q_plus = flux_convection(temperature_surface_k + eps, temperature_air_k, config)
    q_moins = flux_convection(temperature_surface_k - eps, temperature_air_k, config)
    return (q_plus - q_moins) / (2.0 * eps)


def temperature_air(surface, config):
    valeur = _float_fini(surface.get("temperature_2m_k"))
    if valeur is not None:
        return valeur
    return config.temperature_air_defaut_k
