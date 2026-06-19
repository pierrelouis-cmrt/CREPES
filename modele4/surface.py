"""Briques de surface du modele 4.

Les constantes et formules viennent du modele 0 lorsque le modele 4 a besoin
d'un terme non radiatif : capacite thermique, chaleur latente et convection.
Le module reste autonome pour eviter les imports fragiles du dossier de
maintenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np


RHO_W = 1000.0
RHO_BULK = 2600.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09
EPAISSEUR_ACTIVE_M = 0.5

DELTA_HVAP = 2_453_000.0
RHO_EAU = 1000.0
DELTA_T_AN = 365.25 * 24.0 * 3600.0

Q_LATENT_CONTINENT_W_M2 = {
    "Europe": DELTA_HVAP * RHO_EAU * (0.49 / DELTA_T_AN),
    "North America": DELTA_HVAP * RHO_EAU * (0.47 / DELTA_T_AN),
    "South America": DELTA_HVAP * RHO_EAU * (0.94 / DELTA_T_AN),
    "Oceania": DELTA_HVAP * RHO_EAU * (0.41 / DELTA_T_AN),
    "Africa": DELTA_HVAP * RHO_EAU * (0.58 / DELTA_T_AN),
    "Asia": DELTA_HVAP * RHO_EAU * (0.37 / DELTA_T_AN),
    "Ocean": DELTA_HVAP * RHO_EAU * (1.40 / DELTA_T_AN),
    "Antarctica": 0.0,
}

Q_LATENT_TERRE_MOYEN_W_M2 = sum(
    Q_LATENT_CONTINENT_W_M2[nom]
    for nom in ("Europe", "North America", "South America", "Oceania", "Africa", "Asia")
) / 6.0

TEMPERATURE_AIR_DEFAUT_K = 288.0
VENT_DEFAUT_M_S = 2.5


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

    rzsm = max(1e-6, min(0.999, float(rzsm)))
    if abs(rzsm - 0.9) < 1e-9:
        cp_kj_kg_k = CP_ICE
    else:
        w = (RHO_W * rzsm) / (RHO_BULK * (1.0 - rzsm) + RHO_W * rzsm)
        cp_kj_kg_k = CP_SEC + w * (CP_WATER - CP_SEC)
    return cp_kj_kg_k * 1000.0 * RHO_BULK * EPAISSEUR_ACTIVE_M


def charger_grille_rzsm(csv_path):
    """Charge le CSV RZSM du modele 0 dans une grille reguliere."""

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV RZSM introuvable: {csv_path}")

    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    latitudes = np.unique(table["lat"])
    longitudes = np.unique(table["lon"])
    valeurs = table["RZSM"]
    if len(latitudes) * len(longitudes) == len(valeurs):
        rzsm = valeurs.reshape((len(latitudes), len(longitudes)))
    else:
        rzsm = np.full((len(latitudes), len(longitudes)), np.nan)
        index_lat = {valeur: indice for indice, valeur in enumerate(latitudes)}
        index_lon = {valeur: indice for indice, valeur in enumerate(longitudes)}
        for lat, lon, valeur in zip(table["lat"], table["lon"], valeurs):
            rzsm[index_lat[lat], index_lon[lon]] = valeur
    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "rzsm": rzsm,
        "source": str(csv_path),
    }


def rzsm_plus_proche(grille_rzsm, lat_deg, lon_deg):
    if grille_rzsm is None:
        return None
    lon_deg = ((float(lon_deg) + 180.0) % 360.0) - 180.0
    latitudes = grille_rzsm["latitudes"]
    longitudes = grille_rzsm["longitudes"]
    indice_lat = int(np.nanargmin(np.abs(latitudes - float(lat_deg))))
    indice_lon = int(np.nanargmin(np.abs(longitudes - lon_deg)))
    valeur = float(grille_rzsm["rzsm"][indice_lat, indice_lon])
    if not isfinite(valeur):
        return None
    return valeur


def capacite_surface(surface, rzsm=None):
    """Retourne C en J m-2 K-1 pour une cellule du modele 4.

    Le modele 0 utilisait RZSM quand il etait disponible. Si `rzsm` est fourni,
    la partie continentale reprend cette capacite. Sinon la V1 utilise les
    constantes du modele 0 avec un melange simple terre/ocean/glace depuis les
    fractions deja presentes dans le paquet.
    """

    land = fraction(surface.get("land_fraction"), defaut=1.0)
    snow_ice = fraction(surface.get("snow_ice_fraction"), defaut=0.0)

    if rzsm is None:
        capacite_land = CP_SEC * 1000.0 * RHO_BULK * EPAISSEUR_ACTIVE_M
    else:
        capacite_land = capacite_depuis_rzsm(rzsm)
    capacite_ocean = CP_WATER * 1000.0 * RHO_W * EPAISSEUR_ACTIVE_M
    capacite_glace = CP_ICE * 1000.0 * RHO_BULK * EPAISSEUR_ACTIVE_M

    capacite_sans_glace = land * capacite_land + (1.0 - land) * capacite_ocean
    return snow_ice * capacite_glace + (1.0 - snow_ice) * capacite_sans_glace


def source_capacite_surface(rzsm_csv=None):
    if rzsm_csv is not None:
        return f"modele0 RZSM {rzsm_csv} + melange land/ocean/snow_ice"
    return "modele0 constantes capacite + melange land/ocean/snow_ice du paquet modele3"


def flux_latent_moyen(surface, facteur=1.0):
    """Flux latent positif quand la surface perd de l'energie."""

    latitude = _float_fini(surface.get("latitude_deg"), 0.0)
    if latitude is not None and latitude > 75.0:
        return 0.0

    land = fraction(surface.get("land_fraction"), defaut=1.0)
    snow_ice = fraction(surface.get("snow_ice_fraction"), defaut=0.0)
    q_terre = Q_LATENT_TERRE_MOYEN_W_M2
    q_ocean = Q_LATENT_CONTINENT_W_M2["Ocean"]
    q_base = land * q_terre + (1.0 - land) * q_ocean
    return max(0.0, float(facteur)) * q_base * (1.0 - snow_ice)


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
