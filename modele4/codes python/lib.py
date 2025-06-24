# conventions:
# lat: float (radian), 0 is at equator, -pi/2 is at south pole, and +pi/2 is at north pole
# long: float (radian), 0 is at greenwich meridiant
# t: float (s), 0 is at 00:00 (greenwich time) january 1, 365*24*60*60 is at the end of the year


import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
import fonctions as f 

# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 253.15  # atmosphère radiative (‑20 °C)
dt = 1800.0  # pas de temps : 30 min

# Masse de la couche de surface active thermiquement (kg m-2)
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# ────────────────────────────────────────────────
# DATA – Chaleur latente en fontion du continent 
# ────────────────────────────────────────────────

Q_CONTINENT = {
    "Africa": 46.8,  # Note: Le shapefile utilise les noms anglais
    "Asia": 40.1,
    "South America": 99.8,
    "North America": 35.4,
    "Europe": 36.6,
    "Oceania": 28.4,  # 'Oceania' contient l'Australie
    "Antarctica": 0.0,
    "Océan": 0.0,
}


def P_inc_solar(lat_rad, day, hour, albedo_sol, albedo_nuages):
    """ Puissance radiative du Soleil (W m-2) à la surface terrestre en prenant l'albédo en compte.
    Lat : Radians, 0 est à l'équateur, -pi/2 est au pôle sud et +pi/2 est au pôle nord.
    """
    phi_entrant = constante_solaire * f.cos_incidence(lat_rad, day, hour)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)


# Surface


def P_em_surf_thermal(T: float):
    return sigma * (T**4)


def P_em_surf_conv(lat: float, long: float, t: float):
    return 0

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

def P_em_surf_evap(lat: float, lon: float) -> float:
    """Récupère la valeur de Q (W m-2) pour un point géographique .
    Latitude : en degrés 
    Longitude : en degrés
    """
    continent = f.continent_finder(lat, lon)
    q_val = Q_LATENT_CONTINENT.get(continent, Q_LATENT_CONTINENT["Océan"])
    print(
        f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur : "
        f"{continent} (Q base = {q_val:.2f} W m⁻²)"
    )
    return q_val

# atmosphere
def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    return 0 


def P_em_atm_thermal(T_atm: float):
    """"
    Puissance émise par l'atmosphère (W m-2) en fonction de la température atmosphérique.
    T_atm : Température de l'atmosphère en Kelvin
    """
    return sigma*(T_atm**4) #Loi de Stefan-Boltzmann


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    return 0


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    return 0
