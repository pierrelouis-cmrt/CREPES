<<<<<<< HEAD:modele4/lib.py
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


def P_em_surf_evap(lat: float, lon: float) -> float:
    """Récupère la valeur de Q (W m-2) pour un point géographique .
    Latitude : en degrés 
    Longitude : en degrés
    """
    continent = f.continent_finder(lat, lon)
    q_val = 0.0
    for key, value in Q_CONTINENT.items():
        if key in continent:
            q_val = value
            break
    else:
        q_val = Q_CONTINENT["Océan"]

    print(
        f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur le continent : "
        f"{continent} (Q = {q_val} W m⁻²)"
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
=======
#Cette librairie propose une convention pour le nom des puissances surfaciques considérées, mais n'a pas vocation à être réutilisée telle quelle.
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


def P_em_surf_evap(lat: float, lon: float) -> float:
    """Récupère la valeur de Q (W m-2) pour un point géographique .
    Latitude : en degrés 
    Longitude : en degrés
    """
    continent = f.continent_finder(lat, lon)
    q_val = 0.0
    for key, value in Q_CONTINENT.items():
        if key in continent:
            q_val = value
            break
    else:
        q_val = Q_CONTINENT["Océan"]

    print(
        f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur le continent : "
        f"{continent} (Q = {q_val} W m⁻²)"
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
>>>>>>> 4f09c01891f27ae1424a6e4761ccda5af06095c0:modele4/codes python/lib.py
