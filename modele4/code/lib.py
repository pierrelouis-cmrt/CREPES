#Cette librairie propose une convention pour le nom des puissances surfaciques considérées, mais n'a pas vocation à être réutilisée telle quelle.
# conventions:
# lat: float (radian), 0 is at equator, -pi/2 is at south pole, and +pi/2 is at north pole
# long: float (radian), 0 is at greenwich meridiant
# t: float (s), 0 is at 00:00 (greenwich time) january 1, 365*24*60*60 is at the end of the year, (maybe use 365.25? no idea what is best, or maybe use UTC ?)


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



def P_inc_solar(lat_rad, day, hour, albedo):
    """
    Solar irradiance
    """
    return constante_solaire * f.cos_incidence(lat_rad, day, hour) * (1 - albedo)

# Surface
def P_abs_surf_solar(lat: float, long: float, t: float, Pinc: float):
    return 0


def P_em_surf_thermal(T: float):
    return sigma * (T**4)


def P_em_surf_conv(lat: float, long: float, t: float):
    return 0


def P_em_surf_evap(lat: float, long: float, t: float):
    return 0


# atmosphere
def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    return 0 


def P_abs_atm_thermal(T_atm: float):
    return sigma*(T_atm**4)


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    return 0


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    return 0
