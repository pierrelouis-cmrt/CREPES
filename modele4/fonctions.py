#Cette librairie propose une convention pour le nom des puissances surfaciques considérées, mais n'a pas vocation à être réutilisée telle quelle.
# conventions:
# lat: float (radian), 0 is at equator, -pi/2 is at south pole, and +pi/2 is at north pole
# long: float (radian), 0 is at greenwich meridiant
# t: float (s), 0 is at 00:00 (greenwich time) january 1, 365*24*60*60 is at the end of the year, (maybe use 365.25? no idea what is best, or maybe use UTC ?)


import numpy as np

P0 = 1360  # W/m² – zenith irradiance at the top of the atmosphere
PHI = 0.409  # precession angle rad  (23.45 deg)
SIGMA = 5.67e-8  # W/m²K⁴ – Stefan-Boltzmann constant


def P_inc_solar(lat: float, long: float, t: float):
    """
    Solar irradiance
    """
    day_decimal = t / 86400

    t_hours = (t % 86400) / 3600.0

    delta = PHI * np.sin(2 * np.pi * (day_decimal - 81.0) / 365)

    H = np.pi / 12 * (t_hours - 12.0)

    sin_h = np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(H)

    return np.where(sin_h > 0.0, P0 * sin_h, 0.0)


# Surface
def P_abs_surf_solar(lat: float, long: float, t: float, Pinc: float):
    AbsSurf = 0.62
    return AbsSurf * Pinc


def P_em_surf_thermal(lat: float, long: float, t: float, T: float):
    return SIGMA * (T**4)


def P_em_surf_conv(lat: float, long: float, t: float):
    return 18


def P_em_surf_evap(lat: float, long: float, t: float):
    return 86


# atmosphere
def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    AbsAtmo = 0.22
    return AbsAtmo * Pinc


def P_abs_atm_thermal(lat: float, long: float, t: float, T: float):
    return 358


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    return 170


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    return 340
