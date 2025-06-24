# lib.py

# conventions:
# lat: float (radian), 0 is at equator, -pi/2 is at south pole, and +pi/2 is at north pole
# long: float (radian), 0 is at greenwich meridiant
# t: float (s), 0 is at 00:00 (greenwich time) january 1, 365*24*60*60 is at the end of the year

import numpy as np
import fonctions as f

# ---------- constantes physiques centralisées ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (-50 °C)
dt = 1800.0  # pas de temps : 30 min
EPAISSEUR_ACTIVE = 0.5  # m

# ────────────────────────────────────────────────
# Données de chaleur latente (Q) via évaporation
# ────────────────────────────────────────────────

Delta_hvap = 2453000  # J/kg
rho_eau = 1000  # kg/m^3
Delta_t_an = 365.25 * 24 * 3600  # s/an

# Taux d'évaporation en m/an, convertis en m/s
evap_Eur = (0.49 / Delta_t_an)
evap_Am_Nord = (0.47 / Delta_t_an)
evap_Am_sud = (0.94 / Delta_t_an)
evap_oceanie = (0.41 / Delta_t_an)
evap_Afr = (0.58 / Delta_t_an)
evap_Asi = (0.37 / Delta_t_an)
evap_ocean = (1.40 / Delta_t_an)

# Flux de chaleur latente en W/m^2 (J/s/m^2)
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


def P_inc_solar(lat_rad, day, hour, albedo_sol, albedo_nuages):
    """Puissance radiative du Soleil (W m-2) à la surface terrestre en prenant l'albédo en compte."""
    phi_entrant = constante_solaire * f.cos_incidence(lat_rad, day, hour)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)


def P_em_surf_thermal(T: float):
    """Puissance thermique émise par la surface (W m-2)."""
    return sigma * (T**4)


def P_em_surf_evap(lat: float, lon: float) -> float:
    """Récupère la valeur de Q (W m-2) pour un point géographique."""
    continent = f.continent_finder(lat, lon)
    q_val = Q_LATENT_CONTINENT.get(continent, Q_LATENT_CONTINENT["Océan"])
    print(
        f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur : "
        f"{continent} (Q base = {q_val:.2f} W m⁻²)"
    )
    # Correction pour l'Arctique qui est un océan mais sans évaporation
    if lat > 75:
        return 0.0
    return q_val


def P_em_atm_thermal(T_atm: float):
    """Puissance thermique émise par l'atmosphère (W m-2)."""
    return sigma * (T_atm**4)


# Fonctions prévues pour un modèle plus complexe
def P_em_surf_conv(lat: float, long: float, t: float):
    return 0


def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    return 0


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    return 0


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    return 0