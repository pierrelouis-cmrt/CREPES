# lib.py
# ==============================================================================
# BIBLIOTHÈQUE DE CONSTANTES ET FORMULES PHYSIQUES
# Rôle : Centralise les constantes physiques universelles et les équations
#        fondamentales du modèle de bilan énergétique (rayonnement, flux...).
# ==============================================================================

import numpy as np
import fonctions as f

# ---------- CONSTANTES PHYSIQUES CENTRALISÉES ----------
constante_solaire = 1361.0  # Irradiance solaire au sommet de l'atmosphère [W m⁻²]
sigma = 5.670374419e-8  # Constante de Stefan-Boltzmann [W m⁻² K⁻⁴]
Tatm = 223.15  # Température effective de l'atmosphère pour le rayonnement IR [-50 °C, en K]
dt = 1800.0  # Pas de temps de la simulation [s], soit 30 minutes
EPAISSEUR_ACTIVE = 0.5  # Épaisseur de la couche de surface active [m]

# ────────────────────────────────────────────────
# FORMULES DE FLUX RADIATIFS
# ────────────────────────────────────────────────
def P_inc_solar(lat_rad, day, hour, albedo_sol, albedo_nuages):
    """
    Calcule la puissance solaire nette absorbée par la surface.

    Prend en compte l'angle d'incidence, l'albédo des nuages (A1) et
    l'albédo du sol (A2).

    IN:
        lat_rad (float): Latitude [radians].
        day (int): Jour de l'année (1-365).
        hour (float): Heure solaire locale (0-24).
        albedo_sol (float): Albédo de la surface (0-1).
        albedo_nuages (float): Albédo des nuages (0-1).

    OUT:
        float: Puissance solaire nette absorbée [W m⁻²].
    """
    phi_entrant = constante_solaire * f.cos_incidence(lat_rad, day, hour)
    # Le flux traverse les nuages (1-A1) puis est réfléchi par le sol (1-A2)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)


def P_em_surf_thermal(T: float):
    """
    Calcule la puissance thermique émise par la surface (loi de Stefan-Boltzmann).

    IN:
        T (float): Température de la surface [K].

    OUT:
        float: Puissance thermique émise [W m⁻²].
    """
    return sigma * (T**4)


def P_em_atm_thermal(T_atm: float):
    """
    Calcule la puissance thermique émise par l'atmosphère vers la surface.

    IN:
        T_atm (float): Température radiative de l'atmosphère [K].

    OUT:
        float: Puissance thermique reçue de l'atmosphère [W m⁻²].
    """
    return sigma * (T_atm**4)


# ────────────────────────────────────────────────
# DONNÉES DE CHALEUR LATENTE (Q) VIA ÉVAPORATION
# ────────────────────────────────────────────────
Delta_hvap = 2453000  # Enthalpie de vaporisation de l'eau [J kg⁻¹]
rho_eau = 1000  # Masse volumique de l'eau [kg m⁻³]
Delta_t_an = 365.25 * 24 * 3600  # Durée d'une année en secondes [s]

# Taux d'évaporation moyens par continent [m an⁻¹], convertis en [m s⁻¹]
evap_Eur = 0.49 / Delta_t_an
evap_Am_Nord = 0.47 / Delta_t_an
evap_Am_sud = 0.94 / Delta_t_an
evap_oceanie = 0.41 / Delta_t_an
evap_Afr = 0.58 / Delta_t_an
evap_Asi = 0.37 / Delta_t_an
evap_ocean = 1.40 / Delta_t_an

# Flux de chaleur latente correspondants [W m⁻²]
Q_LATENT_CONTINENT = {
    "Europe": Delta_hvap * rho_eau * evap_Eur,
    "North America": Delta_hvap * rho_eau * evap_Am_Nord,
    "South America": Delta_hvap * rho_eau * evap_Am_sud,
    "Oceania": Delta_hvap * rho_eau * evap_oceanie,
    "Africa": Delta_hvap * rho_eau * evap_Afr,
    "Asia": Delta_hvap * rho_eau * evap_Asi,
    "Océan": Delta_hvap * rho_eau * evap_ocean,
    "Antarctica": 0.0,  # Pas d'évaporation en Antarctique
}


def P_em_surf_evap(lat: float, lon: float, verbose: bool = False) -> float:
    """
    Récupère la valeur du flux de chaleur latente (Q) pour un point géographique.

    IN:
        lat (float): Latitude [degrés].
        lon (float): Longitude [degrés].
        verbose (bool): Si True, affiche le continent détecté.

    OUT:
        float: Flux de chaleur latente de base [W m⁻²].
    """
    continent = f.continent_finder(lat, lon)
    q_val = Q_LATENT_CONTINENT.get(continent, Q_LATENT_CONTINENT["Océan"])

    if verbose:
        print(
            f"Coordonnées ({lat:.2f}, {lon:.2f}) détectées sur : "
            f"{continent} (Q base = {q_val:.2f} W m⁻²)"
        )

    # Heuristique pour les zones polaires glacées
    if lat > 75:
        return 0.0
    return q_val


# ────────────────────────────────────────────────
# FONCTIONS PRÉVUES POUR UN MODÈLE PLUS COMPLEXE (non utilisées ici)
# ────────────────────────────────────────────────
def P_em_surf_conv(lat: float, long: float, t: float):
    """Flux de chaleur par convection (placeholder)."""
    return 0


def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    """Absorption solaire par l'atmosphère (placeholder)."""
    return 0


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    """Émission thermique de l'atmosphère vers le haut (placeholder)."""
    return 0


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    """Émission thermique de l'atmosphère vers le bas (placeholder)."""
    return 0