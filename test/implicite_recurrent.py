#!/usr/bin/env python3
"""
Simulation 0‑D de la température de surface à Paris avec la méthode récurrente
(Euler explicite) et Δt = 60 s.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import datetime as dt

from fonction import solve_ode_recurrent

# ---------- constantes physiques ----------
S0    = 1361.0                     # constante solaire (W m⁻²)
sigma = 5.670374419e-8             # Stefan‑Boltzmann (W m⁻² K⁻⁴)
alpha = 0.25                       # albédo de surface (—)
Tatm  = 253.15                     # T radiative atmosphérique (K)
C     = 8.36e5                     # capacité surfacique (J m⁻² K⁻¹)

# ---------- localisation ----------
lat_deg = 49.0                     # latitude Paris (°N)
lon_deg = 2.0                      # longitude Paris (°E)

# ---------- discrétisation temporelle ----------
dt_sec = 60.0                      # pas de temps (s)

# ---------- géométrie solaire ----------

def update_sun_vector(month: int, sun_vector: np.ndarray) -> np.ndarray:
    """Applique l’inclinaison axiale pour le mois donné au vecteur Soleil→Terre."""
    obliquité = np.radians(23.44)
    angle_saisonnier = obliquité * np.cos(2 * pi * (month - 1) / 12)
    R_saison = np.array([
        [np.cos(angle_saisonnier), 0, np.sin(angle_saisonnier)],
        [0, 1, 0],
        [-np.sin(angle_saisonnier), 0, np.cos(angle_saisonnier)]
    ])
    return R_saison @ sun_vector


def puissance_recue_point(lat: float, lon: float, month: int, hour: float,
                          albedo: float = alpha) -> float:
    """Flux solaire net absorbé par le point de surface (W m⁻²)."""
    theta = np.radians(90 - lat)
    phi   = np.radians(lon % 360)

    # Soleil situé à +∞ sur l’axe +x
    sun_vec = np.array([1.0, 0.0, 0.0])
    sun_vec = update_sun_vector(month, sun_vec)

    # rotation diurne
    omega = 2 * pi * hour / 24.0
    R_heure = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega),  np.cos(omega), 0],
        [0, 0, 1]
    ])
    sun_vec = R_heure @ sun_vec

    # normale au point
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    cos_inc = max(np.dot(normal, sun_vec), 0.0)
    return S0 * cos_inc * (1 - albedo)


def day_to_month(day_of_year: int) -> int:
    date_ref = dt.date(2025, 1, 1) + dt.timedelta(days=day_of_year - 1)
    return date_ref.month


def phi_net(day_of_year: int, hour: float) -> float:
    return puissance_recue_point(lat_deg, lon_deg, day_to_month(day_of_year),
                                 hour, alpha)

# ---------- RHS de l’EDO ----------

def rhs_temperature(t: float, T: float) -> float:
    day_of_year = int(t // 86400) + 1
    hour = (t % 86400) / 3600.0
    phinet = phi_net(day_of_year, hour)
    return (phinet + sigma * Tatm**4 - sigma * T**4) / C

# ---------- fonctions utilitaires ----------

def run_simulation(days: int, T0: float = 288.0):
    """Lance la simulation sur *days* jours."""
    t_end = days * 86400.0
    return solve_ode_recurrent(rhs_temperature, 0.0, t_end, dt_sec, T0)


def plot_temperature(t: np.ndarray, T: np.ndarray, titre: str):
    plt.figure(figsize=(10, 4))
    plt.plot(t / 86400.0, T - 273.15, lw=1.2)
    plt.xlabel("Jour")
    plt.ylabel("Température surface (°C)")
    plt.title(titre)
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()

# ---------- programme principal ----------

if __name__ == "__main__":
    for nb_days in (1, 365):
        t, Tsurf = run_simulation(nb_days)
        plot_temperature(t, Tsurf,
                         f"Euler explicite – {nb_days} jour{'s' if nb_days > 1 else ''}")
