#!/usr/bin/env python3
"""
Simulation 0‑D de la température de surface à Paris avec schéma récurrent
(Euler explicite, Δt = 60 s).
Le flux solaire est calculé **exactement** avec le script fourni.
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from fonction import solve_ode_recurrent

# ---------- constantes physiques ----------
constante_solaire = 1361          # W/m² (script fourni)
sigma = 5.67e-8                   # W/m²·K⁴ (script fourni)
alpha = 0.25                       # albédo de surface (on passera alpha à la fonction)
Tatm  = 253.15                     # température radiative atmosphère (K)
C     = 8.36e5                     # capacité surfacique (J m⁻² K⁻¹)

# ---------- localisation ----------
lat_deg = 49.0                     # Paris
lon_deg = 2.0

# ---------- discrétisation temporelle ----------
dt_sec = 60.0                      # pas de temps (s)

# ---------- script flux solaire copié tel quel ----------

def update_sun_vector(mois, sun_vector):
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    return np.dot(rotation_matrix_saison, sun_vector)


def puissance_recue_point(lat_deg, lon_deg, mois, time, albedo=0.3):
    theta = np.radians(90 - lat_deg)  # colatitude
    phi = np.radians(lon_deg % 360)
    sun_vector = np.array([1, 0, 0])

    # Inclinaison saisonnière
    sun_vector = update_sun_vector(mois, sun_vector)

    # Rotation diurne
    angle_rotation = (time / 24) * 2 * np.pi
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation),  np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector = np.dot(rotation_matrix, sun_vector)

    # Vecteur normal du point
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    cos_incidence = max(np.dot(normal, sun_vector), 0)
    puissance_recue = constante_solaire * cos_incidence * (1 - albedo)
    temperature = (puissance_recue / sigma) ** 0.25
    return temperature, puissance_recue

# ---------- utilitaire calendrier ----------

def day_to_month(day_of_year: int) -> int:
    """Convertit un jour de l’année (1‑365/366) en mois (1‑12)."""
    date_ref = dt.date(2025, 1, 1) + dt.timedelta(days=day_of_year - 1)
    return date_ref.month

# ---------- RHS de l’EDO ----------

def rhs_temperature(t: float, T: float) -> float:
    day_of_year = int(t // 86400) + 1
    hour = (t % 86400) / 3600.0
    mois = day_to_month(day_of_year)

    # Puissance absorbée (on extrait la 2ᵉ composante du tuple)
    _, flux_abs = puissance_recue_point(lat_deg, lon_deg, mois, hour, alpha)

    return (flux_abs + sigma * Tatm**4 - sigma * T**4) / C

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
