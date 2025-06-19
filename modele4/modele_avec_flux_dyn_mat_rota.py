# ---------------------------------------------------------------
# Backward-Euler implicite pour l’équa diff
# Avec flux dynamique
# ---------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import datetime as dt

# ---------- constantes & paramètres physiques ----------
S0    = 1361.0                     # constante solaire (W m-2)
sigma = 5.670374419e-8             # Stefan-Boltzmann
alpha = 0.25                       # albédo de surface
Tatm  = 253.15                     # atmosphère radiative (-20 °C)
C     = 8.36e5                     # capacité surfacique (J m-2 K-1)

# ---------- localisation du point d'étude ----------
lat_deg = 49.0                     # latitude Paris en degrés
lon_deg = 2.0                      # longitude Paris en degrés

dt_sec = 1800.0                    # pas de temps (30 min)

# ---------- outils solaires basés sur matrices de rotation ----------

def update_sun_vector(month: int, sun_vector: np.ndarray) -> np.ndarray:
    """Applique l'inclinaison saisonnière à un vecteur Soleil→Terre 1 AU"""
    # Inclinaison axiale 23°26' = 23.44°
    angle_inclinaison = np.radians(23.44) * np.cos(2*pi*(month-1)/12)
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    return rotation_matrix_saison @ sun_vector


def puissance_recue_point(lat_deg: float, lon_deg: float, month: int, hour: float, albedo: float = alpha) -> float:
    """Flux solaire net reçu par un point donné en W m⁻²"""
    theta = np.radians(90 - lat_deg)              # colatitude
    phi   = np.radians(lon_deg % 360)

    sun_vector = np.array([1.0, 0.0, 0.0])        # Soleil à +∞ sur l'axe +x
    # saison
    sun_vector = update_sun_vector(month, sun_vector)
    # rotation diurne
    angle_rotation = (hour / 24.0) * 2*pi
    rotation_matrix_diurne = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation),  np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector = rotation_matrix_diurne @ sun_vector

    # vecteur normal au point
    normal = np.array([
        np.sin(theta)*np.cos(phi),
        np.sin(theta)*np.sin(phi),
        np.cos(theta)
    ])

    cos_incidence = max(np.dot(normal, sun_vector), 0.0)
    return S0 * cos_incidence * (1 - albedo)


def day_to_month(day_of_year: int) -> int:
    """Convertit un jour de l'année (1-365/366) en mois (1-12)"""
    base_date = dt.date(2025, 1, 1) + dt.timedelta(days=day_of_year - 1)  # année de référence non bissextile
    return base_date.month


def phi_net(day_of_year: int, hour: float) -> float:
    month = day_to_month(day_of_year)
    return puissance_recue_point(lat_deg, lon_deg, month, hour, alpha)


def f_rhs(T: float, phinet: float) -> float:
    """Terme de droite de l'équation d'énergie surfacique"""
    return (phinet + sigma*Tatm**4 - sigma*T**4) / C

# ---------- intégrateur Backward-Euler (Newton) ----------

def backward_euler(days: int, T0: float = 288.0):
    """Intègre l'équation 0‑D sur *days* jours avec un pas dt_sec"""
    N = int(days*24*3600/dt_sec)
    times = np.arange(N + 1) * dt_sec
    T = np.empty(N + 1)
    T[0] = T0
    for k in range(N):
        t_sec   = k * dt_sec
        day     = int(t_sec // 86400) + 1        # jour de l'année (1‑365)
        hour    = (t_sec % 86400) / 3600.0
        phi_n   = phi_net(day, hour)

        # Newton pour résoudre F(X) = X − T[k] − dt*f(X) = 0
        X = T[k]
        for _ in range(8):
            F  = X - T[k] - dt_sec * f_rhs(X, phi_n)
            dF = 1 - dt_sec * (-4 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X
    return times, T

# ---------- tracé ----------

def tracer(times, T, titre: str):
    plt.figure(figsize=(10, 4))
    plt.plot(times/86400.0, T - 273.15, lw=1.2)
    plt.xlabel("Jour")
    plt.ylabel("Température surface (°C)")
    plt.title(titre)
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    for jours in (1, 365):
        t, Tsurf = backward_euler(jours)
        tracer(t, Tsurf, f"Backward-Euler – {jours} jour{'s' if jours > 1 else ''}")