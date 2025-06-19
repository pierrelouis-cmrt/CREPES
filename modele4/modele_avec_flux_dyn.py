# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
# Le flux solaire est calculé avec votre fonction puissance_recue_point
# (inclinaison saisonnière + rotation diurne par matrice 3-D)
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from datetime import datetime, timedelta

# ---------- constantes physiques ----------
constante_solaire = 1361.0          # W m-2
sigma   = 5.670374419e-8            # Stefan-Boltzmann (SI)
alpha   = 0.25                      # albédo de surface (modifiable)
Tatm    = 253.15                    # atmosphère radiative (-20 °C)
C       = 8.36e5                    # capacité surfacique (J m-2 K-1)
dt      = 1800.0                    # pas de temps : 30 min

# ---------- votre moteur solaire intact ----------
def update_sun_vector(mois, sun_vector):
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    return rotation_matrix_saison @ sun_vector

def puissance_recue_point(lat_deg, lon_deg, mois, time, albedo=0.3):
    """Flux solaire net reçu par le point (W m-2) et T_eq (K)"""
    theta = np.radians(90 - lat_deg)        # colatitude
    phi   = np.radians(lon_deg % 360)
    sun_vector = np.array([1.0, 0.0, 0.0])

    # Inclinaison saisonnière
    sun_vector = update_sun_vector(mois, sun_vector)

    # Rotation diurne
    angle_rotation = (time / 24.0) * 2.0 * np.pi
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation),  np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector = rotation_matrix @ sun_vector

    # Vecteur normal du point
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    cos_incidence = max(np.dot(normal, sun_vector), 0.0)
    puissance_recue = constante_solaire * cos_incidence * (1 - albedo)
    temperature_eq  = (puissance_recue / sigma) ** 0.25
    return temperature_eq, puissance_recue

# ---------- petite aide : jour → mois 1-12 ----------
# tableau des jours cumulés (année non bissextile)
_jcum = np.array([0, 31, 59, 90, 120, 151, 181,
                  212, 243, 273, 304, 334, 365])
def mois_from_jour(j):
    """Retourne le mois (1-12) pour le jour de l’année j (1-365)"""
    return int(np.searchsorted(_jcum, j, side='right'))

# ---------- RHS de l’EDO ----------
def f_rhs(T, phinet):
    return (phinet + sigma * Tatm**4 - sigma * T**4) / C

# ---------- intégrateur Backward-Euler ----------
def backward_euler(days,
                   lat_deg=49.0,
                   lon_deg=2.3,
                   albedo=alpha,
                   T0=288.0):
    """
    Intègre la température de surface pendant *days* jours.
    lat_deg : latitude  (+ N, – S)
    lon_deg : longitude (+ E, – W)
    """
    N = int(days * 24 * 3600 / dt)
    times = np.arange(N + 1) * dt
    T = np.empty(N + 1);  T[0] = T0

    for k in range(N):
        t_sec   = k * dt
        jour    = int(t_sec // 86400) + 1          # 1 ⇒ 1er janvier
        heure_utc = (t_sec % 86400) / 3600.0

        # Heure locale pour la rotation diurne
        heure_locale = (heure_utc + lon_deg / 15.0) % 24.0
        mois = mois_from_jour(jour)

        # Flux solaire « exact » (déjà ×(1-albedo))
        _, phi_n = puissance_recue_point(lat_deg, lon_deg,
                                         mois, heure_locale, albedo)

        # Newton pour Backward-Euler : F(X) = X - T[k] - dt·f_rhs(X)
        X = T[k]
        for _ in range(8):
            F  = X - T[k] - dt * f_rhs(X, phi_n)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return times, T

# ---------- tracé ----------
def tracer(times, T, titre):
    plt.figure(figsize=(10, 4))
    plt.plot(times / 86400, T - 273.15, lw=1.2)
    plt.xlabel("Jour")
    plt.ylabel("Température surface (°C)")
    plt.title(titre)
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()

# ---------- exécution test ----------
if __name__ == "__main__":
    lat_paris, lon_paris = 9, 8
    for jours in (150, 152):
        t, T = backward_euler(jours, lat_paris, lon_paris, albedo=alpha)
        tracer(t, T,
               f"Backward-Euler (flux solaire « sun-vector ») – {jours} jour{'s' if jours > 1 else ''}")
