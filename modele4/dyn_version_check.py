import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from fonctions import (
    P_inc_solar,
    P_em_surf_thermal,
    P_em_surf_conv,
    P_em_surf_evap,
)

# constantes
dt = 1800.0  # pas de temps (s)
MASSE_SURFACIQUE_ACTIVE = 400.0  # kg/m²
SIGMA = 5.670374419e-8

# charge albédo
_DEF_ALBEDO = {
    "ice": 0.60,
    "water": 0.10,
    "snow": 0.80,
    "desert": 0.35,
    "forest": 0.20,
    "land": 0.15,
}
_CAPACITY_BY_TYPE = {
    "ice": 2.0,
    "water": 4.18,
    "snow": 2.0,
    "desert": 0.8,
    "forest": 1.0,
    "land": 1.0,
}

def capacite_thermique_massique(albedo):
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    closest = min(_DEF_ALBEDO, key=lambda k: abs(albedo - _DEF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[closest]

def lisser_donnees_annuelles(valeurs_mensuelles, sigma):
    jours_par_mois = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    valeurs_journalieres = np.repeat(valeurs_mensuelles, jours_par_mois)
    return gaussian_filter1d(valeurs_journalieres, sigma=sigma, mode="wrap")

def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N+1)
    T[0] = T0

    albedo_annuel = np.full(12, 0.3)
    albedo_lisse = lisser_donnees_annuelles(albedo_annuel, sigma=15.0)
    cap_massique = np.vectorize(capacite_thermique_massique)(albedo_annuel) * 1000.0
    C_surf_mensuelle = cap_massique * MASSE_SURFACIQUE_ACTIVE
    C_lisse = lisser_donnees_annuelles(C_surf_mensuelle, sigma=15.0)

    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    for k in range(N):
        t = k * dt
        jour = int((t // 86400) % 365)
        albedo = albedo_lisse[jour]
        C = C_lisse[jour]

        Pinc = P_inc_solar(lat_rad, lon_rad, t)
        Pnet = (Pinc * (1 - albedo)
                - P_em_surf_thermal(lat_rad, lon_rad, t, T[k])
                - P_em_surf_conv(lat_rad, lon_rad, t)
                - P_em_surf_evap(lat_rad, lon_rad, t))

        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * (Pnet - SIGMA * X**4) / C
            dF = 1 - dt * (-4 * SIGMA * X**3) / C
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k+1] = X

    return T

def tracer_temperature(T, titre):
    times = np.arange(len(T)) * dt / 86400
    plt.figure(figsize=(12,5))
    plt.plot(times, T-273.15, label='Température surface (°C)')
    plt.xlabel("Jour de l'année")
    plt.ylabel('Température (°C)')
    plt.title(titre)
    plt.grid(ls=':')
    plt.xlim(0, 365)
    plt.show()

if __name__ == "__main__":
    jours_simulation = 365 * 2
    lat, lon = 49.0, 2.3
    print(f"Simulation pour latitude {lat}° et longitude {lon}°")
    T = backward_euler(jours_simulation, lat, lon)
    steps_per_year = int(365*24*3600/dt)
    tracer_temperature(T[steps_per_year:], f"Simulation année 2 (lat={lat}, lon={lon})")
