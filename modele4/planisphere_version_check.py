import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import pi
from tqdm import tqdm
import os

# Fonctions physiques importées (tes définitions)
from fonctions import (  # adapte ce import selon ton fichier
    P_inc_solar,
    P_abs_surf_solar,
    P_em_surf_thermal,
    P_em_surf_conv,
    P_em_surf_evap,
    P_abs_atm_solar,
    P_abs_atm_thermal,
    P_em_atm_thermal_up,
    P_em_atm_thermal_down,
)

# ---------- constantes ----------
SIGMA = 5.67e-8
Tatm = 253.15  # K
C = 8.36e5  # J/m²/K
dt = 3600.0  # 1h

NLAT = 91
NLON = 181
lats = np.linspace(-pi/2, pi/2, NLAT)  # en radians
lons = np.linspace(-pi, pi, NLON)

LON_PARIS = 2.35 * pi / 180
TIME_OFFSET_SEC = (LON_PARIS / (2 * pi)) * 86400.0

# ---------- EDO et intégrateur ----------
def bilan_flux(lat, lon, t, Tsurf):
    """Calcul du flux net en W/m²"""
    Pinc = P_inc_solar(lat, lon, t)
    Pabs_surf = P_abs_surf_solar(lat, lon, t, Pinc)
    Pem_surf_thermal = P_em_surf_thermal(lat, lon, t, Tsurf)
    Pem_surf_conv = P_em_surf_conv(lat, lon, t)
    Pem_surf_evap = P_em_surf_evap(lat, lon, t)
    Pabs_atm_solar = P_abs_atm_solar(lat, lon, t, Pinc)
    Pabs_atm_thermal = P_abs_atm_thermal(lat, lon, t, Tatm)
    Pem_atm_down = P_em_atm_thermal_down(lat, lon, t)

    # Bilan de puissance nette sur la surface
    flux_net = (
        Pabs_surf
        + Pem_atm_down
        - Pem_surf_thermal
        - Pem_surf_conv
        - Pem_surf_evap
    )
    return flux_net

def backward_euler(days, lat, lon, T0=288.0):
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0

    for k in range(N):
        t_sim_sec = k * dt
        t_utc_sec = t_sim_sec - TIME_OFFSET_SEC
        X = T[k]

        for _ in range(8):
            Pnet = bilan_flux(lat, lon, t_utc_sec, X)
            F = X - T[k] - dt * (Pnet / C)
            dF = 1 - dt * (-4 * SIGMA * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return T

# ---------- simulation grille globale ----------
def run_full_simulation(days, result_file="temp_grid.npy"):
    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print("Simulation globale...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))

    for i in tqdm(range(NLAT), desc="Latitudes"):
        for j in range(NLON):
            lat = lats[i]
            lon = lons[j]
            T0 = 288.15 - 20 * np.sin(lat)**2
            T_series = backward_euler(days, lat, lon, T0=T0)
            T_grid[:, i, j] = T_series

    print(f"Sauvegarde dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid

# ---------- visualisation ----------
if __name__ == "__main__":
    SIM_DAYS = 365
    T_grid_all_times = run_full_simulation(SIM_DAYS)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        T_grid_all_times[0, :, :],
        origin="lower",
        extent=[-180, 180, -90, 90],
        cmap="inferno",
        vmin=220,
        vmax=320,
    )
    plt.colorbar(im, ax=ax, label="Température (K)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_day = Slider(ax_slider, "Jour", 0, SIM_DAYS-1, valinit=0, valstep=1)

    def _update(val):
        day = int(slider_day.val)
        idx = day * 24
        im.set_data(T_grid_all_times[idx, :, :])
        ax.set_title(f"Jour {day}")
        fig.canvas.draw_idle()

    slider_day.on_changed(_update)
    _update(0)

    plt.show()