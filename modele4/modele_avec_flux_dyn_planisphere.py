# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Intégration sur planisphère
#
# - Le flux solaire est calculé avec la fonction puissance_recue_point.
# - L'intégration temporelle utilise un schéma Backward-Euler implicite.
# - La simulation est effectuée pour une grille de points sur tout le globe.
# - L'origine du temps (jour=0, heure=0) est minuit à Paris.
# - La visualisation est interactive avec un curseur pour le temps.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from math import pi
from tqdm import tqdm
import os

# Optionnel : utiliser cartopy pour un meilleur rendu des côtes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

# ---------- constantes physiques et de simulation ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan-Boltzmann (SI)
alpha = 0.25  # albédo de surface global (modifiable)
Tatm = 253.15  # température radiative de l'atmosphère (-20 °C)
C = 8.36e5  # capacité thermique surfacique (J m-2 K-1)
dt = 3600.0  # pas de temps : 1 heure

# ---------- paramètres de la grille et du temps ----------
NLAT = 91  # Nombre de points en latitude (-90 à +90)
NLON = 181  # Nombre de points en longitude (-180 à +180)
lats = np.linspace(-90, 90, NLAT)
lons = np.linspace(-180, 180, NLON)

# Définition de l'origine temporelle : minuit à Paris
LON_PARIS = 2.35  # Longitude de Paris en degrés Est
# Offset en secondes pour que t_sim=0 corresponde à 00:00 à Paris
# heure_locale = heure_utc + lon/15. On veut heure_locale(Paris)=0 pour t_sim=0.
# 0 = (t_utc_sec / 3600) + LON_PARIS / 15.
# t_utc_sec = - (LON_PARIS / 15) * 3600
# Notre temps de simulation t_sim est lié au temps UTC par :
# t_utc_sec = t_sim_sec - OFFSET.
# Donc, à t_sim_sec=0, t_utc_sec = -OFFSET.
# On en déduit que OFFSET = (LON_PARIS / 15) * 3600
TIME_OFFSET_SEC = (LON_PARIS / 15.0) * 3600.0

# ---------- moteur solaire (inchangé) ----------
def update_sun_vector(mois, sun_vector):
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array(
        [
            [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
            [0, 1, 0],
            [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)],
        ]
    )
    return rotation_matrix_saison @ sun_vector


def puissance_recue_point(lat_deg, lon_deg, mois, time, albedo=0.3):
    """Flux solaire net reçu par le point (W m-2) et T_eq (K)"""
    theta = np.radians(90 - lat_deg)  # colatitude
    phi = np.radians(lon_deg % 360)
    sun_vector = np.array([1.0, 0.0, 0.0])
    sun_vector = update_sun_vector(mois, sun_vector)
    angle_rotation = (time / 24.0) * 2.0 * np.pi
    rotation_matrix = np.array(
        [
            [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
            [np.sin(angle_rotation), np.cos(angle_rotation), 0],
            [0, 0, 1],
        ]
    )
    sun_vector = rotation_matrix @ sun_vector
    normal = np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    cos_incidence = max(np.dot(normal, sun_vector), 0.0)
    puissance_recue = constante_solaire * cos_incidence * (1 - albedo)
    temperature_eq = (puissance_recue / sigma) ** 0.25
    return temperature_eq, puissance_recue


# ---------- helper : jour → mois 1-12 ----------
_jcum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])


def mois_from_jour(j):
    """Retourne le mois (1-12) pour le jour de l’année j (1-365)"""
    return int(np.searchsorted(_jcum, j, side="right"))


# ---------- EDO et intégrateur ----------
def f_rhs(T, phinet):
    return (phinet + sigma * Tatm ** 4 - sigma * T ** 4) / C


def backward_euler(days, lat_deg, lon_deg, albedo=alpha, T0=288.0):
    """Intègre la température de surface pour UN SEUL point."""
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0

    for k in range(N):
        t_sim_sec = k * dt
        # On applique le décalage pour que t_sim=0 soit minuit à Paris
        t_utc_sec = t_sim_sec - TIME_OFFSET_SEC

        # Jour et heure UTC (pour les saisons et la position du soleil)
        jour_utc = int((t_utc_sec / 86400) % 365) + 1
        heure_utc = (t_utc_sec % 86400) / 3600.0
        mois = mois_from_jour(jour_utc)

        # Heure locale pour la rotation diurne
        heure_locale = (heure_utc + lon_deg / 15.0) % 24.0

        _, phi_n = puissance_recue_point(
            lat_deg, lon_deg, mois, heure_locale, albedo
        )

        # Newton pour Backward-Euler : F(X) = X - T[k] - dt·f_rhs(X)
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n)
            dF = 1.0 - dt * (-4.0 * sigma * X ** 3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X
    return T


def run_full_simulation(days, result_file="temp_grid.npy"):
    """Exécute la simulation pour toute la grille et sauvegarde le résultat."""
    if os.path.exists(result_file):
        print(f"Chargement des résultats depuis '{result_file}'...")
        return np.load(result_file)

    print("Lancement de la simulation globale (cela peut prendre du temps)...")
    N_steps = int(days * 24 * 3600 / dt) + 1
    T_grid = np.zeros((N_steps, NLAT, NLON))

    # Utilisation de tqdm pour la barre de progression
    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat = lats[i]
            lon = lons[j]
            # On initialise à une température raisonnable
            T0 = 288.15 - 20 * np.sin(np.radians(lat)) ** 2
            T_series = backward_euler(days, lat, lon, albedo=alpha, T0=T0)
            T_grid[:, i, j] = T_series

    print(f"Sauvegarde des résultats dans '{result_file}'...")
    np.save(result_file, T_grid)
    return T_grid


# ---------- exécution principale et tracé ----------
if __name__ == "__main__":
    SIM_DAYS = 365  # Simuler une année complète
    T_grid_all_times = run_full_simulation(SIM_DAYS)

    # --- Configuration de la figure et des widgets ---
    plt.close("all")
    if USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 7))
        ax = plt.axes(projection=proj)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        proj = None

    # Affichage de l'état initial (Jour 0)
    initial_T_grid = T_grid_all_times[0, :, :]
    _ORIGIN = "lower"  # lats vont de -90 à 90

    if proj is not None:
        im = ax.imshow(
            initial_T_grid,
            origin=_ORIGIN,
            extent=[-180, 180, -90, 90],
            transform=proj,
            cmap="inferno",
            vmin=220,
            vmax=320,
        )
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="w")
            ax.set_global()
    else:
        im = ax.imshow(
            initial_T_grid,
            origin=_ORIGIN,
            extent=[-180, 180, -90, 90],
            cmap="inferno",
            interpolation="nearest",
            vmin=220,
            vmax=320,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cb.set_label("Température de surface (K)")

    # Création du curseur pour le jour
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_day = Slider(
        ax_slider, "Jour", 0, SIM_DAYS - 1, valinit=0, valstep=1, color="0.5"
    )

    def _refresh(val):
        day = int(slider_day.val)
        # Il y a 24 pas de temps par jour car dt=3600s
        time_step_index = day * 24
        T_slice = T_grid_all_times[time_step_index, :, :]
        im.set_data(T_slice)
        ax.set_title(f"Température de surface (K) - Jour {day}")
        fig.canvas.draw_idle()

    slider_day.on_changed(_refresh)
    _refresh(0)  # Appel initial pour régler le titre

    # --- Gestion des clics sur la carte ---
    _annotation = None

    def _lat_idx(lat):
        return np.abs(lats - lat).argmin()

    def _lon_idx(lon):
        return np.abs(lons - lon).argmin()

    def _onclick(event):
        global _annotation
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        lon, lat = float(event.xdata), float(event.ydata)
        li, lj = _lat_idx(lat), _lon_idx(lon)
        day = int(slider_day.val)
        time_step_index = day * 24

        temp_simulee = T_grid_all_times[time_step_index, li, lj]

        if _annotation is not None:
            _annotation.remove()

        txt = (
            f"Coord: ({lat:+.1f}°, {lon:+.1f}°)\n"
            f"Jour {day}\n"
            f"Temp. simulée: {temp_simulee - 273.15:.1f} °C ({temp_simulee:.1f} K)"
        )
        _annotation = ax.annotate(
            txt,
            xy=(lon, lat),
            xycoords="data",
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="white",
            backgroundcolor=(0, 0, 0, 0.6),
            ha="left",
            va="bottom",
            zorder=10,
        )

        def _hide():
            global _annotation
            if _annotation is not None:
                _annotation.remove()
                _annotation = None
                fig.canvas.draw_idle()

        # Cache l'annotation après 4 secondes
        fig.canvas.new_timer(interval=4000, callbacks=[(_hide, [], {})]).start()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _onclick)

    plt.subplots_adjust(bottom=0.15)
    plt.show()