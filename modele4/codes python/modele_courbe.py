# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# VERSION MISE À JOUR :
# - Ajout de l'albédo des nuages (A1) en plus de l'albédo de
#   surface (A2).
# - Lissage des données mensuelles (albédo sol, albédo nuages)
#   par convolution gaussienne.
# - Utilisation de GeoPandas pour une détection précise
#   des continents.
# - Visualisation du flux de chaleur latente (Q) dans
#   le graphique de sortie.
# - CORRIGÉ : Gestion des géométries nulles dans le shapefile.
# - NOUVEAU : Remplacement des données mock de l'albédo des nuages
#   par un calcul basé sur les données CERES (fichier .nc).
# - NOUVEAU : La capacité thermique est calculée à partir des
#   données d'humidité du sol (RZSM).
# - NOUVEAU : Le flux de chaleur latente (Q) est calculé à partir
#   des taux d'évaporation annuels par continent.
# - NOUVEAU : La variation saisonnière de Q est
#   supprimée. Q est une valeur de base constante (positive le
#   jour, négative la nuit).
# - NOUVEAU : Correction manuelle pour la détection
#   de l'Arctique (Q=0).
# - NOUVEAU (votre demande) : Visualisation du flux de chaleur
#   latent alterné pour un seul jour, avec une transition lissée.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import sys
import cftime
import subprocess
import fonctions as f 
import lib as lib

try:
    import sys
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", sys])
    import sys

try:
    import cftime
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", cftime])
    import cftime

try:
    import xarray
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", xarray])
    import xarray

try:
    import numpy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", numpy])
    import numpy

try:
    import matplotlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", matplotlib])
    import matplotlib

try:
    import math
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", math])
    import math

try:
    import pathlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pathlib])
    import pathlib

try:
    import pandas
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pandas])
    import pandas

try:
    import scipy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", scipy])
    import scipy

# NOUVEAU : Importations pour la partie géospatiale et NetCDF
try:
    import geopandas as gpd
    from shapely.geometry import Point

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

# NOUVEAU : Importation pour le traitement des données NetCDF
try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False

# NOUVEAU : Importation pour le griddage des données d'humidité
try:
    from scipy.stats import binned_statistic_2d

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (‑50 °C)
dt = 1800.0  # pas de temps : 30 min
EPAISSEUR_ACTIVE = 0.5  # m (20 cm)

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo de surface (inchangé)
# ────────────────────────────────────────────────

try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = f.load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    exit()

# ────────────────────────────────────────────────
# Capacité thermique depuis l'humidité du sol (RZSM) (inchangé)
# ────────────────────────────────────────────────

RHO_W = 1000.0
RHO_BULK = 1300.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09
RZSM_CSV_PATH = pathlib.Path("ressources/Cp_humidity/average_rzsm_tout.csv")

try:
    RZSM_GRID, RZSM_LAT_BINS, RZSM_LON_BINS = f.load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )
    if RZSM_GRID is None:
        raise RuntimeError("Scipy manquant ou échec du griddage RZSM.")
    _rzsm_lat_idx = lambda lat: min(
        np.abs(RZSM_LAT_BINS - lat).argmin(), RZSM_GRID.shape[0] - 1
    )
    _rzsm_lon_idx = lambda lon: min(
        np.abs(RZSM_LON_BINS - lon).argmin(), RZSM_GRID.shape[1] - 1
    )
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR: Impossible de charger les données d'humidité du sol : {e}")
    exit()

# ────────────────────────────────────────────────
# Bilan de flux thermique 
# ────────────────────────────────────────────────
def f_rhs(T, phinet, C, q_latent):
    return (phinet - q_latent + lib.P_em_atm_thermal(Tatm) - lib.P_em_surf_thermal(T)) / C


# ────────────────────────────────────────────────
# Intégrateur Backward‑Euler (inchangé)
# ────────────────────────────────────────────────

def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0, sigma_q=3.0):
    """Intégrateur Backward‑Euler avec flux latent journalier lissé.

    - Pré‑calcul du signal jour/nuit sur l'ensemble de la simulation.
    - Convolution gaussienne (sigma_q en *pas de temps*) appliquée au flux
      latent afin que la version lissée soit directement injectée dans le
      modèle thermique.
    """

    N = int(days * 24 * 3600 / dt)  # nombre total de pas
    T = np.empty(N + 1)
    (
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
    ) = (np.empty(N + 1) for _ in range(5))

    T[0] = T0
    lat_rad = np.radians(lat_deg)
    lat_idx, lon_idx = _lat_idx(lat_deg), _lon_idx(lon_deg)

    # Valeur de base de Q pour la zone étudiée
    q_base = lib.P_em_surf_evap(lat_deg, lon_deg)

    # ── Pré‑calcul jour/nuit pour toute la simulation ────────────────────
    sign_daynight = np.empty(N)
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        sign_daynight[k] = 1.0 if f.cos_incidence(lat_rad, jour, heure_solaire) > 0 else -1.0

    q_latent_raw = q_base * sign_daynight
    q_latent_smoothed = gaussian_filter1d(q_latent_raw, sigma=sigma_q, mode="wrap")

    # ── Données d'albédo (sol & nuages) et capacité thermique ────────────
    albedo_sol_m_loc = monthly_albedo_sol[:, lat_idx, lon_idx]
    alb_sol_daily = f.lisser_donnees_annuelles(albedo_sol_m_loc, sigma=15.0)

    alb_nuages_m = f.load_monthly_cloud_albedo_from_ceres(lat_deg, lon_deg)
    alb_nuages_daily = f.lisser_donnees_annuelles(alb_nuages_m, sigma=15.0)

    rzsm_lat_idx = _rzsm_lat_idx(lat_deg)
    rzsm_lon_idx = _rzsm_lon_idx(lon_deg)
    rzsm_val = RZSM_GRID[rzsm_lat_idx, rzsm_lon_idx]
    cp_kj = f.compute_cp_from_rzsm(np.array([rzsm_val]))[0] if not np.isnan(rzsm_val) else CP_SEC
    C_const = (cp_kj * 1000.0) * RHO_BULK * EPAISSEUR_ACTIVE

    (
        albedo_sol_hist[0],
        albedo_nuages_hist[0],
        C_hist[0],
        q_latent_hist[0],
        q_latent_step_hist[0],
    ) = (
        alb_sol_daily[0],
        alb_nuages_daily[0],
        C_const,
        q_base,
        q_latent_smoothed[0],
    )

    # ── Boucle d'intégration ─────────────────────────────────────────────
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        day_of_year = (jour - 1) % 365

        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[k]

        phi_n = lib.P_inc_solar(lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages)

        # Newton–Raphson implicite
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_step)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

        # Historique pour diagnostics
        albedo_sol_hist[k + 1] = albedo_sol
        albedo_nuages_hist[k + 1] = albedo_nuages
        C_hist[k + 1] = C_const
        q_latent_hist[k + 1] = q_base
        q_latent_step_hist[k + 1] = q_latent_step

    return (
        T,
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
    )

# ────────────────────────────────────────────────
# Fonctions de tracé et exécution principale (MODIFIÉ)
# ────────────────────────────────────────────────


def tracer_comparaison(
    times,
    T,
    albedo_sol_hist,
    albedo_nuages_hist,
    C_hist,
    q_latent_hist,
    q_latent_step_hist,
    titre,
    jour_a_afficher,
    sigma_plot=3.0,
):
    """Trace la température, les albédos, la capacité thermique et le flux latent.

    - Le flux latent affiché est déjà lissé (issu du modèle). Pour encore plus
      de lisibilité on applique à l'affichage une petite convolution gaussienne
      supplémentaire (sigma_plot).
    """

    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, height_ratios=[3, 2, 2])
    days_axis = times / 86400
    steps_per_day = int(24 * 3600 / dt)

    # ── Température ───────────────────────────────────────────────────────
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)

    axs[0].plot(days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8, label="Simulation Année 2")
    axs[0].plot(days_axis[start_idx:end_idx + 1], T[start_idx:end_idx + 1] - 273.15, lw=2.5, color="firebrick", label=f"Jour n°{jour_a_afficher}")
    axs[0].set_ylabel("Température surface (°C)",fontsize=14)
    axs[0].set_title(titre)
    axs[0].grid(ls=":")
    axs[0].legend(fontsize=12)
    axs[0].set_xlim(0, 365)

    # ── Albédos ───────────────────────────────────────────────────────────
    axs[1].set_ylabel("Albédo (sans unité)",fontsize=14)
    axs[1].plot(days_axis, albedo_sol_hist, color="tab:blue", lw=2.0, label="Albédo Sol (A2)")
    axs[1].plot(days_axis, albedo_nuages_hist, color="cyan", lw=2.0, ls=":", label="Albédo Nuages (A1)")
    axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5))
    axs[1].legend(loc="upper left",fontsize=12)
    axs[1].grid(ls=":")

    # ── Flux latent & capacité thermique ──────────────────────────────────
    q_plot = gaussian_filter1d(q_latent_step_hist, sigma=sigma_plot, mode="wrap")
    axs[2].plot(days_axis, q_plot, color="tab:green", lw=1.5, alpha=0.6, label="Flux Latent lissé (Q)")
    axs[2].set_ylabel("Flux Chaleur Latente (W m⁻²)",fontsize=14)
    axs[2].legend(loc="upper left",fontsize=12)

    ax3 = axs[2].twinx()
    ax3.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color="tab:red",fontsize=14)
    ax3.plot(days_axis, C_hist, color="tab:red", lw=2.0, ls="--", label="Capacité thermique")
    ax3.tick_params(axis="y", labelcolor="tab:red")
    ax3.legend(loc="upper right",fontsize=12)

    axs[2].set_xlabel("Jour de l'année (simulation stabilisée)",fontsize=14)
    axs[2].grid(ls=":")

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.subplots_adjust(bottom=0.12)

    plt.show()




if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182  # Solstice d'été, bonne journée pour voir l'effet

    # Pour Paris (Europe)
    lat_sim, lon_sim = 48.5, 2.3
    # Pour l'Amazonie (Amérique du Sud, Q élevé)
<<<<<<< Updated upstream
    #lat_sim, lon_sim = -3.46, -62.21
=======
    # lat_sim, lon_sim = -3.46, -62.21
>>>>>>> Stashed changes
    # Pour le Sahara (Afrique, Q modéré, Cp faible)
    # lat_sim, lon_sim = 25.0, 15.0
    # Pour l'Océan Arctique (Pôle Nord)
    # lat_sim, lon_sim = 82.0, 135.0
    # Pour l'Antarctique (Pôle Sud)
    lat_sim, lon_sim = -76.0, 100.0

    print(
        f"Lancement de la simulation pour Lat={lat_sim}N, Lon={lon_sim}E..."
    )
    (
        T_full,
        alb_sol_full,
        alb_nuages_full,
        C_full,
        q_latent_full,
        q_latent_step_full,
    ) = backward_euler(jours_de_simulation, lat_sim, lon_sim)

    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt
    (
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_yr2,
        q_latent_step_yr2,
    ) = (
        arr[steps_per_year:]
        for arr in [
            T_full,
            alb_sol_full,
            alb_nuages_full,
            C_full,
            q_latent_full,
            q_latent_step_full,
        ]
    )

    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_yr2,
        q_latent_step_yr2,
        f"",
        jour_a_afficher,
    )

    