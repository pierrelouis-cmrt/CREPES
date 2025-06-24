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
# - NOUVEAU (votre demande) : La variation saisonnière de Q est
#   supprimée. Q est une valeur de base constante (positive le
#   jour, négative la nuit).
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import fonctions as f
import lib as lib 
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
EPAISSEUR_ACTIVE = 0.2  # m (20 cm)

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

# Définition de constantes physiques (identiques à celles vues précédemment)

RHO_W = 1000.0       # Densité de l'eau (kg/m³)
RHO_BULK = 1300.0    # Densité apparente du sol sec (kg/m³)
CP_SEC = 0.8         # Capacité thermique spécifique du sol sec (kJ/kg·K)
CP_WATER = 4.187     # Capacité thermique spécifique de l'eau (kJ/kg·K)
CP_ICE = 2.09        # Capacité thermique spécifique de la glace (kJ/kg·K)

# Chemin vers le fichier CSV contenant les données RZSM (humidité relative du sol)
RZSM_CSV_PATH = pathlib.Path("ressources/Cp_humidity/average_rzsm_tout.csv")

# ────────────────────────────────────────────────
# Capacité thermique depuis l'humidité du sol (RZSM) (inchangé)
# ────────────────────────────────────────────────

try:
    RZSM_GRID, RZSM_LAT_BINS, RZSM_LON_BINS = f.load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )
    if RZSM_GRID is None:
        raise RuntimeError("Scipy manquant ou échec du griddage RZSM.")
    _rzsm_lat_idx = lambda lat: np.abs(RZSM_LAT_BINS - lat).argmin()
    _rzsm_lon_idx = lambda lon: np.abs(RZSM_LON_BINS - lon).argmin()
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR: Impossible de charger les données d'humidité du sol : {e}")
    exit()


try:
    RZSM_GRID, RZSM_LAT_BINS, RZSM_LON_BINS = lib.load_and_grid_rzsm_data(
        RZSM_CSV_PATH
    )
    if RZSM_GRID is None:
        raise RuntimeError("Scipy manquant ou échec du griddage RZSM.")
    _rzsm_lat_idx = lambda lat: np.abs(RZSM_LAT_BINS - lat).argmin()
    _rzsm_lon_idx = lambda lon: np.abs(RZSM_LON_BINS - lon).argmin()
except (FileNotFoundError, RuntimeError) as e:
    print(f"ERREUR: Impossible de charger les données d'humidité du sol : {e}")
    exit()


# ────────────────────────────────────────────────
# Données de chaleur latente (Q) via évaporation (inchangé)
# ────────────────────────────────────────────────

Delta_hvap = 2453000
rho_eau = 1000
Delta_t = 31557600

evap_Eur = 0.49 / Delta_t
evap_Am_Nord = 0.47 / Delta_t
evap_Am_sud = 0.94 / Delta_t
evap_oceanie = 0.41 / Delta_t
evap_Afr = 0.58 / Delta_t
evap_Asi = 0.37 / Delta_t
evap_ocean = 1.40 / Delta_t

phi_Eur = Delta_hvap * rho_eau * evap_Eur
phi_Am_Nord = Delta_hvap * rho_eau * evap_Am_Nord
phi_Am_sud = Delta_hvap * rho_eau * evap_Am_sud
phi_oceanie = Delta_hvap * rho_eau * evap_oceanie
phi_Afr = Delta_hvap * rho_eau * evap_Afr
phi_Asi = Delta_hvap * rho_eau * evap_Asi
phi_ocean = Delta_hvap * rho_eau * evap_ocean

Q_LATENT_CONTINENT = {
    "Europe": phi_Eur,
    "North America": phi_Am_Nord,
    "South America": phi_Am_sud,
    "Oceania": phi_oceanie,
    "Africa": phi_Afr,
    "Asia": phi_Asi,
    "Océan": phi_ocean,
    "Antarctica": 0.0,
}

SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)


continent_finder = f.create_continent_finder(SHAPEFILE_PATH)


# ────────────────────────────────────────────────
# Données d'albédo des nuages depuis CERES (inchangé)
# ────────────────────────────────────────────────

CERES_FILE_PATH = (
    pathlib.Path("ressources/albedo")
    / "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
)



#Bilan

def f_rhs(T, phinet, C, q_latent):
    return (phinet - q_latent + sigma * Tatm**4 - sigma * T**4) / C


# ────────────────────────────────────────────────
# Intégrateur Backward‑Euler (MODIFIÉ pour Q)
# ────────────────────────────────────────────────


def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    albedo_sol_hist, albedo_nuages_hist, C_hist, q_latent_hist = (
        np.empty(N + 1) for _ in range(4)
    )
    T[0] = T0
    lat_rad, lat_idx, lon_idx = (
        np.radians(lat_deg),
        _lat_idx(lat_deg),
        _lon_idx(lon_deg),
    )

    # MODIFIÉ : q_base est maintenant la valeur constante pour toute la simulation
    q_base = lib.P_em_surf_evap(lat_deg, lon_deg)

    print("Lissage des données annuelles (albédo)...")
    albedo_sol_mensuel_loc = monthly_albedo_sol[:, lat_idx, lon_idx]
    albedo_sol_journalier_lisse = f.lisser_donnees_annuelles(
        albedo_sol_mensuel_loc, sigma=15.0
    )
    albedo_nuages_mensuel = f.load_monthly_cloud_albedo_from_ceres(
        lat_deg, lon_deg
    )
    albedo_nuages_journalier_lisse = f.lisser_donnees_annuelles(
        albedo_nuages_mensuel, sigma=15.0
    )

    print("Calcul de la capacité thermique à partir des données RZSM...")
    rzsm_lat_idx = _rzsm_lat_idx(lat_deg)
    rzsm_lon_idx = _rzsm_lon_idx(lon_deg)
    rzsm_value = RZSM_GRID[rzsm_lat_idx, rzsm_lon_idx]
    cp_kj = (
        f.compute_cp_from_rzsm(np.array([rzsm_value]))[0]
        if not np.isnan(rzsm_value)
        else CP_SEC
    )
    C_const = (cp_kj * 1000.0) * RHO_BULK * EPAISSEUR_ACTIVE
    print(
        f"RZSM={rzsm_value:.3f} -> c_p={cp_kj:.3f} kJ/kg/K -> "
        f"C={C_const:.2e} J m⁻² K⁻¹"
    )

    # Initialisation des tableaux d'historique
    albedo_sol_hist[0], albedo_nuages_hist[0], C_hist[0], q_latent_hist[0] = (
        albedo_sol_journalier_lisse[0],
        albedo_nuages_journalier_lisse[0],
        C_const,
        q_base,  # MODIFIÉ
    )

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        jour_dans_annee = (jour - 1) % 365

        albedo_sol = albedo_sol_journalier_lisse[jour_dans_annee]
        albedo_nuages = albedo_nuages_journalier_lisse[jour_dans_annee]
        # MODIFIÉ : q_latent_daily est maintenant la valeur de base constante
        q_latent_daily = q_base

        albedo_sol_hist[k + 1], albedo_nuages_hist[k + 1], C_hist[
            k + 1
        ], q_latent_hist[k + 1] = (
            albedo_sol,
            albedo_nuages,
            C_const,
            q_latent_daily,
        )

        phi_n = lib.P_em_surf_thermal(
            lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages
        )
        # La logique d'inversion jour/nuit est conservée
        q_latent_step = q_latent_daily if phi_n > 0 else -q_latent_daily

        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_step)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return T, albedo_sol_hist, albedo_nuages_hist, C_hist, q_latent_hist


# ────────────────────────────────────────────────
# Fonctions de tracé et exécution principale (inchangées)
# ────────────────────────────────────────────────


def tracer_comparaison(
    times,
    T,
    albedo_sol_hist,
    albedo_nuages_hist,
    C_hist,
    q_latent_hist,
    titre,
    jour_a_afficher,
):
    fig, axs = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True, height_ratios=[3, 2, 2]
    )
    days_axis = times / 86400

    axs[0].plot(
        days_axis,
        T - 273.15,
        lw=1.0,
        color="gray",
        alpha=0.8,
        label="Simulation Année 2",
    )
    steps_per_day = int(24 * 3600 / dt)
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)
    start_idx = min(start_idx, end_idx)
    axs[0].plot(
        days_axis[start_idx : end_idx + 1],
        T[start_idx : end_idx + 1] - 273.15,
        lw=2.5,
        color="firebrick",
        label=f"Jour n°{jour_a_afficher}",
    )
    axs[0].set_ylabel("Température surface (°C)")
    axs[0].set_title(titre)
    axs[0].grid(ls=":")
    axs[0].legend()
    axs[0].set_xlim(0, 365)

    color1 = "tab:blue"
    axs[1].set_ylabel("Albédo (sans unité)", color=color1)
    axs[1].plot(
        days_axis, albedo_sol_hist, color=color1, lw=2.0, label="Albédo Sol (A2)"
    )
    axs[1].plot(
        days_axis,
        albedo_nuages_hist,
        color="cyan",
        lw=2.0,
        ls=":",
        label="Albédo Nuages (A1)",
    )
    axs[1].tick_params(axis="y", labelcolor=color1)
    axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5))
    axs[1].legend(loc="upper left")
    axs[1].grid(ls=":")

    color_q = "tab:green"
    axs[2].set_ylabel("Flux Chaleur Latente (W m⁻²)", color=color_q)
    # Le graphique affichera une ligne constante pour Q
    axs[2].plot(
        days_axis,
        q_latent_hist,
        color=color_q,
        lw=2.0,
        label="Flux Latent de base (Q)",
    )
    axs[2].tick_params(axis="y", labelcolor=color_q)
    axs[2].legend(loc="upper left")

    ax3 = axs[2].twinx()
    color_c = "tab:red"
    ax3.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color=color_c)
    ax3.plot(
        days_axis,
        C_hist,
        color=color_c,
        lw=2.0,
        ls="--",
        label="Capacité (droite)",
    )
    ax3.tick_params(axis="y", labelcolor=color_c)
    ax3.legend(loc="upper right")

    axs[2].set_xlabel("Jour de l'année (simulation stabilisée)")
    axs[2].grid(ls=":")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182

    # Pour Paris (Europe)
    lat_sim, lon_sim = 48.5, 2.3
    # Pour l'Amazonie (Amérique du Sud, Q élevé)
    # lat_sim, lon_sim = -3.46, -62.21
    # Pour le Sahara (Afrique, Q modéré, Cp faible)
    # lat_sim, lon_sim = 25.0, 15.0

    print(
        f"Lancement de la simulation pour Lat={lat_sim}N, Lon={lon_sim}E..."
    )
    (
        T_full,
        alb_sol_full,
        alb_nuages_full,
        C_full,
        q_latent_full,
    ) = backward_euler(jours_de_simulation, lat_sim, lon_sim)

    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt
    T_yr2, alb_sol_yr2, alb_nuages_yr2, C_yr2, q_latent_yr2 = (
        arr[steps_per_year:]
        for arr in [
            T_full,
            alb_sol_full,
            alb_nuages_full,
            C_full,
            q_latent_full,
        ]
    )

    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_yr2,
        f"Simulation (Q constant) pour Lat={lat_sim}, Lon={lon_sim}",
        jour_a_afficher,
    )