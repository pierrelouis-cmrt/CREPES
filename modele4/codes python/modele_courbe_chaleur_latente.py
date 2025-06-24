# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# VERSION MISE À JOUR :
# - Ajout de l'albédo des nuages (A1) en plus de l'albédo de
#   surface (A2).
# - Lissage des données mensuelles (albédo sol, albédo nuages,
#   capacité thermique) par convolution gaussienne pour obtenir
#   des variations journalières continues et cycliques.
# - Ajout du flux de chaleur latente (Q) dépendant du continent.
# - NOUVEAU : Utilisation de GeoPandas pour une détection précise
#   des continents.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import fonctions as f 
import lib as lib



# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (‑20 °C)
dt = 1800.0  # pas de temps : 30 min
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo de surface (inchangé)
# ────────────────────────────────────────────────


SHAPEFILE_PATH = (
    pathlib.Path("ressources/map") / "ne_110m_admin_0_countries.shp"
)
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = f.load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    print("La simulation ne peut pas continuer sans les données d'albédo.")
    exit()




# --- Création de la fonction de recherche au démarrage ---
continent_finder = f.create_continent_finder(SHAPEFILE_PATH)

# --- Bilan thermodynamique ---
def f_rhs(T, phinet, C, q_latent):
    return (phinet - q_latent + lib.P_em_atm_thermal(Tatm) -lib.P_em_surf_thermal(T)) / C


# ────────────────────────────────────────────────
# Intégrateur Backward‑Euler (légèrement modifié)
# ────────────────────────────────────────────────


def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    albedo_sol_hist, albedo_nuages_hist, C_hist = (np.empty(N + 1) for _ in range(3))
    T[0] = T0
    lat_rad, lat_idx, lon_idx = np.radians(lat_deg), _lat_idx(lat_deg), _lon_idx(lon_deg)

    # MODIFIÉ : Appel de la nouvelle fonction pour obtenir Q
    q_latent_base = lib.P_em_surf_evap(lat_deg, lon_deg)

    print("Lissage des données annuelles par convolution gaussienne...")
    albedo_sol_mensuel_loc = monthly_albedo_sol[:, lat_idx, lon_idx]
    albedo_sol_journalier_lisse = f.lisser_donnees_annuelles(albedo_sol_mensuel_loc, sigma=15.0)
    albedo_nuages_mensuel = f.load_monthly_cloud_albedo_mock(lat_deg, lon_deg)
    albedo_nuages_journalier_lisse = f.lisser_donnees_annuelles(albedo_nuages_mensuel, sigma=15.0)
    v_capacite = np.vectorize(f.capacite_thermique_massique)
    cap_massique_mensuelle = v_capacite(albedo_sol_mensuel_loc) * 1000.0
    cap_surfacique_mensuelle = cap_massique_mensuelle * MASSE_SURFACIQUE_ACTIVE
    C_journalier_lisse = f.lisser_donnees_annuelles(cap_surfacique_mensuelle, sigma=15.0)

    albedo_sol_hist[0], albedo_nuages_hist[0], C_hist[0] = (
        albedo_sol_journalier_lisse[0], albedo_nuages_journalier_lisse[0], C_journalier_lisse[0]
    )

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        jour_dans_annee = (jour - 1) % 365

        albedo_sol = albedo_sol_journalier_lisse[jour_dans_annee]
        albedo_nuages = albedo_nuages_journalier_lisse[jour_dans_annee]
        C = C_journalier_lisse[jour_dans_annee]

        albedo_sol_hist[k + 1], albedo_nuages_hist[k + 1], C_hist[k + 1] = albedo_sol, albedo_nuages, C

        phi_n = lib.P_inc_solar(lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages)
        q_latent_step = q_latent_base if phi_n > 0 else -q_latent_base

        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C, q_latent_step)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6: break
        T[k + 1] = X

    return T, albedo_sol_hist, albedo_nuages_hist, C_hist


# ────────────────────────────────────────────────
# Fonctions de tracé et exécution principale (inchangées)
# ────────────────────────────────────────────────


def tracer_comparaison(times, T, albedo_sol_hist, albedo_nuages_hist, C_hist, titre, jour_a_afficher):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True, height_ratios=[2, 1])
    days_axis = times / 86400
    axs[0].plot(days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8, label="Simulation Année 2")
    steps_per_day = int(24 * 3600 / dt)
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)
    start_idx = min(start_idx, end_idx)
    axs[0].plot(days_axis[start_idx:end_idx + 1], T[start_idx:end_idx + 1] - 273.15, lw=2.5, color="firebrick", label=f"Jour n°{jour_a_afficher}")
    axs[0].set_ylabel("Température surface (°C)"); axs[0].set_title(titre); axs[0].grid(ls=":"); axs[0].legend(); axs[0].set_xlim(0, 365)
    color1 = "tab:blue"; axs[1].set_ylabel("Albédo (sans unité)", color=color1)
    axs[1].plot(days_axis, albedo_sol_hist, color=color1, lw=2.0, label="Albédo Sol (A2)")
    axs[1].plot(days_axis, albedo_nuages_hist, color="cyan", lw=2.0, ls=":", label="Albédo Nuages (A1)")
    axs[1].tick_params(axis="y", labelcolor=color1); axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5)); axs[1].legend(loc="upper left")
    ax2 = axs[1].twinx(); color2 = "tab:red"; ax2.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color=color2)
    ax2.plot(days_axis, C_hist, color=color2, lw=2.0, ls="--", label="Capacité (droite)"); ax2.tick_params(axis="y", labelcolor=color2)
    axs[1].set_xlabel("Jour de l'année (simulation stabilisée)"); axs[1].grid(ls=":"); fig.tight_layout(); plt.show()


if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182

    #Pour Paris (Europe)
    lat_sim, lon_sim = 48.85, 2.35


    print(f"Lancement de la simulation pour Lat={lat_sim}N, Lon={lon_sim}E...")
    T_full, alb_sol_full, alb_nuages_full, C_full = backward_euler(jours_de_simulation, lat_sim, lon_sim)

    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt
    T_yr2, alb_sol_yr2, alb_nuages_yr2, C_yr2 = (
        arr[steps_per_year:] for arr in [T_full, alb_sol_full, alb_nuages_full, C_full]
    )

    tracer_comparaison(
        t_yr2_plot, T_yr2, alb_sol_yr2, alb_nuages_yr2, C_yr2,
        f"Simulation stabilisée (avec chaleur latente via GeoPandas) pour Lat={lat_sim}, Lon={lon_sim}",
        jour_a_afficher,
    )
