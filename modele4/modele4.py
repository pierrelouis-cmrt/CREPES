# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# FUSION:
# - Flux solaire calculé avec la déclinaison saisonnière (Script 1)
# - Albédo et Capacité Thermique variables basés sur des
#   données géographiques mensuelles (Script 2)
# - Ajout d'un tracé pour l'albédo et la capacité thermique
#
# AMÉLIORATIONS:
# - Simulation sur 2 ans pour stabilisation (spin-up)
# - Affichage des résultats de la 2ème année uniquement
# - Mise en évidence d'un jour spécifique sur le graphique de température
#
# REMANIEMENT:
# - Suppression de la logique basée sur l'UTC
# - Calcul direct de l'heure solaire locale en fonction de la longitude
#   → plus de décalage horaire explicite à gérer
# ---------------------------------------------------------------
 

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
import lib as lib 
import fonctions as f

# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 253.15  # atmosphère radiative (‑20 °C)
dt = 1800.0  # pas de temps : 30 min

# Masse de la couche de surface active thermiquement (kg m-2)
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo mensuel (depuis Script 2)
# ────────────────────────────────────────────────



# --- Chargement des données au démarrage ---
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo, LAT, LON = f.load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    print("La simulation ne peut pas continuer sans les données d'albédo.")
    exit()




# ---------- RHS de l’EDO ----------

def f_rhs(T, phinet, C):
    return (phinet + lib.P_abs_atm_thermal(Tatm) - lib.P_em_surf_thermal(T)) / C


# ---------- intégrateur Backward‑Euler ----------

def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    """
    Intègre la température de surface et retourne l'historique de T,
    de l'albédo et de la capacité thermique C.

    REMARQUE : le calcul d'heure se fait désormais en heure solaire locale
    (HSL) directement dérivée de la longitude ; il n'existe plus de notion
    explicite d'UTC ou de décalage horaire.
    """
    N = int(days * 24 * 3600 / dt)
    times = np.arange(N + 1) * dt
    T = np.empty(N + 1)
    albedo_hist = np.empty(N + 1)
    C_hist = np.empty(N + 1)

    T[0] = T0
    lat_rad = np.radians(lat_deg)
    lat_idx = _lat_idx(lat_deg)
    lon_idx = _lon_idx(lon_deg)

    # --- Calcul des valeurs initiales (k=0) ---
    jour_init = 1
    mois_init = 1
    albedo_hist[0] = monthly_albedo[mois_init - 1, lat_idx, lon_idx]
    c_massique_j_init = f.capacite_thermique_massique(albedo_hist[0]) * 1000.0
    C_hist[0] = c_massique_j_init * MASSE_SURFACIQUE_ACTIVE

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1

        # Heure solaire locale (HSL) : temps écoulé + correction de longitude
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0

        # Le mois est cyclique sur 12 mois
        jour_dans_annee = (jour - 1) % 365
        mois = int(jour_dans_annee / 30.4) + 1
        mois = min(max(mois, 1), 12)

        albedo = monthly_albedo[mois - 1, lat_idx, lon_idx]
        c_massique_j = f.capacite_thermique_massique(albedo) * 1000.0
        C = c_massique_j * MASSE_SURFACIQUE_ACTIVE

        albedo_hist[k + 1] = albedo
        C_hist[k + 1] = C

        phi_n = lib.P_inc_solar(lat_rad, jour, heure_solaire, albedo)

        # Itération de Newton-Raphson pour résoudre l'équation implicite
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return times, T, albedo_hist, C_hist


# ---------- tracé ----------

def tracer_comparaison(
    times, T, albedo_hist, C_hist, titre, jour_a_afficher
):
    """
    Crée une figure avec deux sous-graphiques.
    - Haut: Température de l'année, avec un jour spécifique mis en évidence.
    - Bas: Albédo et Capacité thermique.
    """
    fig, axs = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, height_ratios=[2, 1]
    )
    days_axis = times / 86400

    # --- Graphique du haut : Température ---
    axs[0].plot(
        days_axis,
        T - 273.15,
        lw=1.0,
        color="gray",
        alpha=0.8,
        label="Simulation Année 2",
    )

    # Mise en évidence du jour choisi
    steps_per_day = int(24 * 3600 / dt)
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = jour_a_afficher * steps_per_day
    end_idx = min(end_idx, len(days_axis) - 1)
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

    # --- Graphique du bas : Albédo et Capacité ---
    color1 = "tab:blue"
    axs[1].set_ylabel("Albédo (sans unité)", color=color1)
    axs[1].plot(days_axis, albedo_hist, color=color1, lw=1.5)
    axs[1].tick_params(axis="y", labelcolor=color1)
    axs[1].set_ylim(0, 1)

    ax2 = axs[1].twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color=color2)
    ax2.plot(days_axis, C_hist, color=color2, lw=1.5, ls="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    axs[1].set_xlabel("Jour de l'année (simulation stabilisée)")
    axs[1].grid(ls=":")

    fig.tight_layout()
    plt.show()


# ---------- exécution ----------
if __name__ == "__main__":
    # --- Paramètres de la simulation ---
    jours_de_simulation = 365 * 2  # 2 ans pour spin-up
    jour_a_afficher = 182  # 1er juillet (approx.)

    # --- Simulation pour Pole Nord ---
    lat_Paris, lon_Paris = 48.866667 , 2.333333
    print("Lancement de la simulation pour Pole Nord...")
    t_full, T_full, alb_full, C_full = backward_euler(
        jours_de_simulation, lat_Paris, lon_Paris
    )

    # Extraction de la deuxième année
    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2 = t_full[steps_per_year:]
    T_yr2 = T_full[steps_per_year:]
    alb_yr2 = alb_full[steps_per_year:]
    C_yr2 = C_full[steps_per_year:]
    t_yr2_plot = t_yr2 - t_yr2[0]

    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_yr2,
        C_yr2,
        f"Simulation stabilisée pour Paris (Lat={lat_Paris}, Lon={lon_Paris})",
        jour_a_afficher,
    )


