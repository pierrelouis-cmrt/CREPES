# modele_courbe.py
# ==============================================================================
# SCRIPT PRINCIPAL POUR LA SIMULATION DU MODÈLE THERMIQUE DE SURFACE
# Rôle : Orchestre la simulation en définissant les paramètres, en appelant
#        les fonctions de préparation des données, en exécutant l'intégrateur
#        temporel et en affichant les résultats.
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys

# --- Import des modules locaux ---
# fonctions.py : pour la préparation des données et les calculs spatio-temporels
# lib.py : pour les constantes et équations physiques fondamentales
import fonctions as f
import lib as lib


# ────────────────────────────────────────────────
# BILAN DE FLUX THERMIQUE
# ────────────────────────────────────────────────
def f_rhs(T, phinet, C, q_latent):
    """
    Calcule la partie droite de l'équation différentielle (dT/dt).

    Cette fonction représente le bilan énergétique à la surface.

    IN:
        T (float): Température de surface actuelle [K].
        phinet (float): Flux solaire net absorbé par la surface [W m⁻²].
        C (float): Capacité thermique surfacique [J m⁻² K⁻¹].
        q_latent (float): Flux de chaleur latente [W m⁻²].

    OUT:
        float: La dérivée de la température par rapport au temps, dT/dt [K s⁻¹].
    """
    # Bilan : (Solaire absorbé - Latent) + (IR atm entrant - IR surface sortant)
    return (
        phinet
        - q_latent
        + lib.P_em_atm_thermal(lib.Tatm)
        - lib.P_em_surf_thermal(T)
    ) / C


# ────────────────────────────────────────────────
# INTÉGRATEUR BACKWARD-EULER
# ────────────────────────────────────────────────
def backward_euler(
    days, T0, dt, lat_rad, lon_deg, sim_params
):
    """
    Intègre l'équation de température sur le temps via la méthode Backward-Euler.

    Cette méthode implicite est résolue numériquement avec la méthode de Newton.

    IN:
        days (int): Nombre total de jours à simuler.
        T0 (float): Température initiale de la surface [K].
        dt (float): Pas de temps de la simulation [s].
        lat_rad (float): Latitude du point de simulation [radians].
        lon_deg (float): Longitude du point de simulation [degrés].
        sim_params (dict): Dictionnaire contenant les séries temporelles
                           pré-calculées (albédo, flux latent, etc.).

    OUT:
        tuple: Contient les séries temporelles des résultats de la simulation
               (Température, albédos, capacité, flux latents).
    """
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    T[0] = T0

    # Unpack les paramètres pré-calculés pour plus de lisibilité
    C_const = sim_params["C"]
    q_base = sim_params["q_base"]
    alb_sol_daily = sim_params["albedo_sol_daily"]
    alb_nuages_daily = sim_params["albedo_nuages_daily"]
    q_latent_smoothed = sim_params["q_latent_smoothed"]

    # Initialisation des tableaux pour stocker l'historique des variables
    hist_shape = (np.empty(N + 1) for _ in range(5))
    (
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
    ) = hist_shape

    # Stockage des valeurs initiales (k=0)
    albedo_sol_hist[0] = alb_sol_daily[0]
    albedo_nuages_hist[0] = alb_nuages_daily[0]
    C_hist[0] = C_const
    q_latent_hist[0] = q_base
    q_latent_step_hist[0] = q_latent_smoothed[0]

    # --- Boucle principale de la simulation ---
    for k in range(N):
        t_sec = k * dt
        day_of_year, heure_solaire = f.get_time_variables(t_sec, lon_deg)

        # Récupération des paramètres du jour depuis les tableaux pré-calculés
        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[k]

        # Calcul du flux solaire incident net pour ce pas de temps
        phi_n = lib.P_inc_solar(
            lat_rad, day_of_year + 1, heure_solaire, albedo_sol, albedo_nuages
        )

        # --- Résolution implicite par la méthode de Newton ---
        # On cherche X = T[k+1] tel que F(X) = 0
        # F(X) = X - T[k] - dt * f_rhs(X, ...)
        X = T[k]  # Estimation initiale
        for _ in range(8):  # 8 itérations suffisent largement
            F = X - T[k] - dt * f_rhs(X, phi_n, C_const, q_latent_step)
            # Dérivée de F par rapport à X : dF/dX = 1 - dt * d(f_rhs)/dX
            dF = 1.0 - dt * (-4.0 * lib.sigma * X**3 / C_const)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

        # Stockage des valeurs pour l'historique
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
# FONCTION DE TRACÉ
# ────────────────────────────────────────────────
def tracer_comparaison(
    times,
    T,
    albedo_sol_hist,
    albedo_nuages_hist,
    C_hist,
    q_latent_step_hist,
    titre,
    jour_a_afficher,
    sigma_plot=3.0,
):
    """
    Trace les résultats de la simulation sur plusieurs graphiques.

    IN:
        times (np.ndarray): Tableau des temps en secondes.
        T (np.ndarray): Série temporelle des températures [K].
        albedo_sol_hist (np.ndarray): Série temporelle de l'albédo du sol.
        albedo_nuages_hist (np.ndarray): Série temporelle de l'albédo des nuages.
        C_hist (np.ndarray): Série temporelle de la capacité thermique [J m⁻² K⁻¹].
        q_latent_step_hist (np.ndarray): Série temporelle du flux latent [W m⁻²].
        titre (str): Titre principal du graphique.
        jour_a_afficher (int): Jour spécifique à mettre en évidence sur le tracé.
        sigma_plot (float): Lissage gaussien pour l'affichage du flux latent.
    """
    fig, axs = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True, height_ratios=[3, 2, 2]
    )
    days_axis = times / 86400
    steps_per_day = int(24 * 3600 / lib.dt)

    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)

    # --- Axe 1: Température ---
    axs[0].plot(
        days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8, label="Année 2"
    )
    axs[0].plot(
        days_axis[start_idx : end_idx + 1],
        T[start_idx : end_idx + 1] - 273.15,
        lw=2.5,
        color="firebrick",
        label=f"Jour n°{jour_a_afficher}",
    )
    axs[0].set_ylabel("Température surface (°C)", fontsize=14)
    axs[0].set_title(titre, fontsize=16)
    axs[0].grid(ls=":")
    axs[0].legend(fontsize=12)
    axs[0].set_xlim(0, 365)

    # --- Axe 2: Albédos ---
    axs[1].set_ylabel("Albédo (sans unité)", fontsize=14)
    axs[1].plot(
        days_axis, albedo_sol_hist, color="tab:blue", lw=2.0, label="Sol (A2)"
    )
    axs[1].plot(
        days_axis,
        albedo_nuages_hist,
        color="cyan",
        lw=2.0,
        ls=":",
        label="Nuages (A1)",
    )
    axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5))
    axs[1].legend(loc="upper left", fontsize=12)
    axs[1].grid(ls=":")

    # --- Axe 3: Flux Latent et Capacité Thermique ---
    q_plot = f.gaussian_filter1d(
        q_latent_step_hist, sigma=sigma_plot, mode="wrap"
    )
    axs[2].plot(
        days_axis,
        q_plot,
        color="tab:green",
        lw=1.5,
        alpha=0.8,
        label="Flux Latent lissé (Q)",
    )
    axs[2].set_ylabel("Flux Chaleur Latente (W m⁻²)", fontsize=14)
    axs[2].legend(loc="upper left", fontsize=12)
    axs[2].grid(ls=":")

    ax3 = axs[2].twinx()
    ax3.set_ylabel(
        "Capacité Surfacique (J m⁻² K⁻¹)", color="tab:red", fontsize=14
    )
    ax3.plot(
        days_axis, C_hist, color="tab:red", lw=2.0, ls="--", label="Capacité"
    )
    ax3.tick_params(axis="y", labelcolor="tab:red")
    ax3.legend(loc="upper right", fontsize=12)

    axs[2].set_xlabel(
        "Jour de l'année (simulation stabilisée)", fontsize=14
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()


# ────────────────────────────────────────────────
# EXÉCUTION PRINCIPALE
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # --- Paramètres de la simulation ---
    jours_de_simulation = 365 * 2  # 2 ans pour atteindre un régime stable
    jour_a_afficher = 182  # Jour à mettre en évidence (solstice d'été approx)
    T_initial = 288.0  # Température de départ [K]

    # --- Choix des coordonnées ---
    # Paris (Europe)
    lat_sim, lon_sim = 48.5, 2.3
    # Amazonie (Amérique du Sud, Q élevé)
    # lat_sim, lon_sim = -3.46, -62.21
    # Sahara (Afrique, Q modéré, Cp faible)
    # lat_sim, lon_sim = 25.0, 15.0
    # Océan Arctique (Pôle Nord)
    # lat_sim, lon_sim = 82.0, 135.0
    # Antarctique (Pôle Sud)
    # lat_sim, lon_sim = -76.0, 100.0

    print(
        f"Lancement de la simulation pour Lat={lat_sim}°N, Lon={lon_sim}°E..."
    )

    # 1. Préparation de toutes les données d'entrée (albédo, Q, C, etc.)
    try:
        sim_params = f.prepare_simulation_inputs(
            lat_deg=lat_sim,
            lon_deg=lon_sim,
            total_days=jours_de_simulation,
            dt=lib.dt,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERREUR CRITIQUE lors de la préparation des données: {e}")
        sys.exit(1)

    # 2. Lancement de l'intégrateur numérique avec les paramètres préparés
    (
        T_full,
        alb_sol_full,
        alb_nuages_full,
        C_full,
        q_latent_full,
        q_latent_step_full,
    ) = backward_euler(
        jours_de_simulation,
        T_initial,
        lib.dt,
        np.radians(lat_sim),
        lon_sim,
        sim_params,
    )

    # 3. Extraction des données de la deuxième année pour l'affichage
    # (on ignore la première année de "warm-up")
    steps_per_year = int(365 * 24 * 3600 / lib.dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * lib.dt
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

    # 4. Affichage des résultats
    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_sol_yr2,
        alb_nuages_yr2,
        C_yr2,
        q_latent_step_yr2,
        f"Simulation pour Lat={lat_sim}°N, Lon={lon_sim}°E",
        jour_a_afficher,
    )