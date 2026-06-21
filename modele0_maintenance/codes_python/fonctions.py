"""Fonctions utilitaires publiques du moteur CREPES.

Ce fichier conserve les noms du groupe Carcajous, mais delegue les grandeurs
physiques a `physique/` pour rendre la structure plus lisible.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from chemins import ALBEDO_DIR
from physique import albedo as albedo_module
from physique import capacite_surface
from physique import chaleur_latente as latent_module
from physique import solaire

# Reexports Carcajous conserves.
get_time_variables = solaire.get_time_variables
declination = solaire.declination
cos_incidence = solaire.cos_incidence
compute_cp_from_rzsm = capacite_surface.compute_cp_from_rzsm
load_and_grid_rzsm_data = capacite_surface.load_and_grid_rzsm_data
lisser_donnees_annuelles = albedo_module.lisser_donnees_annuelles
load_albedo_series = albedo_module.load_albedo_series
load_monthly_cloud_albedo_from_ceres = albedo_module.load_monthly_cloud_albedo_from_ceres
create_continent_finder = latent_module.create_continent_finder
continent_finder = latent_module.continent_finder

# Ces constantes restent exposées ici pour les anciens notebooks et scripts.
RHO_W = capacite_surface.RHO_W
RHO_BULK = capacite_surface.RHO_BULK
CP_SEC = capacite_surface.CP_SEC
CP_WATER = capacite_surface.CP_WATER
CP_ICE = capacite_surface.CP_ICE


def prepare_simulation_inputs(lat_deg, lon_deg, total_days, dt, sigma_q=3.0):
    """Charge et prepare les series necessaires a une simulation ponctuelle.

    La base est Carcajous modele 4: capacite RZSM, albedo sol mensuel, albedo
    nuage CERES et flux latent par continent. Le fallback de capacite
    Ornithorynquietant est utilise seulement si RZSM manque localement.
    """
    print("--- Preparation des parametres de simulation ---")

    # Capacité, albédo et flux latent sont préparés une fois avant l'intégration.
    C_const, capacity_source = capacite_surface.compute_surface_capacity(lat_deg, lon_deg)
    print(
        f"Capacite thermique: {C_const:.2e} J m-2 K-1 "
        f"({capacity_source})"
    )

    monthly_albedo_sol, latitudes, longitudes = load_albedo_series(ALBEDO_DIR)
    # On reprend simplement la maille la plus proche des données Carcajous.
    lat_idx = lambda lat: int(np.abs(latitudes - lat).argmin())
    lon_idx = lambda lon: int(
        np.abs(longitudes - (((lon + 180) % 360) - 180)).argmin()
    )
    albedo_sol_m_loc = monthly_albedo_sol[:, lat_idx(lat_deg), lon_idx(lon_deg)]
    alb_sol_daily = lisser_donnees_annuelles(albedo_sol_m_loc, sigma=15.0)

    alb_nuages_m = load_monthly_cloud_albedo_from_ceres(lat_deg, lon_deg)
    alb_nuages_daily = lisser_donnees_annuelles(alb_nuages_m, sigma=15.0)

    q_base = latent_module.P_em_surf_evap(lat_deg, lon_deg, verbose=True)
    step_count = int(total_days * 24 * 3600 / dt)
    sign_daynight = np.empty(step_count)
    lat_rad = np.radians(lat_deg)

    # Le flux latent historique est un flux moyen, modulé ici selon jour/nuit.
    for index in range(step_count):
        t_sec = index * dt
        jour, heure_solaire = get_time_variables(t_sec, lon_deg)
        sign_daynight[index] = (
            1.0 if cos_incidence(lat_rad, jour + 1, heure_solaire) > 0 else -1.0
        )

    q_latent_raw = q_base * sign_daynight
    q_latent_smoothed = gaussian_filter1d(q_latent_raw, sigma=sigma_q, mode="wrap")
    print("--- Preparation terminee ---")

    return {
        "C": C_const,
        "capacity_source": capacity_source,
        "q_base": q_base,
        "albedo_sol_daily": alb_sol_daily,
        "albedo_nuages_daily": alb_nuages_daily,
        "q_latent_smoothed": q_latent_smoothed,
    }
