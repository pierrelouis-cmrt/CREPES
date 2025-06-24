# ---------------------------------------------------------------
# Modèle 0‑D couplé surface‑atmosphère – Backward‑Euler implicite  
#  
# VERSION MISE À JOUR (v2) :  
# • Albédo nuages (A1) et albédo surface (A2) lissés (déjà présent).  
# • **NOUVEAU :** l'atmosphère radiative n'est plus figée ; on résout  
#   désormais un système de deux équations différentielles :  
#       C_surf dT/dt   = Φ_net + σ T_atm⁴ − σ T⁴  
#       C_air  dT_atm/dt = σ T⁴ − 2 σ T_atm⁴  
#   avec C_air = 1 763 775 J m⁻² K⁻¹ et T_atm⁰ = −50 °C (223.15 K).  
# • Schéma Backward‑Euler pleinement implicite sur (T, T_atm) :  
#   résolution Newton–Raphson 2×2 à chaque pas de temps.  
# ---------------------------------------------------------------

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ---------- constantes physiques ----------
constante_solaire = 1361.0            # W m⁻²
sigma = 5.670374419e-8                # Stefan‑Boltzmann (SI)
C_AIR = 1_763_775.0                   # J m⁻² K⁻¹ (capacité surfacique de l'air)
TATM0 = 223.15                        # K  (‑50 °C)
dt = 1800.0                           # pas de temps : 30 min
MASSE_SURFACIQUE_ACTIVE = 4.0e2       # kg m⁻² (couche «active»)

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo de surface (Script 2)
# ────────────────────────────────────────────────

def load_albedo_series(csv_dir: str | pathlib.Path,
                       pattern: str = "albedo{:02d}.csv"):
    """Charge les 12 fichiers CSV d'albédo de surface mensuel."""
    csv_dir = pathlib.Path(csv_dir)
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    cubes: list[np.ndarray] = []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    print("Données d'albédo de surface chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes

# --- Chargement des données au démarrage ---
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo_sol, LAT, LON = load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(np.abs(LON - (((lon + 180) % 360) - 180)).argmin())
except FileNotFoundError:
    print("ERREUR : Le dossier 'ressources/albedo' est introuvable.")
    print("La simulation ne peut pas continuer sans les données d'albédo.")
    exit()

# ────────────────────────────────────────────────
# Albédo des nuages (mock ‑ substitut)
# ────────────────────────────────────────────────

def load_monthly_cloud_albedo_mock(lat_deg: float, lon_deg: float):
    """Profil annuel simulé d'albédo des nuages mensuel."""
    print("NOTE : Utilisation de données simulées pour l'albédo des nuages.")
    amplitude = 0.15 * np.sin(np.radians(abs(lat_deg)))
    avg_cloud_albedo = 0.3
    mois = np.arange(12)
    variation_saisonniere = amplitude * np.cos(2 * pi * (mois - 0.5) / 12)
    return avg_cloud_albedo - variation_saisonniere

# ────────────────────────────────────────────────
# Capacité thermique basée sur l'albédo (inchangé)
# ────────────────────────────────────────────────

_REF_ALBEDO = {
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

def capacite_thermique_massique(albedo: float) -> float:
    """Capacité thermique massique (kJ kg⁻¹ K⁻¹) en fonction de l'albédo."""
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[surf]

# ────────────────────────────────────────────────
# Lissage gaussien cyclique des données annuelles
# ────────────────────────────────────────────────

def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """Lisse 12 valeurs mensuelles → profil journalier continu (365 j)."""
    jours_par_mois = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    valeurs_journalieres_discontinues = np.repeat(valeurs_mensuelles, jours_par_mois)
    return gaussian_filter1d(valeurs_journalieres_discontinues, sigma=sigma, mode="wrap")

# ────────────────────────────────────────────────
# Fonctions astronomiques & radiatives
# ────────────────────────────────────────────────

def declination(day):
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)

def cos_incidence(lat_rad, day, hour):
    δ = declination(day)
    H = np.radians(15 * (hour - 12))
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)

def phi_net(lat_rad, day, hour, albedo_sol, albedo_nuages):
    """Flux solaire net absorbé par la surface."""
    phi_entrant = constante_solaire * cos_incidence(lat_rad, day, hour)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)

# ────────────────────────────────────────────────
# Intégrateur Backward‑Euler couplé (surface + air)
# ────────────────────────────────────────────────

def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    """Renvoie l'historique de T, T_atm, albédos et capacité."""
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    Tatm = np.empty(N + 1)
    albedo_sol_hist = np.empty(N + 1)
    albedo_nuages_hist = np.empty(N + 1)
    C_hist = np.empty(N + 1)

    T[0] = T0
    Tatm[0] = TATM0

    lat_rad = np.radians(lat_deg)
    lat_idx = _lat_idx(lat_deg)
    lon_idx = _lon_idx(lon_deg)

    # --- Pré‑calcul des profils lissés annuels ---
    print("Lissage des données annuelles par convolution gaussienne…")

    albedo_sol_mensuel_loc = monthly_albedo_sol[:, lat_idx, lon_idx]
    albedo_sol_journalier = lisser_donnees_annuelles(albedo_sol_mensuel_loc, sigma=15.0)

    albedo_nuages_mensuel = load_monthly_cloud_albedo_mock(lat_deg, lon_deg)
    albedo_nuages_journalier = lisser_donnees_annuelles(albedo_nuages_mensuel, sigma=15.0)

    v_capacite = np.vectorize(capacite_thermique_massique)
    cap_massique_mensuelle = v_capacite(albedo_sol_mensuel_loc) * 1000.0
    cap_surfacique_mensuelle = cap_massique_mensuelle * MASSE_SURFACIQUE_ACTIVE
    C_journalier = lisser_donnees_annuelles(cap_surfacique_mensuelle, sigma=15.0)

    # Historiques initiaux
    albedo_sol_hist[0] = albedo_sol_journalier[0]
    albedo_nuages_hist[0] = albedo_nuages_journalier[0]
    C_hist[0] = C_journalier[0]

    # --- Boucle temporelle ---
    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        jour_dans_annee = (jour - 1) % 365

        albedo_sol = albedo_sol_journalier[jour_dans_annee]
        albedo_nuages = albedo_nuages_journalier[jour_dans_annee]
        C = C_journalier[jour_dans_annee]

        albedo_sol_hist[k + 1] = albedo_sol
        albedo_nuages_hist[k + 1] = albedo_nuages
        C_hist[k + 1] = C

        phi_n = phi_net(lat_rad, jour, heure_solaire, albedo_sol, albedo_nuages)

        # -------- Newton–Raphson 2×2 (T, Tatm) implicite --------
        X = T[k]
        Y = Tatm[k]
        for _ in range(10):
            # Résidus
            B = 0.9
            F1 = X - T[k] - dt * ((phi_n + B*sigma * Y**4 - sigma * X**4) / C)
            F2 = Y - Tatm[k] - dt * ((B*sigma * X**4 - 2 * B*sigma * Y**4) / C_AIR)

            # Jacobien
            J11 = 1 - dt * (-4 * sigma * X**3) / C          # ∂F1/∂X
            J12 = - dt * (4 * sigma * Y**3) / C             # ∂F1/∂Y
            J21 = - dt * (4 * sigma * X**3) / C_AIR         # ∂F2/∂X
            J22 = 1 - dt * (-8 * sigma * Y**3) / C_AIR      # ∂F2/∂Y

            det = J11 * J22 - J12 * J21
            if abs(det) < 1e-12:
                raise ZeroDivisionError("Jacobian determinant ~ 0")

            dX = (F1 * J22 - F2 * J12) / det
            dY = (J11 * F2 - J21 * F1) / det

            X -= dX
            Y -= dY

            if max(abs(dX), abs(dY)) < 1e-6:
                break

        T[k + 1] = X
        Tatm[k + 1] = Y

    return T, Tatm, albedo_sol_hist, albedo_nuages_hist, C_hist

# ────────────────────────────────────────────────
# Fonctions de tracé (T & T_atm)
# ────────────────────────────────────────────────

def tracer_comparaison(times, T, Tatm, albedo_sol_hist, albedo_nuages_hist, C_hist,
                       titre, jour_a_afficher):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                            height_ratios=[2, 1])
    days_axis = times / 86400

    # --- Graphe supérieur : températures ---
    axs[0].plot(days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8,
                label="T surface (année 2)")
    axs[0].plot(days_axis, Tatm - 273.15, lw=1.0, color="skyblue", alpha=0.8,
                label="T atmosphère (année 2)")

    steps_per_day = int(24 * 3600 / dt)
    start_idx = (jour_a_afficher - 1) * steps_per_day
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)

    axs[0].plot(days_axis[start_idx:end_idx + 1],
                T[start_idx:end_idx + 1] - 273.15,
                lw=2.5, color="firebrick",
                label=f"T surf jour {jour_a_afficher}")

    axs[0].set_ylabel("Température (°C)")
    axs[0].set_title(titre)
    axs[0].grid(ls=":")
    axs[0].legend()
    axs[0].set_xlim(0, 365)

    # --- Graphe inférieur : albédos + capacité ---
    color1 = "tab:blue"
    axs[1].set_ylabel("Albédo (‑)", color=color1)
    axs[1].plot(days_axis, albedo_sol_hist, color=color1, lw=2.0,
                label="Albédo Sol (A2)")
    axs[1].plot(days_axis, albedo_nuages_hist, color="cyan", lw=2.0, ls=":",
                label="Albédo Nuages (A1)")
    axs[1].tick_params(axis="y", labelcolor=color1)
    axs[1].set_ylim(0, max(np.max(albedo_sol_hist) * 1.2, 0.5))
    axs[1].legend(loc="upper left")

    ax2 = axs[1].twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Capacité surfacique (J m⁻² K⁻¹)", color=color2)
    ax2.plot(days_axis, C_hist, color=color2, lw=2.0, ls="--",
             label="Capacité (dte)")
    ax2.tick_params(axis="y", labelcolor=color2)

    axs[1].set_xlabel("Jour de l'année (simulation stabilisée)")
    axs[1].grid(ls=":")
    fig.tight_layout()
    plt.show()

# ────────────────────────────────────────────────
# Exécution principale
# ────────────────────────────────────────────────

if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182  # ≈ solstice d'été

    # Exemple : Paris (48.85 N, 2.35 E) → modifier si besoin
    lat_sim, lon_sim = 48.85, 2.35

    print(f"Lancement de la simulation pour Lat={lat_sim} N, Lon={lon_sim} E…")

    T_full, Tatm_full, alb_sol_full, alb_nuages_full, C_full = backward_euler(
        jours_de_simulation, lat_sim, lon_sim)

    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt

    T_yr2 = T_full[steps_per_year:]
    Tatm_yr2 = Tatm_full[steps_per_year:]
    alb_sol_yr2 = alb_sol_full[steps_per_year:]
    alb_nuages_yr2 = alb_nuages_full[steps_per_year:]
    C_yr2 = C_full[steps_per_year:]

    tracer_comparaison(t_yr2_plot, T_yr2, Tatm_yr2,
                       alb_sol_yr2, alb_nuages_yr2, C_yr2,
                       titre=("Simulation stabilisée (surface + atmosphère) "
                              f"Lat={lat_sim}, Lon={lon_sim}"),
                       jour_a_afficher=jour_a_afficher)
