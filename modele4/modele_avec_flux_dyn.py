# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# ... (autres commentaires inchangés) ...
#
# LISSAGE (NOUVEAU):
# - Lissage des données mensuelles (albédo, capacité thermique)
#   par convolution gaussienne pour obtenir des variations journalières
#   continues et cycliques.
#   // CLARIFICATION // : Seules les données d'ENTRÉE sont lissées.
#   // La température, qui est le RÉSULTAT, n'est pas lissée.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (‑20 °C)
dt = 1800.0  # pas de temps : 30 min
MASSE_SURFACIQUE_ACTIVE = 4.0e2  # kg m-2

# ────────────────────────────────────────────────
# DATA – Chargement de l'albédo mensuel (depuis Script 2)
# ────────────────────────────────────────────────


def load_albedo_series(
    csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    """Charge les 12 fichiers CSV d'albédo mensuel."""
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
    print("Données d'albédo chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes


# --- Chargement des données au démarrage ---
try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo, LAT, LON = load_albedo_series(ALBEDO_DIR)
    _lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
    _lon_idx = lambda lon: int(
        np.abs(LON - (((lon + 180) % 360) - 180)).argmin()
    )
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    print("La simulation ne peut pas continuer sans les données d'albédo.")
    exit()


# ────────────────────────────────────────────────
# Capacité thermique basée sur l'albédo (depuis Script 2)
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
    """Retourne la capacité thermique massique (kJ kg-1 K-1) pour un albedo."""
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[surf]


# ────────────────────────────────────────────────
# Lissage des données annuelles par convolution gaussienne (NOUVEAU)
# ────────────────────────────────────────────────


def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """
    Lisse 12 valeurs mensuelles en un profil journalier continu (365 j)
    en utilisant une convolution gaussienne cyclique.
    // CLARIFICATION // : Cette fonction ne traite que les données
    // d'entrée (comme l'albédo), jamais la température.
    """
    jours_par_mois = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    valeurs_journalieres_discontinues = np.repeat(
        valeurs_mensuelles, jours_par_mois
    )
    valeurs_lissees = gaussian_filter1d(
        valeurs_journalieres_discontinues, sigma=sigma, mode="wrap"
    )
    return valeurs_lissees


def declination(day):
    day_in_year = (day - 1) % 365 + 1
    return np.radians(23.44) * np.sin(2 * pi * (284 + day_in_year) / 365)


def cos_incidence(lat_rad, day, hour):
    δ = declination(day)
    H = np.radians(15 * (hour - 12))
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


def phi_net(lat_rad, day, hour, albedo):
    return constante_solaire * cos_incidence(lat_rad, day, hour) * (1 - albedo)


def f_rhs(T, phinet, C):
    return (phinet + sigma * Tatm**4 - sigma * T**4) / C


# ---------- intégrateur Backward‑Euler ----------


def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    """
    Intègre la température de surface et retourne l'historique de T,
    de l'albédo et de la capacité thermique C.
    """
    N = int(days * 24 * 3600 / dt)
    T = np.empty(N + 1)
    albedo_hist = np.empty(N + 1)
    C_hist = np.empty(N + 1)

    T[0] = T0
    lat_rad = np.radians(lat_deg)
    lat_idx = _lat_idx(lat_deg)
    lon_idx = _lon_idx(lon_deg)

    # --- Pré-calcul des profils annuels lissés ---
    print("Lissage des données annuelles par convolution gaussienne...")
    albedo_mensuel_loc = monthly_albedo[:, lat_idx, lon_idx]
    # // CLARIFICATION // : L'albédo est lissé ici, AVANT la simulation.
    albedo_journalier_lisse = lisser_donnees_annuelles(
        albedo_mensuel_loc, sigma=15.0
    )

    # --- Lissage de la capacité surfacique (input)
    v_capacite = np.vectorize(capacite_thermique_massique)
    # capacité massique mensuelle [kJ·kg⁻¹·K⁻¹] → [J·kg⁻¹·K⁻¹]
    cap_massique_mensuelle = v_capacite(albedo_mensuel_loc) * 1000.0
    # capacité surfacique mensuelle [J·m⁻²·K⁻¹]
    cap_surfacique_mensuelle = cap_massique_mensuelle * MASSE_SURFACIQUE_ACTIVE
    # lissage cyclique sur 365 jours
    C_journalier_lisse = lisser_donnees_annuelles(
        cap_surfacique_mensuelle,
        sigma=15.0
    )

    albedo_hist[0] = albedo_journalier_lisse[0]
    C_hist[0] = C_journalier_lisse[0]

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
        jour_dans_annee = (jour - 1) % 365

        # // CLARIFICATION // : On utilise les données lissées comme entrées.
        albedo = albedo_journalier_lisse[jour_dans_annee]
        C = C_journalier_lisse[jour_dans_annee]

        albedo_hist[k + 1] = albedo
        C_hist[k + 1] = C

        phi_n = phi_net(lat_rad, jour, heure_solaire, albedo)

        # // CLARIFICATION // : Ici, la température T est CALCULÉE par le
        # // solveur de Newton-Raphson. Elle n'est PAS lissée. Sa douceur
        # // est une CONSÉQUENCE des entrées lissées (albédo, C).
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return T, albedo_hist, C_hist


# ... (fonctions de tracé et exécution inchangées) ...
def tracer_comparaison(
    times, T, albedo_hist, C_hist, titre, jour_a_afficher
):
    fig, axs = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, height_ratios=[2, 1]
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
    color1 = "tab:blue"
    axs[1].set_ylabel("Albédo (sans unité)", color=color1)
    axs[1].plot(days_axis, albedo_hist, color=color1, lw=2.0)
    axs[1].tick_params(axis="y", labelcolor=color1)
    axs[1].set_ylim(0, max(np.max(albedo_hist) * 1.1, 0.3))
    ax2 = axs[1].twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color=color2)
    ax2.plot(days_axis, C_hist, color=color2, lw=2.0, ls="--")
    ax2.tick_params(axis="y", labelcolor=color2)
    axs[1].set_xlabel("Jour de l'année (simulation stabilisée)")
    axs[1].grid(ls=":")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    jours_de_simulation = 365 * 2
    jour_a_afficher = 182
    lat_paris, lon_paris = 40.4, 74
    print(f"Lancement de la simulation pour Paris ({lat_paris}N, {lon_paris}E)...")
    T_full, alb_full, C_full = backward_euler(
        jours_de_simulation, lat_paris, lon_paris
    )
    steps_per_year = int(365 * 24 * 3600 / dt)
    t_yr2_plot = np.arange(len(T_full) - steps_per_year) * dt
    T_yr2 = T_full[steps_per_year:]
    alb_yr2 = alb_full[steps_per_year:]
    C_yr2 = C_full[steps_per_year:]
    tracer_comparaison(
        t_yr2_plot,
        T_yr2,
        alb_yr2,
        C_yr2,
        f"Simulation stabilisée pour Paris (Lat={lat_paris}, Lon={lon_paris})",
        jour_a_afficher,
    )