# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Backward-Euler implicite
#
# FUSION:
# - Flux solaire calculé avec la déclinaison saisonnière (Script 1)
# - Albédo et Capacité Thermique variables basés sur des
#   données géographiques mensuelles (Script 2)
# - Ajout d'un tracé pour l'albédo et la capacité thermique
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd

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


# ---------- aides astronomiques (depuis Script 1) ----------
def declination(day):
    """Retourne la déclinaison solaire (rad) pour le jour de l’année (1‑365)."""
    return np.radians(23.44) * np.sin(2 * pi * (284 + day) / 365)


def cos_incidence(lat_rad, day, hour):
    """Cosinus de l’angle d’incidence du rayonnement sur le plan local."""
    δ = declination(day)
    H = np.radians(15 * (hour - 12))
    ci = np.sin(lat_rad) * np.sin(δ) + np.cos(lat_rad) * np.cos(δ) * np.cos(H)
    return max(ci, 0.0)


def phi_net(lat_rad, day, hour, albedo):
    """Flux solaire net (W m‑2) reçu par la surface."""
    return constante_solaire * cos_incidence(lat_rad, day, hour) * (1 - albedo)


# ---------- RHS de l’EDO ----------
def f_rhs(T, phinet, C):
    return (phinet + sigma * Tatm**4 - sigma * T**4) / C


# ---------- intégrateur Backward‑Euler ----------
def backward_euler(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    """
    Intègre la température de surface et retourne l'historique de T,
    de l'albédo et de la capacité thermique C.
    """
    N = int(days * 24 * 3600 / dt)
    times = np.arange(N + 1) * dt
    T = np.empty(N + 1)
    # MODIFICATION: Création des tableaux pour stocker l'historique
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
    c_massique_j_init = capacite_thermique_massique(albedo_hist[0]) * 1000.0
    C_hist[0] = c_massique_j_init * MASSE_SURFACIQUE_ACTIVE

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        heure_utc = (t_sec % 86400) / 3600.0
        heure_locale = (heure_utc + lon_deg / 15.0) % 24.0

        mois = int((jour - 1) / 30.4) + 1
        mois = min(max(mois, 1), 12)

        albedo = monthly_albedo[mois - 1, lat_idx, lon_idx]
        c_massique_j = capacite_thermique_massique(albedo) * 1000.0
        C = c_massique_j * MASSE_SURFACIQUE_ACTIVE

        # MODIFICATION: Stockage des valeurs calculées
        albedo_hist[k + 1] = albedo
        C_hist[k + 1] = C

        phi_n = phi_net(lat_rad, jour, heure_locale, albedo)

        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs(X, phi_n, C)
            dF = 1.0 - dt * (-4.0 * sigma * X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    # MODIFICATION: Retourner les historiques
    return times, T, albedo_hist, C_hist


# ---------- tracé ----------
def tracer(times, T, albedo_hist, C_hist, titre):
    """
    MODIFICATION: Crée une figure avec deux sous-graphiques.
    - Haut: Température
    - Bas: Albédo et Capacité thermique (avec double axe Y)
    """
    # Crée une figure avec 2 lignes, 1 colonne de graphiques.
    # sharex=True lie les axes X des deux graphiques.
    fig, axs = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1]
    )
    days_axis = times / 86400

    # --- Graphique du haut : Température ---
    axs[0].plot(days_axis, T - 273.15, lw=1.5, color="black")
    axs[0].set_ylabel("Température surface (°C)")
    axs[0].set_title(titre)
    axs[0].grid(ls=":")

    # --- Graphique du bas : Albédo et Capacité ---
    # Axe Y de gauche pour l'albédo
    color1 = "tab:blue"
    axs[1].set_ylabel("Albédo (sans unité)", color=color1)
    axs[1].plot(days_axis, albedo_hist, color=color1, lw=1.5)
    axs[1].tick_params(axis="y", labelcolor=color1)
    axs[1].set_ylim(0, 1)  # L'albédo est toujours entre 0 et 1

    # Créer un deuxième axe Y qui partage le même axe X
    ax2 = axs[1].twinx()

    # Axe Y de droite pour la capacité thermique
    color2 = "tab:red"
    ax2.set_ylabel("Capacité Surfacique (J m⁻² K⁻¹)", color=color2)
    ax2.plot(days_axis, C_hist, color=color2, lw=1.5, ls="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    axs[1].set_xlabel("Jour")
    axs[1].grid(ls=":")

    fig.tight_layout()  # Ajuste automatiquement les graphiques
    plt.show()


# ---------- exécution test ----------
if __name__ == "__main__":
    # Paris, France
    lat_paris, lon_paris = 49, 2.3
    # MODIFICATION: Récupérer les 4 valeurs retournées
    t, T, alb, C = backward_euler(365, lat_paris, lon_paris)
    # MODIFICATION: Passer les nouvelles données au traceur
    tracer(
        t,
        T,
        alb,
        C,
        f"Simulation 365 jours pour Paris (Lat={lat_paris}, Lon={lon_paris})\n"
        + "Albédo et Capacité Thermique variables",
    )

    # Pôle Nord, Arctique 
    lat_pole_nord, lon_pole_nord = 90, 0
    t_pn, T_pn, alb_pn, C_pn = backward_euler(365, lat_pole_nord, lon_pole_nord)
    tracer(
        t_pn,
        T_pn,
        alb_pn,
        C_pn,
        f"Simulation 365 jours pour le Pôle Nord, Arctique (Lat={lat_pole_nord}, Lon={lon_pole_nord})\n"
        + "Albédo et Capacité Thermique variables",
    )