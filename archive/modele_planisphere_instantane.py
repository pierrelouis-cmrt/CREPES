# ---------------------------------------------------------------
# Modèle 0-D de température de surface – Planisphère (Version Instantanée)
#
# DESCRIPTION :
# - Calcule la température d'ÉQUILIBRE RADIATIF INSTANTANÉ pour chaque
#   point du globe à un jour et une heure donnés.
# - Le terme de capacité thermique (stockage/restitution de chaleur) est ignoré.
# - L'exécution est très rapide, idéale pour tester l'effet des paramètres.
# - Utilise les données d'albédo lissées pour le jour choisi.
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import pathlib
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

# Optionnel : utiliser cartopy pour un meilleur rendu des côtes
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    USE_CARTOPY = True
except ImportError:
    USE_CARTOPY = False

# ---------- PARAMÈTRES DU CALCUL INSTANTANÉ ----------
JOUR_A_CALCULER = 182  # Jour de l'année (1-365), ex: 182 ~ solstice d'été
HEURE_UTC_A_CALCULER = 14.0  # Heure UTC (0-24), ex: 14.0 pour le début d'après-midi en Europe

# ---------- constantes physiques ----------
constante_solaire = 1361.0  # W m-2
sigma = 5.670374419e-8  # Stefan‑Boltzmann (SI)
Tatm = 223.15  # atmosphère radiative (-50 °C)

# (Les fonctions load_albedo_series, lisser_donnees_annuelles,
# declination, cos_incidence, phi_net sont identiques à la version complète)

def load_albedo_series(
    csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"
):
    csv_dir = pathlib.Path(csv_dir)
    latitudes, longitudes, cubes = None, None, []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    print("Données d'albédo chargées.")
    return np.stack(cubes, axis=0), latitudes, longitudes

try:
    ALBEDO_DIR = pathlib.Path("ressources/albedo")
    monthly_albedo, LAT, LON = load_albedo_series(ALBEDO_DIR)
    NLAT, NLON = len(LAT), len(LON)
except FileNotFoundError:
    print("ERREUR: Le dossier 'ressources/albedo' est introuvable.")
    exit()

def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    jours_par_mois = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    valeurs_journalieres_discontinues = np.repeat(valeurs_mensuelles, jours_par_mois)
    return gaussian_filter1d(valeurs_journalieres_discontinues, sigma=sigma, mode="wrap")

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


# ────────────────────────────────────────────────
# Calculateur de température instantanée
# ────────────────────────────────────────────────

def calculate_snapshot_temperature(jour, heure_utc):
    """Calcule la grille de température pour un instant donné."""
    print(f"Calcul de la température d'équilibre pour le jour {jour}, heure {heure_utc} UTC...")
    T_grid = np.zeros((NLAT, NLON))
    jour_dans_annee = (jour - 1) % 365

    for i in tqdm(range(NLAT), desc="Progression (latitude)"):
        for j in range(NLON):
            lat, lon = LAT[i], LON[j]

            # Lissage de l'albédo pour ce point
            albedo_mensuel_loc = monthly_albedo[:, i, j]
            albedo_journalier = lisser_donnees_annuelles(
                albedo_mensuel_loc, sigma=15.0
            )
            albedo_instantane = albedo_journalier[jour_dans_annee]

            # Calcul de l'heure solaire locale
            heure_solaire = (heure_utc + lon / 15.0) % 24.0

            # Calcul du flux net
            phi_n = phi_net(
                np.radians(lat), jour, heure_solaire, albedo_instantane
            )

            # Calcul de la température d'équilibre radiatif
            # T = ((phi_net + sigma * Tatm^4) / sigma)^(1/4)
            if phi_n > 0: # Côté jour
                temp_rad_term = (phi_n + sigma * Tatm**4) / sigma
                T_grid[i, j] = temp_rad_term**0.25
            else: # Côté nuit (phi_net = 0)
                T_grid[i, j] = Tatm

    return T_grid


# ────────────────────────────────────────────────
# Exécution principale et tracé statique
# ────────────────────────────────────────────────
if __name__ == "__main__":
    T_grid_snapshot = calculate_snapshot_temperature(
        JOUR_A_CALCULER, HEURE_UTC_A_CALCULER
    )

    plt.close("all")
    if USE_CARTOPY:
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(12, 7))
        ax = plt.axes(projection=proj)
    else:
        fig, ax = plt.subplots(figsize=(12, 7))
        proj = None

    _ORIGIN = "lower"

    if proj is not None:
        im = ax.imshow(
            T_grid_snapshot, origin=_ORIGIN, extent=[-180, 180, -90, 90],
            transform=proj, cmap="inferno", vmin=220, vmax=320,
        )
        if USE_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="w")
            ax.set_global()
    else:
        im = ax.imshow(
            T_grid_snapshot, origin=_ORIGIN, extent=[-180, 180, -90, 90],
            cmap="inferno", interpolation="nearest", vmin=220, vmax=320,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    cb.set_label("Température d'équilibre de surface (K)")
    ax.set_title(
        f"Température d'équilibre (K) - Jour {JOUR_A_CALCULER}, {HEURE_UTC_A_CALCULER:.1f}h UTC"
    )
    plt.show()