#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_temperature_annual_plot.py

Télécharge la température ERA5 via l'API CDS, l'interpole à une altitude
donnée et affiche la courbe d'évolution sur l'année.

Dépendances :
    pip install cdsapi xarray netCDF4 numpy pandas matplotlib

ATTENTION : une clé d'API Copernicus est nécessaire pour éxecuter ce script.
"""

# =======================  CONFIGURATION  =====================================
LATITUDE = 48.858  # latitude décimale  (ex. Tour Eiffel)
LONGITUDE = 2.295  # longitude décimale (ex. Tour Eiffel)
ALTITUDE_M = 324.0  # altitude en mètres
ANNEE = "2024"  # année d'intérêt (YYYY)
# ============================================================================

import cdsapi
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def altitude_to_pressure(alt_m: float) -> float:
    """Altitude (m) -> pression ISA (hPa). Approche valable jusqu'à ~11 km."""
    P0, T0, L, g, R = 101325.0, 288.15, 0.0065, 9.80665, 287.05
    p_pa = P0 * (1 - L * alt_m / T0) ** (g / (R * L))
    return p_pa / 100.0


def levels_surrounding(alt_m: float) -> list[int]:
    """Renvoie les deux niveaux ERA5 (hPa) entourant la pression ISA d'alt_m."""
    levels = np.array([
        1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
        225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
        775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
    ])
    p_hpa = altitude_to_pressure(alt_m)
    idx = np.searchsorted(levels, p_hpa)
    low = levels[max(idx - 1, 0)]
    high = levels[min(idx, len(levels) - 1)]
    return [int(low), int(high)]


def fetch_era5_data() -> xr.Dataset:
    plevs = levels_surrounding(ALTITUDE_M)
    cds = cdsapi.Client()

    request = {
        "product_type": "reanalysis",
        "variable": "temperature",
        "pressure_level": [str(p) for p in plevs],
        "year": ANNEE,
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day":  [f"{d:02d}" for d in range(1, 32, 6)],
        "time": ["12:00"],
        "data_format": "netcdf",
    }

    target = f"era5_{ANNEE}_{LATITUDE}_{LONGITUDE}.nc"
    cds.retrieve("reanalysis-era5-pressure-levels", request, target)  # téléchargement local
    return xr.open_dataset(target, engine="h5netcdf")                 # lecture du fichier local



def plot_temperature_series(series: pd.Series):
    """Affiche le graphique de la série temporelle de température."""
    print("📈 Génération du graphique...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7))

    series.plot(ax=ax, label="Température interpolée", color="crimson")

    # Mise en forme du graphique
    ax.set_title(
        f"Température ERA5 en {ANNEE} pour {LATITUDE}°N, {LONGITUDE}°E "
        f"à {ALTITUDE_M:.0f} m",
        fontsize=16,
    )
    ax.set_ylabel("Température (°C)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend()

    # Formatage de l'axe des dates pour une meilleure lisibilité
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()


def main():
    """Fonction principale du script."""
    print("=== Analyse de la température annuelle ERA5 (Copernicus CDS) ===")
    try:
        # 1. Récupérer les données via l'API
        ds = fetch_era5_data()

        # 2. Extraire et interpoler la série temporelle
        lon_era5 = LONGITUDE % 360  # ERA5 utilise 0–360°
        point = ds["t"].sel(
            latitude=LATITUDE, longitude=lon_era5, method="nearest"
        )
        p_hpa = altitude_to_pressure(ALTITUDE_M)
        temp_K = point.interp(level=p_hpa) # 'pressure_level' est renommé 'level' par xarray
        serie = (temp_K - 273.15).to_series().rename("temp_C").dropna()

        print("\n--- Aperçu des 10 premières valeurs ---")
        print(serie.head(10))
        print("\nNombre total de pas de temps :", len(serie))
        print("Plage :", serie.index.min(), "→", serie.index.max())
        print(f"Température moyenne : {serie.mean():.2f} °C")
        print(f"Température min/max : {serie.min():.2f} °C / {serie.max():.2f} °C")

        # 3. Afficher le graphique
        plot_temperature_series(serie)

    except Exception as e:
        print(f"\n❌ Une erreur est survenue : {e}")
        print("Veuillez vérifier votre configuration, vos identifiants CDS")
        print("et votre connexion internet.")


if __name__ == "__main__":
    main()