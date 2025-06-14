#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
era5_temperature_annual.py

Télécharge et lit la température ERA5 sur deux niveaux pressure-levels
encadrant l'altitude, pour toute une année, aux heures UTC suivantes :
00 h, 04 h, 12 h, 16 h, 20 h.

Dépendances :
    pip install cdsapi xarray netCDF4 numpy pandas
"""

# =======================  CONFIGURATION  =====================================
LATITUDE   = 48.858      # latitude décimale  (ex. Tour Eiffel)
LONGITUDE  = 2.295       # longitude décimale (ex. Tour Eiffel)
ALTITUDE_M = 324.0       # altitude en mètres
ANNEE      = "2024"      # année d'intérêt (YYYY)
OUTFILE    = "ressources/lefichier.nc"   # chemin NetCDF de sortie
# ============================================================================

import os
from datetime import datetime
import numpy as np
import cdsapi
import xarray as xr
import pandas as pd


def altitude_to_pressure(alt_m: float) -> float:
    """Altitude (m) -> pression ISA (hPa). Approche valable jusqu'à ~11 km."""
    P0, T0, L, g, R = 101325.0, 288.15, 0.0065, 9.80665, 287.05
    p_pa = P0 * (1 - L * alt_m / T0) ** (g / (R * L))
    return p_pa / 100.0


def levels_surrounding(alt_m: float) -> list[int]:
    """Renvoie les deux niveaux ERA5 (hPa) entourant la pression ISA d'alt_m."""
    levels = np.array([
        1,2,3,5,7,10,20,30,50,70,100,125,150,175,200,225,250,300,
        350,400,450,500,550,600,650,700,750,775,800,825,850,875,900,
        925,950,975,1000
    ])
    p_hpa = altitude_to_pressure(alt_m)
    idx   = np.searchsorted(levels, p_hpa)
    low   = levels[max(idx - 1, 0)]
    high  = levels[min(idx, len(levels) - 1)]
    return [int(low), int(high)]


def download_era5():
    """Télécharge (si besoin) le NetCDF ERA5 pour l'année et l'heure fixées."""
    if os.path.exists(OUTFILE):
        print("✔ Fichier déjà présent :", OUTFILE)
        return

    plevs = levels_surrounding(ALTITUDE_M)
    print(f"Altitude {ALTITUDE_M:.0f} m → ~{altitude_to_pressure(ALTITUDE_M):.0f} hPa"
          f" ; niveaux demandés = {plevs}")

    times = ["00:00", "04:00", "12:00", "16:00", "20:00"]

    cds = cdsapi.Client()
    cds.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type":  "reanalysis",
            "variable":      "temperature",
            "pressure_level": [str(p) for p in plevs],
            "year":          [ANNEE],
            "month":         [f"{m:02d}" for m in range(1, 13)],
            "day":           [f"{d:02d}" for d in range(1, 32)],
            "time":          times,
            "format":        "netcdf",
        },
        OUTFILE
    )
    print("⬇ Téléchargement terminé →", OUTFILE)


def read_temperature_series() -> pd.Series:
    """Lit le NetCDF et renvoie la température interpolée à l'altitude (°C)."""
    ds = xr.open_dataset(OUTFILE)
    lon_era5 = LONGITUDE % 360  # ERA5 utilise 0–360°
    point = ds["t"].sel(latitude=LATITUDE, longitude=lon_era5, method="nearest")
    p_hpa = altitude_to_pressure(ALTITUDE_M)
    temp_K = point.interp(level=p_hpa)
    return (temp_K - 273.15).to_series().rename("temp_C")


def main():
    print("=== Téléchargement / lecture ERA5 (Copernicus CDS) ===")
    download_era5()
    serie = read_temperature_series()
    print("\n--- Aperçu des 10 premières valeurs ---")
    print(serie.head(10))
    print("\nNombre total de pas de temps :", len(serie))
    print("Plage :", serie.index.min(), "→", serie.index.max())


if __name__ == "__main__":
    main()
