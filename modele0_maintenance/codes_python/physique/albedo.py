"""Albedo sol/nuages.

Base retenue: CSV mensuels d'albedo de surface + fichier CERES Carcajous.
Fallback optionnel: cache API NASA inspire de Bernard, non utilise par defaut.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from chemins import ALBEDO_DIR, CACHE_DIR, CERES_FILE, ensure_cache_dir

# xarray sert seulement à lire le NetCDF CERES des nuages.
try:
    import xarray as xr

    XARRAY_AVAILABLE = True
except ImportError:
    xr = None
    XARRAY_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


def lisser_donnees_annuelles(valeurs_mensuelles: np.ndarray, sigma: float):
    """Lisse 12 valeurs mensuelles en 365 valeurs journalieres."""
    # Le mode wrap évite une rupture artificielle entre décembre et janvier.
    jours_par_mois = np.array(
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    )
    valeurs_journalieres = np.repeat(valeurs_mensuelles, jours_par_mois, axis=0)
    return gaussian_filter1d(valeurs_journalieres, sigma=sigma, mode="wrap", axis=0)


def load_albedo_series(csv_dir: Path = ALBEDO_DIR, pattern: str = "albedo{:02d}.csv"):
    """Charge les 12 CSV mensuels d'albedo de surface Carcajous."""
    if not csv_dir.exists():
        raise FileNotFoundError(f"Dossier d'albedo introuvable: {csv_dir}")

    latitudes, longitudes, cubes = None, None, []
    for month in range(1, 13):
        df = pd.read_csv(csv_dir / pattern.format(month))
        if latitudes is None:
            # Le premier CSV donne les axes; les mois suivants gardent la même grille.
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()
        cubes.append(df.set_index("Latitude/Longitude").to_numpy(dtype=float))
    return np.stack(cubes, axis=0), latitudes, longitudes


def load_monthly_cloud_albedo_from_ceres(
    lat_deg: float | None,
    lon_deg: float | None,
    return_full_map: bool = False,
    ceres_file: Path = CERES_FILE,
):
    """Extrait l'albedo mensuel des nuages depuis le NetCDF CERES."""
    if not XARRAY_AVAILABLE:
        return np.zeros(12)
    if not ceres_file.exists():
        raise FileNotFoundError(f"Fichier CERES introuvable: {ceres_file}")

    with xr.open_dataset(ceres_file, decode_times=True) as ds:
        ds.load()
        # Les longitudes CERES sont ramenées en [-180, 180] pour matcher le reste.
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
        toa_sw_all = ds["toa_sw_all_mon"]
        toa_sw_clr = ds["toa_sw_clr_c_mon"]
        solar_in = ds["solar_mon"]
        cloud_albedo = xr.where(
            solar_in > 1e-6, (toa_sw_all - toa_sw_clr) / solar_in, 0.0
        )
        cloud_albedo_monthly = cloud_albedo.groupby("time.month").mean(
            dim="time", skipna=True
        )

        if return_full_map:
            return cloud_albedo_monthly

        monthly_values = cloud_albedo_monthly.sel(
            lat=lat_deg, lon=lon_deg, method="nearest"
        ).to_numpy()

    if len(monthly_values) != 12:
        monthly_values = np.pad(monthly_values, (0, 12 - len(monthly_values)), mode="edge")
    return monthly_values


# Cache CSV très simple pour éviter de rappeler l'API NASA pendant les essais.
def _read_albedo_cache(cache_file: Path):
    if not cache_file.exists():
        return {}
    with cache_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["key"]: float(row["albedo"]) for row in reader}


def _write_albedo_cache(cache_file: Path, values: dict[str, float]):
    ensure_cache_dir()
    with cache_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "albedo"])
        writer.writeheader()
        for key, value in sorted(values.items()):
            writer.writerow({"key": key, "albedo": value})


def get_nasa_albedo_cached(
    lat,
    lon,
    date_debut="2022-01-01",
    duree_simulation_jours=365,
    cache_file: Path = CACHE_DIR / "nasa_albedo_cache.csv",
):
    """Fallback Bernard: albedo moyen NASA POWER avec cache CSV local."""
    cache_key = f"{lat:.4f},{lon:.4f},{date_debut},{duree_simulation_jours}"
    cache = _read_albedo_cache(cache_file)
    if cache_key in cache:
        return cache[cache_key]
    if not REQUESTS_AVAILABLE:
        return 0.3

    date_debut_obj = datetime.strptime(date_debut, "%Y-%m-%d")
    date_fin_obj = date_debut_obj + timedelta(days=min(duree_simulation_jours, 365) - 1)
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_UP",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": date_debut_obj.strftime("%Y%m%d"),
        "end": date_fin_obj.strftime("%Y%m%d"),
        "format": "JSON",
    }
    try:
        response = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()["properties"]["parameter"]
        down = data["ALLSKY_SFC_SW_DWN"]
        up = data["ALLSKY_SFC_SW_UP"]
        values = [
            max(0.0, min(1.0, up[day] / down[day]))
            for day in down
            if down.get(day) and up.get(day) and down[day] > 0
        ]
        albedo = sum(values) / len(values) if values else 0.3
    except Exception:
        # Fallback volontairement neutre si le réseau ou la réponse API échoue.
        albedo = 0.3

    cache[cache_key] = max(0.05, min(0.95, albedo))
    _write_albedo_cache(cache_file, cache)
    return cache[cache_key]
