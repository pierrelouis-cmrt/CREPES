"""Convection.

Deux briques sont conservees:
- Chevreaux: coefficient de convection forcee a partir du vent + API NASA.
- Ornithorynquietant: coefficient naturel via Rayleigh/Nusselt.
"""

from __future__ import annotations

import csv
from pathlib import Path

from chemins import CACHE_DIR, ensure_cache_dir

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


def coefficient_convection(v):
    """Coefficient h Chevreaux a partir de la vitesse du vent en m s-1."""
    rho = 1.2
    mu = 1.8e-5
    length = 1.0
    prandtl = 0.71
    lambda_air = 0.026

    re = rho * max(v, 0.0) * length / mu
    # Corrélation plaque plane: régime laminaire puis turbulent.
    if re < 5e5:
        coeff, power_re, power_pr = 0.664, 0.5, 1 / 3
    else:
        coeff, power_re, power_pr = 0.037, 0.8, 1 / 3
    nusselt = coeff * re**power_re * prandtl**power_pr
    return nusselt * lambda_air / length


def _wind_cache_file(lat, lon, start, end) -> Path:
    safe = f"wind_{lat:.3f}_{lon:.3f}_{start}_{end}.csv".replace("-", "m")
    return CACHE_DIR / safe


def _read_wind_cache(cache_file: Path):
    if not cache_file.exists():
        return None
    with cache_file.open(newline="", encoding="utf-8") as handle:
        return [float(row["wind_speed"]) for row in csv.DictReader(handle)]


def _write_wind_cache(cache_file: Path, values):
    ensure_cache_dir()
    with cache_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["day_index", "wind_speed"])
        writer.writeheader()
        for index, value in enumerate(values):
            writer.writerow({"day_index": index, "wind_speed": value})


def get_daily_wind_speed(lat, lon, start="20240101", end="20241231"):
    """Recupere le vent NASA POWER avec fallback Chevreaux a 2.5 m s-1."""
    cache_file = _wind_cache_file(lat, lon, start, end)
    cached = _read_wind_cache(cache_file)
    if cached:
        return cached
    if not REQUESTS_AVAILABLE:
        # La valeur 2.5 m/s est le vent constant utilisé dans les essais historiques.
        return [2.5] * 365

    params = {
        "parameters": "WS2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    try:
        response = requests.get(
            "https://power.larc.nasa.gov/api/temporal/daily/point",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        wind_data = response.json()["properties"]["parameter"]["WS2M"]
        values = [wind_data[day] for day in sorted(wind_data)]
    except Exception:
        # En cas d'API indisponible, on garde une simulation reproductible.
        values = [2.5] * 365
    _write_wind_cache(cache_file, values)
    return values


def liste_h(lat, long):
    """Liste horaire de h sur un an, selon la fonction Chevreaux."""
    values = []
    daily_wind = get_daily_wind_speed(lat, long)
    for daily_value in daily_wind:
        for _ in range(24):
            values.append(coefficient_convection(daily_value))
    return values


def calcul_h(T_sol, T_air):
    """Coefficient de convection naturelle Ornithorynquietant."""
    lam = 0.026
    length = 0.05
    exponent = 1 / 4
    coeff = 0.54 if T_sol >= T_air else 0.27

    g = 9.81
    nu = 1.5e-5
    alpha = 2e-5
    beta = 1 / T_air
    grashof = (g * beta * (T_sol - T_air) * length**3) / nu**2
    prandtl = nu / alpha
    rayleigh = grashof * prandtl
    nusselt = coeff * abs(rayleigh) ** exponent
    return nusselt * lam / length


def convection_forced_flux(T_sol, T_air, wind_speed):
    """Flux convectif force h(Tsol - Tair)."""
    # Flux positif: la surface est plus chaude que l'air et perd de l'énergie.
    return coefficient_convection(wind_speed) * (T_sol - T_air)


def convection_natural_flux(T_sol, T_air):
    """Flux convectif naturel h(Tsol - Tair)."""
    return calcul_h(T_sol, T_air) * (T_sol - T_air)
