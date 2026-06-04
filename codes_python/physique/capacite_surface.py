"""Capacite thermique de surface.

Base retenue: humidite du sol RZSM Carcajous. Si la source RZSM est absente ou
illisible, le moteur retombe sur une capacite seche constante minimale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chemins import RZSM_CSV

try:
    from scipy.stats import binned_statistic_2d

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

RHO_W = 1000.0
RHO_BULK = 1300.0
CP_SEC = 0.8
CP_WATER = 4.187
CP_ICE = 2.09
EPAISSEUR_ACTIVE = 0.5

_RZSM_CACHE = None


def compute_cp_from_rzsm(rzsm: np.ndarray) -> np.ndarray:
    """Calcule c_p en kJ kg-1 K-1 depuis l'humidite RZSM."""
    is_ice = np.isclose(rzsm, 0.9)
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (RHO_W * rzsm_clipped) / (
        RHO_BULK * (1 - rzsm_clipped) + RHO_W * rzsm_clipped
    )
    cp = CP_SEC + w * (CP_WATER - CP_SEC)
    return np.where(is_ice, CP_ICE, cp)


def load_and_grid_rzsm_data(csv_path: Path = RZSM_CSV):
    """Charge et grille les donnees RZSM sur une grille reguliere."""
    global _RZSM_CACHE
    if _RZSM_CACHE is not None and Path(csv_path) == RZSM_CSV:
        return _RZSM_CACHE
    if not SCIPY_AVAILABLE:
        return None, None, None
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier RZSM introuvable: {csv_path}")

    df = pd.read_csv(csv_path)
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    grid_res = 1.0
    lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    lat_bins = np.arange(-90, 90 + grid_res, grid_res)
    statistic, _, _, _ = binned_statistic_2d(
        x=df["lon"],
        y=df["lat"],
        values=df["RZSM"],
        statistic="mean",
        bins=[lon_bins, lat_bins],
    )
    result = statistic.T, lat_bins, lon_bins
    if Path(csv_path) == RZSM_CSV:
        _RZSM_CACHE = result
    return result


def _rzsm_value_at(lat_deg, lon_deg, csv_path: Path = RZSM_CSV):
    grid, lat_bins, lon_bins = load_and_grid_rzsm_data(csv_path)
    if grid is None:
        return np.nan
    lat_idx = min(np.abs(lat_bins - lat_deg).argmin(), grid.shape[0] - 1)
    lon_idx = min(
        np.abs(lon_bins - (((lon_deg + 180) % 360) - 180)).argmin(),
        grid.shape[1] - 1,
    )
    return grid[lat_idx, lon_idx]


def compute_surface_capacity(lat_deg, lon_deg):
    """Calcule C surfacique pour le moteur Carcajous avec fallback sec."""
    rzsm_val = _rzsm_value_at(lat_deg, lon_deg)
    if not np.isnan(rzsm_val):
        cp_kj = compute_cp_from_rzsm(np.array([rzsm_val]))[0]
        return (cp_kj * 1000.0) * RHO_BULK * EPAISSEUR_ACTIVE, "Carcajous RZSM"

    return (CP_SEC * 1000.0) * RHO_BULK * EPAISSEUR_ACTIVE, "Carcajous CP_SEC fallback"
