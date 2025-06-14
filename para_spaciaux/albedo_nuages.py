"""
Albédo nuageux diurne (jour-seulement) à partir de CERES SYN1deg-Hour Ed4.2
Point : 45° N, 5° E -> janvier 2025
"""

from pathlib import Path
import numpy as np
import xarray as xr

# ──────────────────── 1.  Charger le jeu horaire ────────────────────────────
data_dir = Path("para_spaciaux")
fname = "CERES_EBAF-TOA_Ed4.2_Subset.nc"      # à adapter si besoin
ds = xr.open_dataset(data_dir / fname, decode_times=True)

# ──────────────────── 2.  Sélection des flux TOA ────────────────────────────
toa_sw_all  = ds["toa_sw_all_mon"]        # W m-2 sortant, all-sky
toa_sw_clr  = ds["toa_sw_clr_c_mon"]      # W m-2 sortant, clear-sky mod.
solar_in    = ds["solar_mon"]             # W m-2 entrant

# ─────────── 3.  Normaliser longitudes et trier les axes pour sel() ─────────
ds = ds.assign_coords(
    lon=(((ds.lon + 180) % 360) - 180)           # 0-360 → -180-180
).sortby("lon").sortby("time")

# ─────────────────── 4.  Masque « jour seulement » ──────────────────────────
mask_day = solar_in > 0                         # bool : True quand éclairé

# Albédo instantané (SW-CRE / F☉) uniquement si mask_day ; sinon NaN
cloud_alb_inst = xr.where(
    mask_day,
    (toa_sw_all - toa_sw_clr) / solar_in,
    np.nan,
)

# ──────────────────── 5.  Moyenne mensuelle diurne ──────────────────────────
# On réduit « time » sur janvier 2025 (inclus) après avoir lissé les NaN.
cloud_alb_month = (
    cloud_alb_inst
    .sel(time=slice("2025-01-01", "2025-01-31"))
    .mean(dim="time", skipna=True)
)

# ──────────────────── 6.  Extraction du point 45° N / 5° E ──────────────────
val = (
    cloud_alb_month
    .sel(lat=45, lon=5, method="nearest")
    .item()
)

print(f"Albédo nuageux (jour-seulement) – janv 2025, 45° N 5° E : {val:.3f}")
