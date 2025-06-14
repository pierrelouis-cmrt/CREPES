"""
Calcule une approximation de l’albédo dû aux nuages (SW CRE / F☉) pour
janvier 2025 à 45° N – 5° E, à partir du jeu de données CERES EBAF-TOA Ed4.2.

Notes :
- On laisse les valeurs négatives intactes (NaN) pour repérer les situations
  où les nuages augmentent au contraire l’absorption SW.
- Les longitudes sont ramenées dans l’intervalle [-180°, +180°].
"""

from pathlib import Path
import xarray as xr
import numpy as np

# --------------------------- Chargement du jeu de données -------------------

data_dir = Path("para_spaciaux/albedo")                # répertoire racine
fname = "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"
ds = xr.open_dataset(data_dir / fname, decode_times=True)

# ----------------------------- Sélection des flux ---------------------------

toa_sw_all = ds["toa_sw_all_mon"]   # W m-2 sortant, all-sky
toa_sw_clr = ds["toa_sw_clr_c_mon"] # W m-2 sortant, clear-sky
solar_incident = ds["solar_mon"]    # W m-2 incident au TOA

# --------------------- Calcul de la « cloud-albedo fraction » ---------------

cloud_albedo = (toa_sw_all - toa_sw_clr) / solar_incident
cloud_albedo = cloud_albedo.where(cloud_albedo >= 0)  # conserve NaN sinon

# ------------------ Harmonisation du repère de longitude --------------------
cloud_albedo = cloud_albedo.assign_coords(
    lon=(((cloud_albedo.lon + 180) % 360) - 180)
)
cloud_albedo = cloud_albedo.sortby("lon")

# --------------------- Extraction spatio-temporelle voulue ------------------
val = (
    cloud_albedo
    .sel(time="2025-01", lat=45, lon=5, method="nearest")
    .item()
)
print(f"Approx. albédo nuageux (janv 2025, 45°N 5°E) : {val:.3f}")
