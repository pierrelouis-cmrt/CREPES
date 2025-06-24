# Non fonctionnel, il nous manque le fichier .nc dans sa version horaire et non mensuelle

from pathlib import Path
import numpy as np
import xarray as xr
import calendar
import sys
import subprocess


try:
    import sys
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", sys])
    import sys

try:
    import numpy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", numpy])
    import numpy

try:
    import xarray
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", xarray])
    import xarray

try:
    import calendar
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", calendar])
    import calendar

try:
    import pathlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", pathlib])
    import pathlib


# ─────────────── Paramètres à modifier ───────────────
year = 2025       # année souhaitée
month = 1         # numéro du mois (1 à 12)
# ──────────────────────────────────────────────────────

# Calcul automatique du premier et dernier jour du mois
first_day = f"{year}-{month:02d}-01"
# calendar.monthrange renvoie (weekday_first_day, nombre_de_jours)
last_day = calendar.monthrange(year, month)[1]
last_day_str = f"{year}-{month:02d}-{last_day:02d}"

# ─────────────────── 1. Charger le jeu horaire ───────────────────
data_dir = Path("para_spaciaux/albedo")
fname = "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"      # à adapter si besoin
ds = xr.open_dataset(data_dir / fname, decode_times=True)

# ─────────────── 2. Sélection des flux TOA ───────────────
toa_sw_all  = ds["toa_sw_all_mon"]       # W m-2 sortant, all-sky
toa_sw_clr  = ds["toa_sw_clr_c_mon"]     # W m-2 sortant, clear-sky
solar_in    = ds["solar_mon"]            # W m-2 entrant

# ──── 3. Normaliser longitudes et trier les axes pour sel() ────
ds = ds.assign_coords(
    lon=(((ds.lon + 180) % 360) - 180)   # 0–360 → –180–180
).sortby("lon").sortby("time")

# ──────────────── 4. Masque « jour seulement » ────────────────
mask_day = solar_in > 0                   # True quand éclairé

cloud_alb_inst = xr.where(
    mask_day,
    (toa_sw_all - toa_sw_clr) / solar_in,
    np.nan,
)

# ─────── 5. Moyenne mensuelle diurne automatique ───────
cloud_alb_month = (
    cloud_alb_inst
    .sel(time=slice(first_day, last_day_str))
    .mean(dim="time", skipna=True)
)

# ──── 6. Extraction du point 45° N / 5° E ────
val = (
    cloud_alb_month
    .sel(lat=45, lon=5, method="nearest")
    .item()
)

print(f"Albédo nuageux (jour-seulement) – {year}-{month:02d}, 45° N 5° E : {val:.3f}")
