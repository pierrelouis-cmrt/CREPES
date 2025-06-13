# -*- coding: utf-8 -*-
"""
Interactive planisphère – coherent solar power
==============================================
Correctif : la **puissance solaire locale** affichée dans l’annotation était
incorrecte ; elle n’utilisait pas le même modèle que la température. Désormais :

* nouvelle fonction `power_and_T_point()` reprise du script de test fourni ;
* l’annotation emploie cette fonction, donc **P ≠ 0** chaque fois que T > 0 ;
* présentation inchangée : carte température, Cₚ à 1 décimale, sliders mois &
  heure.
"""

from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    USE_CARTOPY = True
except ModuleNotFoundError:
    USE_CARTOPY = False

# ────────────────────────────────────────────────
# DATA – monthly albedo CSVs
# ────────────────────────────────────────────────

def load_albedo_series(csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"):
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
    return np.stack(cubes, axis=0), latitudes, longitudes

ALBEDO_DIR = pathlib.Path("ressources/albedo")
monthly_albedo, LAT, LON = load_albedo_series(ALBEDO_DIR)

LAT = LAT.astype(float)
LON = LON.astype(float)
_lat_idx = lambda lat: int(np.abs(LAT - lat).argmin())
_lon_idx = lambda lon: int(np.abs(LON - (((lon + 180) % 360) - 180)).argmin())
_ORIGIN = "upper" if LAT[0] > LAT[-1] else "lower"

# 2‑D latitude/longitude grids (broadcast ready)
LAT2D = np.repeat(LAT[:, None], LON.size, axis=1)
LON2D = np.repeat(LON[None, :], LAT.size, axis=0)

# ────────────────────────────────────────────────
# Cₚ – albedo‑based lookup (1 decimal)
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
    "ice": 0.60,
    "water": 4.2,
    "snow": 0.80,
    "desert": 0.35,
    "forest": 0.15,
    "land": 0.15,
}

def capacite_thermique(albedo: float) -> float:
    if np.isnan(albedo):
        return _CAPACITY_BY_TYPE["land"]
    surf = min(_REF_ALBEDO, key=lambda k: abs(albedo - _REF_ALBEDO[k]))
    return _CAPACITY_BY_TYPE[surf]

# ────────────────────────────────────────────────
# Solar power & temperature (vector + point versions)
# ────────────────────────────────────────────────

_S0 = 1361.0
_sigma = 5.67e-8

def _sun_vector(month: int, hour: float):
    """Return 3‑D sun vector for given month (1‑12) and local solar hour."""
    v = np.array([1.0, 0.0, 0.0])
    # axial tilt (season)
    tilt = np.radians(23 * np.cos(2 * np.pi * month / 12))
    v = np.array([
        v[0] * np.cos(tilt) + v[2] * np.sin(tilt),
        v[1],
        -v[0] * np.sin(tilt) + v[2] * np.cos(tilt),
    ])
    # diurnal rotation
    ang = (hour / 24) * 2 * np.pi
    v = np.array([
        v[0] * np.cos(ang) - v[1] * np.sin(ang),
        v[0] * np.sin(ang) + v[1] * np.cos(ang),
        v[2],
    ])
    return v


def power_and_T_point(lat_deg: float, lon_deg: float, month: int, hour: float, albedo: float):
    """Return (T, power) at a single point using the same model as the grid."""
    theta = np.radians(90 - lat_deg)
    phi = np.radians(lon_deg % 360)
    sun = _sun_vector(month, hour)
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])
    cos_i = max(np.dot(normal, sun), 0.0)
    power = _S0 * cos_i * (1 - albedo)
    T = (power / _sigma) ** 0.25 if power > 0 else 0.0
    return T, power


def temperature_grid(month: int, hour: float) -> np.ndarray:
    sun = _sun_vector(month, hour)
    theta = np.radians(90 - LAT2D)
    phi = np.radians(LON2D % 360)
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    cos_i = np.maximum(nx * sun[0] + ny * sun[1] + nz * sun[2], 0.0)
    alb = monthly_albedo[month - 1]
    power = _S0 * cos_i * (1 - alb)
    with np.errstate(invalid="ignore"):
        T = np.where(power > 0, (power / _sigma) ** 0.25, 0.0)
    return T

# ────────────────────────────────────────────────
# FIGURE & WIDGETS
# ────────────────────────────────────────────────

plt.close("all")
if USE_CARTOPY:
    proj = ccrs.PlateCarree(); fig = plt.figure(figsize=(10, 5)); ax = plt.axes(projection=proj)
else:
    fig, ax = plt.subplots(figsize=(10, 5)); proj = None

initial_T = temperature_grid(month=1, hour=12)
if proj is not None:
    im = ax.imshow(initial_T, origin=_ORIGIN, extent=[-180, 180, -90, 90], transform=proj, cmap="inferno")
    if USE_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4); ax.set_global()
else:
    im = ax.imshow(initial_T, origin=_ORIGIN, extent=[-180, 180, -90, 90], cmap="inferno", interpolation="nearest")
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")

ax.set_title("Temperature (K) – Month 1, Hour 12")
cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.04, pad=0.02)
cb.set_label("Temperature (K)")

slider_ax_month = plt.axes([0.15, 0.10, 0.70, 0.03])
slider_ax_hour  = plt.axes([0.15, 0.05, 0.70, 0.03])
slider_month = Slider(slider_ax_month, "Month", 1, 12, valinit=1, valstep=1, color="0.3")
slider_hour  = Slider(slider_ax_hour,  "Hour",  0, 23, valinit=12, valstep=1, color="0.3")


def _refresh(_):
    m = int(slider_month.val); h = slider_hour.val
    im.set_data(temperature_grid(m, h))
    ax.set_title(f"Temperature (K) – Month {m}, Hour {h}")
    fig.canvas.draw_idle()

slider_month.on_changed(_refresh)
slider_hour.on_changed(_refresh)

# ────────────────────────────────────────────────
# CLICK INFO
# ────────────────────────────────────────────────

_annotation = None

def _onclick(event):
    global _annotation
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return
    lon, lat = float(event.xdata), float(event.ydata)
    m, h = int(slider_month.val), slider_hour.val
    li, lj = _lat_idx(lat), _lon_idx(lon)
    alb = float(monthly_albedo[m - 1, li, lj])
    cap = capacite_thermique(alb)
    T, P = power_and_T_point(lat, lon, m, h, alb)
    if _annotation is not None:
        _annotation.remove()
    txt = (
        f"λ = {lat:+.1f}°, φ = {lon:+.1f}°\n"
        f"Month {m}, Hour {h:.1f} h\n"
        f"Albedo = {alb:.2f}\n"
        f"Cₚ = {cap:.1f} J kg⁻¹ K⁻¹\n"
        f"Solar = {P:.1f} W m⁻²\n"
        f"T_bb = {T:.1f} K"
    )
    _annotation = ax.annotate(txt, xy=(lon, lat), xycoords="data", xytext=(5, 5), textcoords="offset points",
                              fontsize=8, color="white", backgroundcolor="black", ha="left", va="bottom", zorder=5)
    def _hide():
        global _annotation
        if _annotation is not None:
            _annotation.remove(); _annotation = None; fig.canvas.draw_idle()
    fig.canvas.new_timer(interval=3000, callbacks=[(_hide, [], {})]).start(); fig.canvas.draw_idle()

fig.canvas.mpl_connect("button_press_event", _onclick)

plt.tight_layout(); plt.show()
