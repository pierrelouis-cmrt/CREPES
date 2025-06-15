
from __future__ import annotations
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def load_albedo_series(csv_dir: str | pathlib.Path, pattern: str = "albedo{:02d}.csv"):
    """Return a 3-D array (month, lat, lon) and the coordinate vectors."""
    csv_dir = pathlib.Path(csv_dir)

    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    cubes: list[np.ndarray] = []

    for month in range(1, 13):
        csv_path = csv_dir / pattern.format(month)
        df = pd.read_csv(csv_path)

        # Grab axes once
        if latitudes is None:
            latitudes = df["Latitude/Longitude"].astype(float).to_numpy()
            longitudes = df.columns[1:].astype(float).to_numpy()

        grid = df.set_index("Latitude/Longitude").to_numpy(dtype=float)
        cubes.append(grid)

    return np.stack(cubes, axis=0), latitudes, longitudes

# Folder containing albedo01.csv … albedo12.csv
ALBEDO_DIR = "ressources/albedo"

monthly_albedo, LAT, LON = load_albedo_series(ALBEDO_DIR)

# ---------------------------------------------------------------------------
# VISUALISATION SET-UP
# ---------------------------------------------------------------------------

# Optional cartopy coastline overlay
USE_CARTOPY = True
try:
    if USE_CARTOPY:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
except ModuleNotFoundError:
    USE_CARTOPY = False

# Create main figure/axes
if USE_CARTOPY:
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

# Plot first month initially
current_month_index = 0  # January
im = ax.imshow(
    monthly_albedo[current_month_index],
    extent=[LON.min(), LON.max(), LAT.min(), LAT.max()],
    origin="lower",
    cmap="viridis",
    transform=proj if USE_CARTOPY else ax.transData,
)

cbar = fig.colorbar(im, ax=ax, label="Albedo")
ax.set_title("Surface Albedo – Month 1 (January)")

# ---------------------------------------------------------------------------
# MONTH SLIDER
# ---------------------------------------------------------------------------

plt.subplots_adjust(bottom=0.20)  # leave space for slider
slider_ax = fig.add_axes([0.15, 0.08, 0.7, 0.04])
month_slider = Slider(
    slider_ax,
    "Month",
    1,
    12,
    valinit=1,
    valstep=1,
    color="0.25",  # dark grey knob improves visibility
)


def _redraw_for_month(m: int):
    """Update raster and title for month *m* (1-based)."""
    im.set_data(monthly_albedo[m - 1])
    ax.set_title(f"Surface Albedo – Month {m}")
    fig.canvas.draw_idle()


def _slider_changed(val):
    _redraw_for_month(int(val))


month_slider.on_changed(_slider_changed)

# ---------------------------------------------------------------------------
# CLICK-TO-READ INTERACTIVITY
# ---------------------------------------------------------------------------

def onclick(event):
    if event.inaxes is not ax or event.xdata is None or event.ydata is None:
        return

    lon_click = event.xdata
    lat_click = event.ydata

    # Snap to nearest cell
    lat_idx = np.abs(LAT - lat_click).argmin()
    lon_idx = np.abs(LON - lon_click).argmin()

    month_index = int(month_slider.val) - 1
    value = monthly_albedo[month_index, lat_idx, lon_idx]

    annotation = ax.text(
        lon_click,
        lat_click,
        f"{value:.3f}",
        fontsize=8,
        color="white",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.6),
        transform=ax.projection if USE_CARTOPY else ax.transData,
    )
    fig.canvas.draw_idle()

    # Auto-remove annotation after 2.5 s
    def _remove():
        annotation.remove()
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=2500)
    timer.add_callback(_remove)
    timer.start()


fig.canvas.mpl_connect("button_press_event", onclick)

plt.show()
