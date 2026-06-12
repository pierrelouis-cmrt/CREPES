"""GUI clic carte -> courbe, inspiree de Bernard et raccordee au moteur final."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from modele_courbe import run_point_simulation  # noqa: E402

try:
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE = True
except ImportError:
    ccrs = None
    CARTOPY_AVAILABLE = False


def launch_gui():
    """Lance l'interface Tkinter Bernard raccordee au moteur CREPES."""
    root = tk.Tk()
    root.title("CREPES - carte et temperature")
    root.geometry("1300x700")

    frame_left = ttk.Frame(root)
    frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame_right = ttk.Frame(root)
    frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    fig_map = plt.Figure(figsize=(7, 5), dpi=100)
    if CARTOPY_AVAILABLE:
        ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax_map.coastlines()
        ax_map.set_global()
        ax_map.gridlines(draw_labels=True, alpha=0.3)
    else:
        ax_map = fig_map.add_subplot(1, 1, 1)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.grid(True, alpha=0.3)
    ax_map.set_title("Cliquez sur une position")

    fig_temp = plt.Figure(figsize=(7, 5), dpi=100)
    ax_temp = fig_temp.add_subplot(1, 1, 1)
    ax_temp.set_title("Temperature")
    ax_temp.set_xlabel("Temps (jours)")
    ax_temp.set_ylabel("Temperature (deg C)")
    ax_temp.grid(True, alpha=0.3)

    canvas_map = FigureCanvasTkAgg(fig_map, master=frame_left)
    canvas_map.draw()
    canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    canvas_temp = FigureCanvasTkAgg(fig_temp, master=frame_right)
    canvas_temp.draw()
    canvas_temp.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_map_click(event):
        if event.inaxes != ax_map or event.xdata is None or event.ydata is None:
            return
        lon, lat = round(event.xdata, 2), round(event.ydata, 2)
        ax_temp.clear()
        ax_temp.set_title(f"Calcul en cours: lat={lat}, lon={lon}")
        canvas_temp.draw()

        output = run_point_simulation(lat_sim=lat, lon_sim=lon, days=365)
        days = output["times"] / 86400
        temps = output["temperature"] - 273.15
        ax_temp.clear()
        ax_temp.plot(days, temps, color="tab:blue", lw=1.8)
        ax_temp.set_title(f"Temperature a lat={lat}, lon={lon}")
        ax_temp.set_xlabel("Temps (jours)")
        ax_temp.set_ylabel("Temperature (deg C)")
        ax_temp.grid(True, alpha=0.3)
        canvas_temp.draw()

    fig_map.canvas.mpl_connect("button_press_event", on_map_click)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()

