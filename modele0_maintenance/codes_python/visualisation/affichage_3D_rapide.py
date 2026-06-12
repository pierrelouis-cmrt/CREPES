"""Viewer 3D rapide repris de l'idee Ornithorynquietant.

Les temperatures mensuelles sont lues dans `ressources/12_mois`, ce qui evite
un recalcul permanent.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider

CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from chemins import MONTHLY_TEMPERATURE_DIR  # noqa: E402
from visualisation.visualisation_commune import tracer_contours_sphere  # noqa: E402

MONTH_FILES = {
    "janvier": "Janvier.csv",
    "fevrier": "Février.csv",
    "mars": "Mars.csv",
    "avril": "Avril.csv",
    "mai": "Mai.csv",
    "juin": "Juin.csv",
    "juillet": "Juillet.csv",
    "aout": "Août.csv",
    "septembre": "Septembre.csv",
    "octobre": "Octobre.csv",
    "novembre": "Novembre.csv",
    "decembre": "Décembre.csv",
}


def _sphere_mesh(rows=30, cols=60):
    phi = np.linspace(0, 2 * np.pi, cols)
    theta = np.linspace(0, np.pi, rows)
    phi, theta = np.meshgrid(phi, theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def load_month(month):
    """Charge un CSV mensuel Ornithorynquietant en matrice (24, 30, 60)."""
    key = month.lower()
    if key not in MONTH_FILES:
        raise ValueError(f"Mois inconnu: {month}")
    path = MONTHLY_TEMPERATURE_DIR / MONTH_FILES[key]
    data = np.loadtxt(path, delimiter=",")
    frames = []
    for hour in range(24):
        frames.append(data[:, hour].reshape((30, 60)))
    return np.array(frames)


def show_3d(month="janvier", initial_hour=0, afficher=True, sauvegarde=None):
    """Affiche le mois choisi sur sphere avec slider horaire."""
    frames = load_month(month)
    hour = min(max(initial_hour, 0), 23)
    x, y, z = _sphere_mesh()
    norm = Normalize(vmin=float(np.nanmin(frames)), vmax=float(np.nanmax(frames)))

    fig = plt.figure(figsize=(9, 8))
    plt.subplots_adjust(bottom=0.16)
    ax = fig.add_subplot(111, projection="3d")

    def draw(hour_index):
        ax.clear()
        colors = cm.viridis(norm(frames[hour_index]))
        ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False)
        tracer_contours_sphere(ax)
        ax.set_axis_off()
        ax.set_title(f"{month.capitalize()} - {hour_index:02d}h")

    draw(hour)
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.08)
    colorbar.set_label("Température (K)")

    slider_ax = fig.add_axes([0.15, 0.06, 0.7, 0.04])
    slider = Slider(slider_ax, "Heure", 0, 23, valinit=hour, valstep=1)

    def update(value):
        draw(int(value))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    if sauvegarde:
        fig.savefig(sauvegarde, dpi=180, bbox_inches="tight")
    if afficher:
        plt.show()
    return fig, ax


def _build_parser():
    parser = argparse.ArgumentParser(description="Affichage 3D rapide CREPES")
    parser.add_argument("--month", default="janvier", choices=sorted(MONTH_FILES))
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    show_3d(
        args.month,
        args.hour,
        afficher=not args.no_show,
        sauvegarde=args.save,
    )
