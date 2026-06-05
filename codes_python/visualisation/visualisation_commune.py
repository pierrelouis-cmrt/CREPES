"""Outils communs pour les visualisations CREPES.

Le but de ce module est uniquement la compatibilite et la qualite d'affichage:
charger les resultats Carcajous, retrouver les axes geographiques, et tracer
les contours des continents depuis les shapefiles locaux.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection

CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

import bibliotheque as lib  # noqa: E402
from chemins import (  # noqa: E402
    COASTLINE_SHP,
    HIRES_FAST_NPY,
    HIRES_NPY,
    HIRES_STABILIZED_NPY,
    LOWRES_FAST_NPY,
    LOWRES_NPY,
    LOWRES_STABILIZED_NPY,
)
from physique.albedo import load_albedo_series  # noqa: E402

try:
    import shapefile

    SHAPEFILE_DISPONIBLE = True
except ImportError:
    shapefile = None
    SHAPEFILE_DISPONIBLE = False


@dataclass(frozen=True)
class GrilleTemperature:
    """Resultats de temperature avec leurs axes geographiques."""

    donnees: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    chemin: Path
    niveau: str


GRILLES_PAR_VARIANTE = {
    "rapide": (HIRES_FAST_NPY, LOWRES_FAST_NPY, "rapide"),
    "1an": (HIRES_NPY, LOWRES_NPY, "1 an"),
    "stabilisee": (HIRES_STABILIZED_NPY, LOWRES_STABILIZED_NPY, "stabilisée"),
}


def _candidats_grille(preferer_haute_resolution, variante):
    if variante == "auto":
        ordre = ("1an", "stabilisee", "rapide")
    else:
        if variante not in GRILLES_PAR_VARIANTE:
            raise ValueError(f"Variante de grille inconnue: {variante}")
        ordre = (variante,)

    candidats = []
    for variante_courante in ordre:
        chemin_hi, chemin_low, libelle = GRILLES_PAR_VARIANTE[variante_courante]
        if preferer_haute_resolution:
            candidats.append((chemin_hi, "haute résolution", libelle))
            candidats.append((chemin_low, "basse résolution", libelle))
        else:
            candidats.append((chemin_low, "basse résolution", libelle))
            candidats.append((chemin_hi, "haute résolution", libelle))
    return candidats


def charger_grille_temperature(
    preferer_haute_resolution=True,
    variante="auto",
) -> GrilleTemperature:
    """Charge une grille de temperature deja generee."""
    chemin = None
    niveau = None
    for chemin_candidat, resolution, libelle in _candidats_grille(
        preferer_haute_resolution,
        variante,
    ):
        if chemin_candidat.exists():
            chemin = chemin_candidat
            niveau = f"{resolution} CREPES - {libelle}"
            break
    if chemin is None:
        raise FileNotFoundError(
            "Aucune grille compatible trouvee dans ressources/grilles"
        )

    donnees = np.load(chemin, mmap_mode="r")
    latitudes, longitudes = axes_geographiques(donnees.shape[1], donnees.shape[2])
    return GrilleTemperature(donnees, latitudes, longitudes, chemin, niveau)


def axes_geographiques(nombre_latitudes, nombre_longitudes):
    """Reconstitue les axes utilises par les sorties Carcajous."""
    _, lat_lowres, lon_lowres = load_albedo_series()
    if len(lat_lowres) == nombre_latitudes and len(lon_lowres) == nombre_longitudes:
        return lat_lowres, lon_lowres
    return (
        np.linspace(float(lat_lowres.min()), float(lat_lowres.max()), nombre_latitudes),
        np.linspace(float(lon_lowres.min()), float(lon_lowres.max()), nombre_longitudes),
    )


def indice_temps(grille: GrilleTemperature, jour=0, heure=0):
    """Convertit jour/heure en indice temporel dans la grille Carcajous."""
    pas_par_jour = int(24 * 3600 / lib.dt)
    pas_par_heure = int(3600 / lib.dt)
    indice = int(jour) * pas_par_jour + int(heure) * pas_par_heure
    return min(max(indice, 0), grille.donnees.shape[0] - 1)


def nombre_jours_affichables(grille: GrilleTemperature):
    """Nombre de jours entiers presents dans la grille."""
    return max(1, grille.donnees.shape[0] // int(24 * 3600 / lib.dt))


def temperature_celsius(grille: GrilleTemperature, indice):
    """Extrait une tranche temporelle en degres Celsius."""
    return np.asarray(grille.donnees[indice, :, :]) - 273.15


def _segments_depuis_points(points):
    """Decoupe les lignes qui traversent l'antimeridien."""
    segment = []
    dernier_lon = None
    for lon, lat in points:
        if dernier_lon is not None and abs(lon - dernier_lon) > 180:
            if len(segment) > 1:
                yield segment
            segment = []
        segment.append((lon, lat))
        dernier_lon = lon
    if len(segment) > 1:
        yield segment


def segments_cotes_lonlat():
    """Retourne les segments de cotes en lon/lat depuis le shapefile local."""
    if not SHAPEFILE_DISPONIBLE or not COASTLINE_SHP.exists():
        return []
    lecteur = shapefile.Reader(str(COASTLINE_SHP))
    segments = []
    for shape in lecteur.shapes():
        points = shape.points
        parts = list(shape.parts) + [len(points)]
        for debut, fin in zip(parts[:-1], parts[1:]):
            segments.extend(_segments_depuis_points(points[debut:fin]))
    return segments


def tracer_contours_planisphere(ax, couleur="black", epaisseur=0.45, alpha=0.9):
    """Ajoute les contours des continents a un axe 2D."""
    segments = segments_cotes_lonlat()
    if not segments:
        return None
    collection = LineCollection(
        segments,
        colors=couleur,
        linewidths=epaisseur,
        alpha=alpha,
        zorder=5,
    )
    ax.add_collection(collection)
    return collection


def _echantillonner_segment(segment, pas):
    """Reduit la densite d'une polyligne en conservant son dernier point."""
    if pas <= 1 or len(segment) <= 2:
        return segment
    segment_reduit = segment[::pas]
    if not np.array_equal(segment_reduit[-1], segment[-1]):
        segment_reduit = np.vstack((segment_reduit, segment[-1]))
    return segment_reduit


def _segments_cotes_xyz(rayon=1.012, pas=2):
    """Convertit les segments de cotes lon/lat vers une sphere 3D."""
    segments_xyz = []
    for segment in segments_cotes_lonlat():
        valeurs = _echantillonner_segment(np.asarray(segment, dtype=float), pas)
        lon_rad = np.radians(valeurs[:, 0])
        lat_rad = np.radians(valeurs[:, 1])
        x = rayon * np.cos(lat_rad) * np.cos(lon_rad)
        y = rayon * np.cos(lat_rad) * np.sin(lon_rad)
        z = rayon * np.sin(lat_rad)
        segments_xyz.append(np.column_stack((x, y, z)))
    return segments_xyz


def _direction_camera(ax):
    """Retourne le vecteur du centre de la sphere vers la camera."""
    elev_rad = np.radians(ax.elev)
    azim_rad = np.radians(ax.azim)
    direction = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=float,
    )
    return direction / np.linalg.norm(direction)


def _intersection_horizon(debut, fin, projection_debut, projection_fin, rayon):
    """Calcule le point ou un segment 3D coupe le bord visible de la sphere."""
    t = projection_debut / (projection_debut - projection_fin)
    point = debut + np.clip(t, 0.0, 1.0) * (fin - debut)
    norme = np.linalg.norm(point)
    if norme == 0:
        return point
    return point * (rayon / norme)


def _segments_visibles_devant_camera(segments_xyz, direction_camera, rayon=1.012):
    """Masque les parties de cotes situees sur l'hemisphere oppose."""
    segments_visibles = []
    for segment in segments_xyz:
        if len(segment) < 2:
            continue

        projections = segment @ direction_camera
        segment_visible = []
        point_precedent = segment[0]
        projection_precedente = projections[0]
        precedent_visible = projection_precedente >= 0
        if precedent_visible:
            segment_visible.append(point_precedent)

        for point, projection in zip(segment[1:], projections[1:]):
            point_visible = projection >= 0
            if precedent_visible and point_visible:
                segment_visible.append(point)
            elif precedent_visible and not point_visible:
                segment_visible.append(
                    _intersection_horizon(
                        point_precedent,
                        point,
                        projection_precedente,
                        projection,
                        rayon,
                    )
                )
                if len(segment_visible) > 1:
                    segments_visibles.append(np.asarray(segment_visible))
                segment_visible = []
            elif not precedent_visible and point_visible:
                segment_visible = [
                    _intersection_horizon(
                        point_precedent,
                        point,
                        projection_precedente,
                        projection,
                        rayon,
                    ),
                    point,
                ]

            point_precedent = point
            projection_precedente = projection
            precedent_visible = point_visible

        if len(segment_visible) > 1:
            segments_visibles.append(np.asarray(segment_visible))
    return segments_visibles


class _CollectionContoursSphere(Line3DCollection):
    """Collection 3D qui recalcule les cotes visibles avant projection."""

    def __init__(self, segments_xyz, **kwargs):
        super().__init__([], **kwargs)
        self._segments_xyz = segments_xyz
        self._vue_precedente = None

    def _mettre_a_jour_contours(self):
        if self.axes is None:
            return
        vue = (self.axes.elev, self.axes.azim, getattr(self.axes, "roll", 0))
        if vue == self._vue_precedente:
            return
        self._vue_precedente = vue
        direction = _direction_camera(self.axes)
        self.set_segments(
            _segments_visibles_devant_camera(self._segments_xyz, direction)
        )

    def do_3d_projection(self):
        self._mettre_a_jour_contours()
        return super().do_3d_projection()


def tracer_contours_sphere(ax, couleur="black", epaisseur=0.55, alpha=0.95):
    """Ajoute les contours des continents a une sphere 3D."""
    segments_xyz = _segments_cotes_xyz()
    if not segments_xyz:
        return None

    collection = _CollectionContoursSphere(
        segments_xyz,
        colors=couleur,
        linewidths=epaisseur,
        alpha=alpha,
        zorder=10,
    )
    ax.add_collection3d(collection, autolim=False)
    return collection



def creer_planisphere(
    preferer_haute_resolution=True,
    variante_grille="auto",
    jour=0,
    heure=0,
    afficher=True,
    sauvegarde=None,
):
    """Cree une figure planisphere avec sliders jour/heure et contours."""
    grille = charger_grille_temperature(preferer_haute_resolution, variante_grille)
    indice = indice_temps(grille, jour, heure)
    tranche = temperature_celsius(grille, indice)

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.22, top=0.92)
    image = ax.imshow(
        tranche,
        origin="lower",
        extent=[
            float(grille.longitudes.min()),
            float(grille.longitudes.max()),
            float(grille.latitudes.min()),
            float(grille.latitudes.max()),
        ],
        cmap="inferno",
        vmin=-50,
        vmax=50,
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    tracer_contours_planisphere(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=2)
    titre = ax.set_title(
        f"Température de surface - {grille.niveau} - jour {jour}, heure {heure}",
        fontsize=14,
    )
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    colorbar.set_label("Température de surface (°C)")

    axe_jour = plt.axes([0.2, 0.10, 0.6, 0.03])
    slider_jour = Slider(
        axe_jour,
        "Jour",
        0,
        nombre_jours_affichables(grille) - 1,
        valinit=jour,
        valstep=1,
    )
    axe_heure = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_heure = Slider(axe_heure, "Heure", 0, 23, valinit=heure, valstep=1)

    def rafraichir(_):
        jour_courant = int(slider_jour.val)
        heure_courante = int(slider_heure.val)
        idx = indice_temps(grille, jour_courant, heure_courante)
        image.set_data(temperature_celsius(grille, idx))
        titre.set_text(
            f"Température de surface - {grille.niveau} - "
            f"jour {jour_courant}, heure {heure_courante}"
        )
        fig.canvas.draw_idle()

    slider_jour.on_changed(rafraichir)
    slider_heure.on_changed(rafraichir)

    if sauvegarde:
        fig.savefig(sauvegarde, dpi=180, bbox_inches="tight")
    if afficher:
        plt.show()
    return fig, ax, grille


def _maillage_sphere(latitudes, longitudes, donnees_2d):
    """Prepare le maillage sphere et ferme la couture longitude."""
    lon_ferme = np.append(longitudes, longitudes[0] + 360)
    donnees_fermees = np.concatenate((donnees_2d, donnees_2d[:, 0:1]), axis=1)
    lon_rad = np.radians(lon_ferme)
    lat_rad = np.radians(latitudes)
    lon_mesh, lat_mesh = np.meshgrid(lon_rad, lat_rad)
    x = np.cos(lat_mesh) * np.cos(lon_mesh)
    y = np.cos(lat_mesh) * np.sin(lon_mesh)
    z = np.sin(lat_mesh)
    return x, y, z, donnees_fermees


def creer_sphere(
    preferer_haute_resolution=True,
    variante_grille="auto",
    jour=0,
    heure=0,
    afficher=True,
    sauvegarde=None,
):
    """Cree une figure sphere 3D avec sliders jour/heure et contours."""
    grille = charger_grille_temperature(preferer_haute_resolution, variante_grille)
    indice = indice_temps(grille, jour, heure)
    tranche = temperature_celsius(grille, indice)
    x, y, z, tranche_fermee = _maillage_sphere(
        grille.latitudes, grille.longitudes, tranche
    )

    fig = plt.figure(figsize=(10, 9))
    plt.subplots_adjust(bottom=0.20, top=0.92)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_xlim([-1.08, 1.08])
    ax.set_ylim([-1.08, 1.08])
    ax.set_zlim([-1.08, 1.08])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    normalisation = Normalize(vmin=-50, vmax=50)
    palette = cm.inferno
    surface = ax.plot_surface(
        x,
        y,
        z,
        facecolors=palette(normalisation(tranche_fermee)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
        edgecolor="none",
    )
    tracer_contours_sphere(ax)
    titre = fig.suptitle(
        f"Température de surface - {grille.niveau} - jour {jour}, heure {heure}",
        fontsize=14,
    )
    mappable = cm.ScalarMappable(cmap=palette, norm=normalisation)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.55, aspect=12, pad=0.01)
    colorbar.set_label("Température de surface (°C)")

    axe_jour = plt.axes([0.2, 0.10, 0.6, 0.03])
    slider_jour = Slider(
        axe_jour,
        "Jour",
        0,
        nombre_jours_affichables(grille) - 1,
        valinit=jour,
        valstep=1,
    )
    axe_heure = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_heure = Slider(axe_heure, "Heure", 0, 23, valinit=heure, valstep=1)

    def rafraichir(_):
        jour_courant = int(slider_jour.val)
        heure_courante = int(slider_heure.val)
        idx = indice_temps(grille, jour_courant, heure_courante)
        nouvelle_tranche = temperature_celsius(grille, idx)
        _, _, _, nouvelle_fermee = _maillage_sphere(
            grille.latitudes, grille.longitudes, nouvelle_tranche
        )
        nouvelles_couleurs = palette(normalisation(nouvelle_fermee))
        surface.set_facecolors(nouvelles_couleurs[:-1, :-1, :].reshape(-1, 4))
        titre.set_text(
            f"Température de surface - {grille.niveau} - "
            f"jour {jour_courant}, heure {heure_courante}"
        )
        fig.canvas.draw_idle()

    slider_jour.on_changed(rafraichir)
    slider_heure.on_changed(rafraichir)

    if sauvegarde:
        fig.savefig(sauvegarde, dpi=180, bbox_inches="tight")
    if afficher:
        plt.show()
    return fig, ax, grille
