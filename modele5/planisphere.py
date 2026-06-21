"""Planisphere des sorties du modele 5.

Par defaut, le script affiche le bilan total de la derniere simulation :
temperature finale, variation de temperature et flux horizontal moyen. Les
puissances globales sont calculees avec l'aire reelle des mailles de latitude.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "crepes_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

if "--no-show" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    import shapefile

    SHAPEFILE_DISPONIBLE = True
except ImportError:
    shapefile = None
    SHAPEFILE_DISPONIBLE = False


RAYON_TERRE_M = 6_371_000.0
SCRIPT_DIR = Path(__file__).resolve().parent
PROJET_DIR = SCRIPT_DIR.parent
SORTIES_DEFAUT = SCRIPT_DIR / "sorties"
COASTLINE_SHP = (
    PROJET_DIR
    / "modele0_maintenance"
    / "ressources"
    / "cotes"
    / "ne_10m_coastline.shp"
)


def _metadata(npz) -> dict:
    if "metadata_json" not in npz.files:
        return {}
    try:
        return json.loads(str(npz["metadata_json"].item()))
    except (ValueError, TypeError, json.JSONDecodeError):
        return {}


def _normaliser_axes(valeurs, latitudes, longitudes):
    # On remet les cartes dans l'ordre habituel : sud-nord et ouest-est.
    ordre_lat = np.argsort(latitudes)
    ordre_lon = np.argsort(((longitudes + 180.0) % 360.0) - 180.0)
    longitudes = ((longitudes + 180.0) % 360.0) - 180.0
    valeurs = {
        nom: tableau[..., ordre_lat, :][..., :, ordre_lon]
        for nom, tableau in valeurs.items()
    }
    return valeurs, latitudes[ordre_lat], longitudes[ordre_lon]


def charger_sortie(chemin):
    """Charge une sortie modele 5 et les diagnostics necessaires au bilan."""

    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable: {chemin}")
    with np.load(chemin, allow_pickle=False) as npz:
        obligatoires = ("temperature_surface_k", "lat_deg", "lon_deg")
        absents = [nom for nom in obligatoires if nom not in npz.files]
        if absents:
            raise KeyError(f"Variables absentes de {chemin.name}: {', '.join(absents)}")
        temperature = np.asarray(npz["temperature_surface_k"], dtype=np.float64)
        latitudes = np.asarray(npz["lat_deg"], dtype=np.float64)
        longitudes = np.asarray(npz["lon_deg"], dtype=np.float64)
        if temperature.ndim != 3 or temperature.shape[1:] != (latitudes.size, longitudes.size):
            raise ValueError("temperature_surface_k doit avoir la forme [temps, lat, lon].")
        diagnostics = {}
        for nom in (
            "flux_horizontal_net_surface_moyen_w_m2",
            "flux_horizontal_atmosphere_moyen_w_m2",
            "flux_net_surface_moyen_w_m2",
        ):
            if nom in npz.files:
                tableau = np.asarray(npz[nom], dtype=np.float64)
                if tableau.shape == (latitudes.size, longitudes.size):
                    diagnostics[nom] = tableau
        metadata = _metadata(npz)

    valeurs = {"temperature": temperature, **diagnostics}
    valeurs, latitudes, longitudes = _normaliser_axes(valeurs, latitudes, longitudes)
    return {
        "chemin": chemin,
        "temperature_surface_k": valeurs.pop("temperature"),
        "diagnostics": valeurs,
        "lat_deg": latitudes,
        "lon_deg": longitudes,
        "metadata": metadata,
    }


def _pas(valeurs, defaut=5.0):
    if len(valeurs) < 2:
        return defaut
    differences = np.diff(np.sort(valeurs))
    differences = differences[differences > 0.0]
    return float(np.median(differences)) if differences.size else defaut


def _extent(latitudes, longitudes):
    pas_lat = _pas(latitudes)
    pas_lon = _pas(longitudes)
    return [
        float(longitudes.min() - pas_lon / 2.0),
        float(longitudes.max() + pas_lon / 2.0),
        float(max(-90.0, latitudes.min() - pas_lat / 2.0)),
        float(min(90.0, latitudes.max() + pas_lat / 2.0)),
    ]


def _aires_mailles(latitudes, longitudes):
    pas_lat = np.deg2rad(_pas(latitudes))
    pas_lon = np.deg2rad(_pas(longitudes))
    lat = np.deg2rad(latitudes)
    sud = np.maximum(lat - pas_lat / 2.0, -np.pi / 2.0)
    nord = np.minimum(lat + pas_lat / 2.0, np.pi / 2.0)
    # Une maille proche des poles couvre moins de surface qu'une maille tropicale.
    aire_lat = RAYON_TERRE_M**2 * pas_lon * (np.sin(nord) - np.sin(sud))
    return np.repeat(aire_lat[:, None], len(longitudes), axis=1)


def puissance_totale_pw(flux_w_m2, latitudes, longitudes):
    """Puissance integree, en petawatts, du flux fourni."""

    return float(np.nansum(flux_w_m2 * _aires_mailles(latitudes, longitudes)) / 1e15)


def _segments_cotes():
    if not SHAPEFILE_DISPONIBLE or not COASTLINE_SHP.exists():
        return []
    segments = []
    for forme in shapefile.Reader(str(COASTLINE_SHP)).shapes():
        bornes = list(forme.parts) + [len(forme.points)]
        for debut, fin in zip(bornes[:-1], bornes[1:]):
            points = forme.points[debut:fin]
            segment, dernier_lon = [], None
            for lon, lat in points:
                if dernier_lon is not None and abs(lon - dernier_lon) > 180.0:
                    if len(segment) > 1:
                        segments.append(segment)
                    segment = []
                segment.append((lon, lat))
                dernier_lon = lon
            if len(segment) > 1:
                segments.append(segment)
    return segments


def _tracer_cotes(ax):
    segments = _segments_cotes()
    if segments:
        ax.add_collection(LineCollection(segments, colors="black", linewidths=0.35, zorder=4))


def _limites(valeurs, divergeant=False):
    valeurs_finies = valeurs[np.isfinite(valeurs)]
    if valeurs_finies.size == 0:
        return -1.0, 1.0
    if divergeant:
        borne = max(abs(np.nanpercentile(valeurs_finies, 2)), abs(np.nanpercentile(valeurs_finies, 98)))
        return -borne or -1.0, borne or 1.0
    bas, haut = np.nanpercentile(valeurs_finies, (2, 98))
    if bas == haut:
        return bas - 1.0, haut + 1.0
    return float(bas), float(haut)


def _tracer_carte(ax, donnees, extent, titre, label, cmap, divergeant=False):
    vmin, vmax = _limites(donnees, divergeant=divergeant)
    # imshow suffit ici : la grille est reguliere apres normalisation des axes.
    image = ax.imshow(
        donnees,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    _tracer_cotes(ax)
    ax.set_title(titre)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.35, zorder=2)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label(label)


def creer_planisphere_total(sortie, sauvegarde=None, afficher=True):
    """Cree le planisphere du bilan total de la simulation modele 5."""

    temperature = sortie["temperature_surface_k"]
    diagnostic = sortie["diagnostics"]
    if "flux_horizontal_net_surface_moyen_w_m2" not in diagnostic:
        raise KeyError("Le fichier ne contient pas le flux horizontal du modele 5.")
    latitudes, longitudes = sortie["lat_deg"], sortie["lon_deg"]
    extent = _extent(latitudes, longitudes)
    final_c = temperature[-1] - 273.15
    delta_t = temperature[-1] - temperature[0]
    flux_horizontal = diagnostic["flux_horizontal_net_surface_moyen_w_m2"]
    # Le flux moyen devient une puissance globale grace aux aires de mailles.
    puissance_horizontale = puissance_totale_pw(flux_horizontal, latitudes, longitudes)

    fig, axes = plt.subplots(3, 1, figsize=(15, 15), constrained_layout=True)
    _tracer_carte(
        axes[0], final_c, extent, "Temperature de surface finale", "Temperature (deg C)", "inferno"
    )
    _tracer_carte(
        axes[1], delta_t, extent, "Variation sur la simulation", "Delta T (K)", "coolwarm", True
    )
    _tracer_carte(
        axes[2],
        flux_horizontal,
        extent,
        "Flux horizontal moyen recu par la surface",
        "Q_horizontal (W m-2)",
        "RdBu_r",
        True,
    )
    nom_modele = sortie["metadata"].get("modele", "modele5")
    fig.suptitle(
        f"Bilan total {nom_modele} — {sortie['chemin'].name}\n"
        f"Puissance horizontale nette integree : {puissance_horizontale:.3e} PW",
        fontsize=15,
    )
    if sauvegarde:
        sauvegarde = Path(sauvegarde)
        sauvegarde.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sauvegarde, dpi=180)
    if afficher:
        plt.show()
    return fig, axes


def _dernier_fichier(dossier):
    fichiers = list(Path(dossier).glob("*.npz"))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier .npz trouve dans {dossier}")
    return max(fichiers, key=lambda chemin: chemin.stat().st_mtime)


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Planisphere du bilan total du modele 5")
    parseur.add_argument("--fichier", type=Path, default=None)
    parseur.add_argument("--sorties", type=Path, default=SORTIES_DEFAUT)
    parseur.add_argument("--save", type=Path, default=None, help="PNG a ecrire.")
    parseur.add_argument("--no-show", action="store_true", help="Genere seulement le PNG.")
    return parseur


def main():
    args = construire_parseur().parse_args()
    try:
        sortie = charger_sortie(args.fichier or _dernier_fichier(args.sorties))
        fig, _ = creer_planisphere_total(
            sortie, sauvegarde=args.save, afficher=not args.no_show
        )
        if args.no_show:
            plt.close(fig)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
