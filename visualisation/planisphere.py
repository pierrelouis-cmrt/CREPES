"""Planisphere interactive pour les sorties NPZ des modeles 4 et 5.

Le script lit les fichiers produits par ``modele4.modele4``,
``modele4.rapide`` et ``modele5.modele5`` sans modifier leur format. Quand
aucun fichier n'est passe en argument, un petit TUI liste les ``.npz``
disponibles dans ``modele4/sorties`` et ``modele5/sorties``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
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
from matplotlib.widgets import Slider

try:
    import shapefile

    SHAPEFILE_DISPONIBLE = True
except ImportError:
    shapefile = None
    SHAPEFILE_DISPONIBLE = False


VISUALISATION_DIR = Path(__file__).resolve().parent
PROJET_DIR = VISUALISATION_DIR.parent
MODELE4_DIR = PROJET_DIR / "modele4"
MODELE5_DIR = PROJET_DIR / "modele5"
SORTIES_DEFAUT = (MODELE4_DIR / "sorties", MODELE5_DIR / "sorties")
COASTLINE_SHP = (
    PROJET_DIR
    / "modele0_maintenance"
    / "ressources"
    / "cotes"
    / "ne_10m_coastline.shp"
)


@dataclass(frozen=True)
class SortieModele4:
    """Tableaux necessaires a l'affichage d'une sortie modele 4 ou 5."""

    chemin: Path
    variable_nom: str
    valeurs: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    temps_s: np.ndarray
    mois: np.ndarray
    metadata: dict
    jours_fichier: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64),
    )
    heures_fichier: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64),
    )

    @property
    def nombre_images(self) -> int:
        return int(self.valeurs.shape[0])

    @property
    def jour_max(self) -> int:
        if self.temps_s.size == 0:
            return max(0, self.nombre_images - 1)
        return max(0, int(np.floor(float(np.nanmax(self.temps_s)) / 86400.0)))


def _charger_metadata(npz) -> dict:
    if "metadata_json" not in npz.files:
        return {}
    try:
        valeur = npz["metadata_json"].item()
        return json.loads(str(valeur))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _axe_temps(npz, nombre_images: int) -> np.ndarray:
    if "temps_s" in npz.files:
        temps = np.asarray(npz["temps_s"], dtype=np.float64)
        if temps.shape == (nombre_images,):
            return temps
    if "jours" in npz.files and "heures" in npz.files:
        jours = np.asarray(npz["jours"], dtype=np.float64)
        heures = np.asarray(npz["heures"], dtype=np.float64)
        if jours.shape == (nombre_images,) and heures.shape == (nombre_images,):
            if np.allclose(jours, np.round(jours)):
                return (jours - 1.0) * 86400.0 + heures * 3600.0
            return (jours - 1.0) * 86400.0
    return np.arange(nombre_images, dtype=np.float64) * 86400.0


def _normaliser_axes(
    valeurs: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if latitudes.size > 1:
        ordre_lat = np.argsort(latitudes)
        if not np.array_equal(ordre_lat, np.arange(latitudes.size)):
            latitudes = latitudes[ordre_lat]
            valeurs = valeurs[:, ordre_lat, :]

    if longitudes.size > 0 and float(np.nanmax(longitudes)) > 180.0:
        longitudes = ((longitudes + 180.0) % 360.0) - 180.0

    if longitudes.size > 1:
        ordre_lon = np.argsort(longitudes)
        if not np.array_equal(ordre_lon, np.arange(longitudes.size)):
            longitudes = longitudes[ordre_lon]
            valeurs = valeurs[:, :, ordre_lon]

    return valeurs, latitudes, longitudes


def charger_sortie_npz(
    chemin: Path,
    variable_nom: str = "temperature_surface_k",
) -> SortieModele4:
    """Charge un fichier ``.npz`` compatible avec les sorties du modele 4."""

    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable: {chemin}")

    with np.load(chemin, allow_pickle=False) as npz:
        if variable_nom not in npz.files:
            disponibles = ", ".join(npz.files)
            raise KeyError(
                f"Variable {variable_nom!r} absente de {chemin.name}. "
                f"Variables disponibles: {disponibles}"
            )
        for axe in ("lat_deg", "lon_deg"):
            if axe not in npz.files:
                raise KeyError(f"Axe {axe!r} absent de {chemin.name}")

        valeurs = np.asarray(npz[variable_nom])
        if valeurs.ndim != 3:
            raise ValueError(
                f"{variable_nom!r} doit avoir la forme [temps, lat, lon], "
                f"forme recue: {valeurs.shape}"
            )

        latitudes = np.asarray(npz["lat_deg"], dtype=np.float64)
        longitudes = np.asarray(npz["lon_deg"], dtype=np.float64)
        if valeurs.shape[1] != latitudes.size or valeurs.shape[2] != longitudes.size:
            raise ValueError(
                "Dimensions incompatibles entre la variable et lat_deg/lon_deg: "
                f"{valeurs.shape}, lat={latitudes.size}, lon={longitudes.size}"
            )

        temps_s = _axe_temps(npz, valeurs.shape[0])
        jours_fichier = (
            np.asarray(npz["jours"], dtype=np.float64)
            if "jours" in npz.files and npz["jours"].shape == (valeurs.shape[0],)
            else np.array([], dtype=np.float64)
        )
        heures_fichier = (
            np.asarray(npz["heures"], dtype=np.float64)
            if "heures" in npz.files and npz["heures"].shape == (valeurs.shape[0],)
            else np.array([], dtype=np.float64)
        )
        mois = (
            np.asarray(npz["mois"], dtype=np.int16)
            if "mois" in npz.files and npz["mois"].shape == (valeurs.shape[0],)
            else np.array([], dtype=np.int16)
        )
        metadata = _charger_metadata(npz)

    ordre_temps = np.argsort(temps_s)
    if not np.array_equal(ordre_temps, np.arange(temps_s.size)):
        valeurs = valeurs[ordre_temps, :, :]
        temps_s = temps_s[ordre_temps]
        if jours_fichier.shape == ordre_temps.shape:
            jours_fichier = jours_fichier[ordre_temps]
        if heures_fichier.shape == ordre_temps.shape:
            heures_fichier = heures_fichier[ordre_temps]
        if mois.shape == ordre_temps.shape:
            mois = mois[ordre_temps]

    valeurs, latitudes, longitudes = _normaliser_axes(valeurs, latitudes, longitudes)
    return SortieModele4(
        chemin=chemin,
        variable_nom=variable_nom,
        valeurs=valeurs,
        latitudes=latitudes,
        longitudes=longitudes,
        temps_s=temps_s,
        jours_fichier=jours_fichier,
        heures_fichier=heures_fichier,
        mois=mois,
        metadata=metadata,
    )


def _segments_depuis_points(points):
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


def segments_cotes_lonlat() -> list[list[tuple[float, float]]]:
    """Retourne les segments de cotes depuis le shapefile du modele 0."""

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


def _pas_axe(valeurs: np.ndarray, defaut: float) -> float:
    if valeurs.size < 2:
        return defaut
    differences = np.diff(np.sort(valeurs))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if differences.size == 0:
        return defaut
    return float(np.median(differences))


def _extent(longitudes: np.ndarray, latitudes: np.ndarray) -> list[float]:
    pas_lon = _pas_axe(longitudes, 5.0)
    pas_lat = _pas_axe(latitudes, 5.0)
    return [
        float(np.nanmin(longitudes) - pas_lon / 2.0),
        float(np.nanmax(longitudes) + pas_lon / 2.0),
        float(np.nanmin(latitudes) - pas_lat / 2.0),
        float(np.nanmax(latitudes) + pas_lat / 2.0),
    ]


def _tranche_affichee(sortie: SortieModele4, indice: int) -> tuple[np.ndarray, str]:
    tranche = np.asarray(sortie.valeurs[indice, :, :], dtype=np.float64)
    if sortie.variable_nom.endswith("_k"):
        return tranche - 273.15, "deg C"
    return tranche, sortie.variable_nom


def _bornes_couleur(
    sortie: SortieModele4,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    donnees, _ = _tranche_affichee(sortie, 0)
    if sortie.nombre_images > 1:
        donnees = np.asarray(sortie.valeurs, dtype=np.float64)
        if sortie.variable_nom.endswith("_k"):
            donnees = donnees - 273.15

    valeurs_finies = donnees[np.isfinite(donnees)]
    if valeurs_finies.size == 0:
        return (-50.0 if vmin is None else vmin, 50.0 if vmax is None else vmax)

    bas = float(np.nanpercentile(valeurs_finies, 2.0)) if vmin is None else vmin
    haut = float(np.nanpercentile(valeurs_finies, 98.0)) if vmax is None else vmax
    if bas == haut:
        bas -= 1.0
        haut += 1.0
    marge = 0.05 * (haut - bas)
    if vmin is None:
        bas -= marge
    if vmax is None:
        haut += marge
    return bas, haut


def indice_pour_jour_heure(sortie: SortieModele4, jour: int, heure: int) -> int:
    cible_s = float(jour) * 86400.0 + float(heure) * 3600.0
    return int(np.nanargmin(np.abs(sortie.temps_s - cible_s)))


def indice_pour_temps_s(sortie: SortieModele4, temps_s: float) -> int:
    return int(np.nanargmin(np.abs(sortie.temps_s - float(temps_s))))


def _valeurs_uniques(valeurs: np.ndarray) -> np.ndarray:
    valeurs_finies = np.asarray(valeurs, dtype=np.float64)
    valeurs_finies = valeurs_finies[np.isfinite(valeurs_finies)]
    if valeurs_finies.size == 0:
        return np.array([], dtype=np.float64)
    return np.unique(np.round(valeurs_finies, 6))


def _resolution_horaire(sortie: SortieModele4) -> bool:
    if sortie.nombre_images <= 1:
        return False
    if (
        sortie.heures_fichier.shape == (sortie.nombre_images,)
        and _valeurs_uniques(sortie.heures_fichier).size > 1
    ):
        return True
    secondes_dans_jour = np.mod(sortie.temps_s, 86400.0)
    return _valeurs_uniques(secondes_dans_jour).size > 1


def _mode_slider_temps(sortie: SortieModele4) -> str | None:
    if sortie.nombre_images <= 1:
        return None
    if _resolution_horaire(sortie):
        return "heure"
    if _valeurs_uniques(np.floor(sortie.temps_s / 86400.0)).size > 1:
        return "jour"
    return None


def _heures_ecoulees(sortie: SortieModele4) -> np.ndarray:
    return np.asarray(sortie.temps_s, dtype=np.float64) / 3600.0


def _valeur_slider_initiale(valeurs: np.ndarray, cible: float) -> float:
    valeurs = np.asarray(valeurs, dtype=np.float64)
    valeurs = valeurs[np.isfinite(valeurs)]
    if valeurs.size == 0:
        return float(cible)
    return float(valeurs[np.nanargmin(np.abs(valeurs - cible))])


def _libelle_temps(sortie: SortieModele4, indice: int) -> str:
    temps = float(sortie.temps_s[indice])
    jour = int(np.floor(temps / 86400.0))
    heure = (temps / 3600.0) % 24.0
    morceaux = [f"jour {jour}", f"heure {heure:g}"]
    if sortie.mois.shape == (sortie.nombre_images,):
        morceaux.append(f"mois {int(sortie.mois[indice])}")
    if _resolution_horaire(sortie):
        heure_ecoulee = temps / 3600.0
        morceaux.append(f"heure ecoulee {heure_ecoulee:g}")
    return ", ".join(morceaux)


def _titre(sortie: SortieModele4, demande: str, indice: int) -> str:
    modele = sortie.metadata.get("modele", "modele4")
    mode = sortie.metadata.get("mode_sortie")
    suffixe_mode = f" - {mode}" if mode else ""
    return (
        f"Temperature de surface - {sortie.chemin.name} - {modele}{suffixe_mode}\n"
        f"demande: {demande} | image: {_libelle_temps(sortie, indice)}"
    )


def _libelle_colorbar(sortie: SortieModele4, unite: str) -> str:
    if sortie.variable_nom == "temperature_surface_k":
        return f"Temperature de surface ({unite})"
    return f"{sortie.variable_nom} ({unite})"


def creer_planisphere(
    sortie: SortieModele4,
    jour: int = 0,
    heure: int = 0,
    afficher: bool = True,
    sauvegarde: Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Cree une figure planisphere avec le slider temporel utile et les contours."""

    jour = max(0, min(int(jour), max(1, sortie.jour_max)))
    heure = max(0, min(int(heure), 23))
    mode_slider = _mode_slider_temps(sortie)
    if mode_slider == "heure":
        heures = _heures_ecoulees(sortie)
        heure_initiale = _valeur_slider_initiale(heures, jour * 24.0 + heure)
        indice = indice_pour_temps_s(sortie, heure_initiale * 3600.0)
        demande = f"heure {heure_initiale:g}"
    else:
        indice = indice_pour_jour_heure(sortie, jour, heure)
        demande = f"jour {jour}" if mode_slider == "jour" else _libelle_temps(sortie, indice)
    tranche, unite = _tranche_affichee(sortie, indice)
    bas, haut = _bornes_couleur(sortie, vmin, vmax)
    limites = _extent(sortie.longitudes, sortie.latitudes)

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.16 if mode_slider else 0.10, top=0.88)
    image = ax.imshow(
        tranche,
        origin="lower",
        extent=limites,
        cmap="inferno",
        vmin=bas,
        vmax=haut,
        interpolation="bilinear",
        aspect="auto",
        zorder=1,
    )
    tracer_contours_planisphere(ax)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(limites[0], limites[1])
    ax.set_ylim(limites[2], limites[3])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=2)
    titre = ax.set_title(_titre(sortie, demande, indice), fontsize=13)
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", fraction=0.03, pad=0.04)
    colorbar.set_label(_libelle_colorbar(sortie, unite))

    sliders = []
    if mode_slider == "jour":
        axe_jour = fig.add_axes([0.2, 0.06, 0.6, 0.03])
        slider_jour = Slider(
            axe_jour,
            "Jour",
            0,
            max(1, sortie.jour_max),
            valinit=jour,
            valstep=1,
        )

        def rafraichir_jour(_):
            jour_courant = int(slider_jour.val)
            indice_courant = indice_pour_jour_heure(sortie, jour_courant, 0)
            nouvelle_tranche, _ = _tranche_affichee(sortie, indice_courant)
            image.set_data(nouvelle_tranche)
            titre.set_text(_titre(sortie, f"jour {jour_courant}", indice_courant))
            fig.canvas.draw_idle()

        slider_jour.on_changed(rafraichir_jour)
        sliders.append(slider_jour)
    elif mode_slider == "heure":
        heures = _heures_ecoulees(sortie)
        heures_disponibles = _valeurs_uniques(heures)
        heure_min = float(np.nanmin(heures_disponibles))
        heure_max = float(np.nanmax(heures_disponibles))
        axe_heure = fig.add_axes([0.2, 0.06, 0.6, 0.03])
        slider_heure = Slider(
            axe_heure,
            "Heure",
            heure_min,
            heure_max,
            valinit=heure_initiale,
            valstep=heures_disponibles,
        )

        def rafraichir_heure(_):
            heure_courante = float(slider_heure.val)
            indice_courant = indice_pour_temps_s(sortie, heure_courante * 3600.0)
            nouvelle_tranche, _ = _tranche_affichee(sortie, indice_courant)
            image.set_data(nouvelle_tranche)
            titre.set_text(_titre(sortie, f"heure {heure_courante:g}", indice_courant))
            fig.canvas.draw_idle()

        slider_heure.on_changed(rafraichir_heure)
        sliders.append(slider_heure)
    fig._planisphere_sliders = tuple(sliders)

    if sauvegarde:
        sauvegarde = Path(sauvegarde)
        sauvegarde.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(sauvegarde, dpi=180, bbox_inches="tight")
    if afficher:
        plt.show()
    return fig, ax


def _chemin_affiche(chemin: Path) -> str:
    try:
        return str(chemin.relative_to(PROJET_DIR))
    except ValueError:
        return str(chemin)


def _resumer_npz(chemin: Path, variable_nom: str) -> str:
    try:
        with np.load(chemin, allow_pickle=False) as npz:
            metadata = _charger_metadata(npz)
            modele = metadata.get("modele", "?")
            mode = metadata.get("mode_sortie") or metadata.get("description", "")
            shape = npz[variable_nom].shape if variable_nom in npz.files else "variable absente"
            temps = npz["temps_s"] if "temps_s" in npz.files else np.array([])
            if temps.size:
                duree = f"{float(np.nanmax(temps)) / 86400.0:.2f} j"
            else:
                duree = "duree inconnue"
    except Exception as exc:  # noqa: BLE001 - resume robuste pour le TUI.
        return f"{chemin.name} | lecture impossible: {exc}"

    taille_mo = chemin.stat().st_size / (1024.0 * 1024.0)
    modifie = datetime.fromtimestamp(chemin.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return (
        f"{_chemin_affiche(chemin)} | {modele} {mode} | {variable_nom}{shape} | "
        f"{duree} | {taille_mo:.1f} Mo | modifie {modifie}"
    )


def _normaliser_dossiers(dossiers: Path | str | list[Path] | tuple[Path, ...]) -> list[Path]:
    if isinstance(dossiers, (str, Path)):
        return [Path(dossiers)]
    return [Path(dossier) for dossier in dossiers]


def fichiers_npz_disponibles(dossiers: Path | str | list[Path] | tuple[Path, ...]) -> list[Path]:
    fichiers = []
    for dossier in _normaliser_dossiers(dossiers):
        if dossier.exists():
            fichiers.extend(dossier.glob("*.npz"))
    return sorted(
        fichiers,
        key=lambda chemin: chemin.stat().st_mtime,
        reverse=True,
    )


def choisir_fichier_tui(
    dossiers: Path | str | list[Path] | tuple[Path, ...],
    variable_nom: str,
    interactif: bool = True,
) -> Path:
    dossiers = _normaliser_dossiers(dossiers)
    fichiers = fichiers_npz_disponibles(dossiers)
    if not fichiers:
        dossiers_lisibles = ", ".join(str(dossier) for dossier in dossiers)
        raise FileNotFoundError(f"Aucun fichier .npz trouve dans: {dossiers_lisibles}")

    if not interactif or not sys.stdin.isatty():
        choix = fichiers[0]
        print(f"Selection automatique: {_chemin_affiche(choix)}")
        return choix

    print("Fichiers NPZ disponibles:")
    for indice, chemin in enumerate(fichiers, start=1):
        print(f"  {indice}. {_resumer_npz(chemin, variable_nom)}")

    while True:
        reponse = input(f"Choisir un fichier [1-{len(fichiers)}] (Entree = 1): ").strip()
        if not reponse:
            return fichiers[0]
        try:
            numero = int(reponse)
        except ValueError:
            print("Veuillez entrer un numero.")
            continue
        if 1 <= numero <= len(fichiers):
            return fichiers[numero - 1]
        print("Numero hors limites.")


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Planisphere interactive pour les sorties NPZ des modeles 4 et 5",
    )
    parseur.add_argument(
        "--fichier",
        type=Path,
        default=None,
        help="Fichier .npz a ouvrir. Si absent, le TUI liste les sorties disponibles.",
    )
    parseur.add_argument(
        "--sorties",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Dossiers contenant les .npz proposes par le TUI. "
            "Defaut: modele4/sorties et modele5/sorties."
        ),
    )
    parseur.add_argument("--variable", default="temperature_surface_k")
    parseur.add_argument("--jour", type=int, default=0)
    parseur.add_argument("--heure", type=int, default=0)
    parseur.add_argument("--vmin", type=float, default=None)
    parseur.add_argument("--vmax", type=float, default=None)
    parseur.add_argument("--save", type=Path, default=None)
    parseur.add_argument("--no-show", action="store_true")
    parseur.add_argument(
        "--no-tui",
        action="store_true",
        help="Selectionne automatiquement le .npz le plus recent si --fichier est absent.",
    )
    return parseur


def main() -> int:
    args = construire_parseur().parse_args()
    dossiers_sorties = args.sorties if args.sorties else SORTIES_DEFAUT
    try:
        chemin = args.fichier or choisir_fichier_tui(
            dossiers_sorties,
            args.variable,
            interactif=not args.no_tui,
        )
        sortie = charger_sortie_npz(chemin, variable_nom=args.variable)
        fig, _ = creer_planisphere(
            sortie,
            jour=args.jour,
            heure=args.heure,
            afficher=not args.no_show,
            sauvegarde=args.save,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        if args.no_show:
            plt.close(fig)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"Erreur: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
