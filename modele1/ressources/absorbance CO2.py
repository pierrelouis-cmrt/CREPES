# Modèle simplifié de l'absorption infrarouge du CO2.

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sys
import warnings
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "données"


def default_matplotlib_cache_dir() -> Path:
    """Cache Matplotlib stable, sans toucher au cache RADIS."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "CREPES" / "absorbance_co2" / "matplotlib"
        return Path.home() / "AppData" / "Local" / "CREPES" / "absorbance_co2" / "matplotlib"

    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "crepes" / "absorbance_co2" / "matplotlib"
    return Path.home() / ".cache" / "crepes" / "absorbance_co2" / "matplotlib"


def iter_strings(obj):
    """Parcourt récursivement un objet JSON et renvoie toutes les chaînes."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from iter_strings(value)


def find_registered_hitran_co2_file() -> Path | None:
    """
    Trouve le fichier HITRAN CO2 déjà enregistré par RADIS dans ~/radis.json.

    Exemple trouvé :
    C:/Users/melvi/.radisdb/hitran/co2.h5
    """
    radis_json = Path.home() / "radis.json"

    if not radis_json.exists():
        return None

    try:
        data = json.loads(radis_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    candidates = []

    for value in iter_strings(data):
        normalized = value.replace("\\", "/").lower()
        if "hitran" in normalized and normalized.endswith("co2.h5"):
            candidates.append(Path(value).expanduser())

    if not candidates:
        return None

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def resolve_radis_download_dir() -> Path:
    """
    Utilise le cache RADIS déjà enregistré si possible.

    Sinon, utilise le cache standard de RADIS :
    ~/.radisdb

    Ça évite les conflits du type :
    HITRAN-CO2 déjà enregistré dans radis.json mais pas dans le dossier attendu.
    """
    registered_co2_file = find_registered_hitran_co2_file()

    if registered_co2_file is not None:
        # Exemple :
        # C:/Users/melvi/.radisdb/hitran/co2.h5
        # -> DEFAULT_DOWNLOAD_PATH = C:/Users/melvi/.radisdb
        return registered_co2_file.parent.parent

    return Path.home() / ".radisdb"


MPL_CACHE_DIR = default_matplotlib_cache_dir()
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))


import numpy as np
import radis
from radis import calc_spectrum
from radis.misc.warning import LinestrengthCutoffWarning, MissingReferenceWarning
from scipy.interpolate import interp1d


RADIS_DOWNLOAD_DIR = resolve_radis_download_dir()
RADIS_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
(RADIS_DOWNLOAD_DIR / "hitran").mkdir(parents=True, exist_ok=True)

radis.config["DEFAULT_DOWNLOAD_PATH"] = str(RADIS_DOWNLOAD_DIR)
radis.config["ALLOW_OVERWRITE"] = True


BANDS_CM_1 = [
    (600, 760),      # bande à 667 cm-1 environ, soit ~ 15 µm
    (1200, 1500),    # bande à 1388 cm-1 environ, soit ~ 7.2 µm
    (2100, 2450),    # bande à 2349 cm-1 environ, soit ~ 4.3 µm
]

BANDES_MODELES_1_2_UM = (
    ("CO2_15um", 14.25, 15.75),
    ("CO2_4_3um", 4.20, 4.35),
)


def prepare_radis_cache(regen_cache: bool = False) -> None:
    """
    Prépare le cache RADIS sans imposer un nouveau chemin.

    Si --regen-cache est utilisé, on supprime seulement le dossier HITRAN actif.
    """
    if regen_cache:
        shutil.rmtree(RADIS_DOWNLOAD_DIR / "hitran", ignore_errors=True)

    (RADIS_DOWNLOAD_DIR / "hitran").mkdir(parents=True, exist_ok=True)


def make_cross_section_co2_all_bands(regen_cache: bool = False):
    prepare_radis_cache(regen_cache=regen_cache)

    all_wavelengths = []
    all_absorbance = []

    for wmin, wmax in BANDS_CM_1:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LinestrengthCutoffWarning)
            warnings.filterwarnings("ignore", category=MissingReferenceWarning)
            warnings.filterwarnings("ignore", message=".*Missing doi reference.*")
            warnings.filterwarnings("ignore", message=".*Reference databank.*")
            warnings.filterwarnings(
                "ignore",
                message=".*Estimated error after discarding lines.*",
            )

            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    spectrum = calc_spectrum(
                        wmin=wmin,
                        wmax=wmax,
                        molecule="CO2",
                        isotope="1,2,3",
                        Tgas=255,
                        pressure=1.013,
                        mole_fraction=425e-6,  # 425 ppm
                        path_length=100,        # 1 m = 100 cm
                        databank="hitran",
                        verbose=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "RADIS n'a pas pu charger les données HITRAN du CO2.\n"
                        "Au premier lancement, il faut une connexion Internet.\n"
                        f"Cache RADIS utilisé : {RADIS_DOWNLOAD_DIR}\n"
                        "Si le problème persiste, supprime le fichier ~/radis.json "
                        "puis relance le script."
                    ) from exc

        wavelength_nm, absorbance = spectrum.get("absorbance", wunit="nm")
        wavelength_um = wavelength_nm * 1e-3

        all_wavelengths.append(wavelength_um)
        all_absorbance.append(absorbance)

    wavelengths = np.concatenate(all_wavelengths)
    absorbances = np.concatenate(all_absorbance)

    sort_idx = np.argsort(wavelengths)

    return interp1d(
        wavelengths[sort_idx],
        absorbances[sort_idx],
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )


def moyenne_trapezes(y, x) -> float:
    """Calcule une moyenne par intégration trapézoïdale."""
    if hasattr(np, "trapezoid"):
        integrale = np.trapezoid(y, x)
    else:
        integrale = np.trapz(y, x)

    return float(integrale / (x[-1] - x[0]))


def calculer_absorbances_moyennes(absorption_co2, points_par_bande: int = 2_000):
    """Calcule l'absorbance moyenne sur les bandes utiles au modèle."""
    moyennes = []

    for nom, longueur_onde_min_um, longueur_onde_max_um in BANDES_MODELES_1_2_UM:
        longueurs_onde_um = np.linspace(
            longueur_onde_min_um,
            longueur_onde_max_um,
            points_par_bande,
        )

        absorbances = absorption_co2(longueurs_onde_um)
        absorbance_moyenne = moyenne_trapezes(absorbances, longueurs_onde_um)

        moyennes.append(
            (
                nom,
                longueur_onde_min_um,
                longueur_onde_max_um,
                absorbance_moyenne,
            )
        )

    return tuple(moyennes)


def afficher_absorbances_moyennes(moyennes) -> None:
    print("absorbances_moyennes_modeles_1_2")
    print("bande, intervalle_um, absorbance_moyenne")

    for nom, longueur_onde_min_um, longueur_onde_max_um, absorbance_moyenne in moyennes:
        print(
            f"{nom}, "
            f"{longueur_onde_min_um:.2f}-{longueur_onde_max_um:.2f}, "
            f"{absorbance_moyenne:.6f}"
        )


def is_headless_environment() -> bool:
    """Détecte un environnement sans affichage graphique."""
    return bool(
        os.environ.get("CI")
        or os.environ.get("CODEX_CI")
        or os.environ.get("CODEX_SANDBOX")
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
    )


def build_plot(absorption_co2, points: int, moyennes, *, use_file_backend: bool):
    if use_file_backend:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    wavelength_um = np.linspace(4, 17, points)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        wavelength_um,
        absorption_co2(wavelength_um),
        color="steelblue",
        linewidth=0.8,
    )

    for nom, longueur_onde_min_um, longueur_onde_max_um, absorbance_moyenne in moyennes:
        ax.axvspan(
            longueur_onde_min_um,
            longueur_onde_max_um,
            color="orange",
            alpha=0.15,
        )

        ax.hlines(
            absorbance_moyenne,
            longueur_onde_min_um,
            longueur_onde_max_um,
            color="darkorange",
            linewidth=1.2,
        )

        texte_x = (longueur_onde_min_um + longueur_onde_max_um) / 2
        alignement = "center"

        if longueur_onde_max_um < 5.0:
            texte_x = longueur_onde_max_um + 0.08
            alignement = "left"

        libelle = (
            nom.replace("_4_3um", " 4.3 µm")
            .replace("_15um", " 15 µm")
            .replace("_", " ")
        )

        ax.text(
            texte_x,
            absorbance_moyenne,
            f"{libelle}: {absorbance_moyenne:.2f}",
            ha=alignement,
            va="bottom",
            fontsize=8,
            color="darkorange",
        )

    ax.set_xlabel("Longueur d'onde (µm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Absorption du CO₂ (425 ppm, 1 m)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, plt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace l'absorbance infrarouge du CO2 avec RADIS/HITRAN."
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="calcule la courbe sans ouvrir de fenêtre graphique",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="enregistre le graphique dans ce fichier",
    )

    parser.add_argument(
        "--regen-cache",
        action="store_true",
        help="supprime et régénère le cache HITRAN actif",
    )

    parser.add_argument(
        "--points",
        type=int,
        default=10_000,
        help="nombre de points de la courbe tracée",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.points <= 1:
        raise ValueError("--points doit être supérieur à 1.")

    absorption_co2 = make_cross_section_co2_all_bands(
        regen_cache=args.regen_cache
    )

    absorbances_moyennes = calculer_absorbances_moyennes(absorption_co2)
    afficher_absorbances_moyennes(absorbances_moyennes)

    headless = is_headless_environment()

    output_path = args.output
    if (headless or args.no_plot) and output_path is None:
        output_path = OUTPUT_DIR / "absorbance_CO2.png"

    fig, plt = build_plot(
        absorption_co2,
        points=args.points,
        moyennes=absorbances_moyennes,
        use_file_backend=headless or args.no_plot,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Graphique enregistré : {output_path}")

    if args.no_plot or headless:
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
