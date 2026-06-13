# Modèle simplifié de l'absorption infrarouge du CO2.
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shutil
import sys
import warnings
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def default_cache_dir() -> Path:
    """Return a simple user cache path that works on Windows, macOS and Linux."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "CREPES" / "absorbance_co2"
        return Path.home() / "AppData" / "Local" / "CREPES" / "absorbance_co2"

    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "crepes" / "absorbance_co2"
    return Path.home() / ".cache" / "crepes" / "absorbance_co2"


def setup_cache_dirs() -> tuple[Path, Path, Path]:
    """Create cache directories, falling back to the script folder if needed."""
    last_error = None
    for cache_dir in (default_cache_dir(), SCRIPT_DIR / ".cache"):
        matplotlib_dir = cache_dir / "matplotlib"
        radis_dir = cache_dir / "radisdb"
        try:
            for directory in (matplotlib_dir, radis_dir):
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            last_error = exc
            continue
        return cache_dir, matplotlib_dir, radis_dir

    raise RuntimeError("Impossible de creer un dossier de cache.") from last_error


CACHE_DIR, MPL_CACHE_DIR, RADIS_CACHE_DIR = setup_cache_dirs()

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import numpy as np
import radis
from radis import calc_spectrum
from radis.misc.warning import LinestrengthCutoffWarning, MissingReferenceWarning
from scipy.interpolate import interp1d


radis.config["DEFAULT_DOWNLOAD_PATH"] = str(RADIS_CACHE_DIR)
radis.config["ALLOW_OVERWRITE"] = True

BANDS_CM_1 = [
    (600, 760),  # bande a 667 cm-1 ~ 15 micrometres
    (1200, 1500),  # bande a 1388 cm-1 ~ 7.2 micrometres
    (2100, 2450),  # bande a 2349 cm-1 ~ 4.3 micrometres
]


def prepare_radis_cache(regen_cache: bool = False) -> None:
    """Prepare RADIS cache; RADIS downloads HITRAN automatically if needed."""
    if regen_cache:
        shutil.rmtree(RADIS_CACHE_DIR, ignore_errors=True)
    RADIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
                "ignore", message=".*Estimated error after discarding lines.*"
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
                        mole_fraction=400e-6,  # 400 ppm
                        path_length=100,  # 1 m = 100 cm
                        databank="hitran",
                        verbose=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "RADIS n'a pas pu charger les donnees HITRAN du CO2. "
                        "Au premier lancement, il faut une connexion Internet. "
                        f"Cache utilise : {RADIS_CACHE_DIR}"
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


def build_plot(absorption_co2, points: int, *, use_file_backend: bool):
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
    ax.set_xlabel("Longueur d'onde (µm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Absorption du CO₂ (400 ppm, 1 m)")
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
        help="enregistre le graphique dans ce fichier au lieu de seulement l'afficher",
    )
    parser.add_argument(
        "--regen-cache",
        action="store_true",
        help="supprime et régénère le cache HITRAN local du projet",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=10_000,
        help="nombre de points de la courbe tracée",
    )
    return parser.parse_args(argv)


def is_headless_environment() -> bool:
    return bool(
        os.environ.get("CI")
        or os.environ.get("CODEX_CI")
        or os.environ.get("CODEX_SANDBOX")
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.points <= 1:
        raise ValueError("--points doit être supérieur à 1.")

    absorption_co2 = make_cross_section_co2_all_bands(regen_cache=args.regen_cache)

    if args.no_plot and not args.output:
        print("Calcul terminé.")
        return 0

    headless = is_headless_environment()
    output_path = args.output
    if headless and output_path is None:
        output_path = SCRIPT_DIR / "absorbance_CO2.png"

    fig, plt = build_plot(
        absorption_co2,
        points=args.points,
        use_file_backend=headless or args.no_plot,
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if args.no_plot or headless:
        plt.close(fig)
        if headless and args.output is None:
            print(f"Graphique enregistré : {output_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
