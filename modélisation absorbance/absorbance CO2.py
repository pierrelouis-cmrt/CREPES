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
CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = CACHE_DIR / "xdg"
RADIS_CACHE_DIR = CACHE_DIR / "radisdb"
HITRAN_DIR = RADIS_CACHE_DIR / "hitran"
CO2_DB_PATH = HITRAN_DIR / "CO2.h5"
LOCAL_DATABANK_NAME = "CREPES-ABSORBANCE-CO2"

for directory in (MPL_CACHE_DIR, XDG_CACHE_DIR, HITRAN_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import numpy as np
import radis
from radis import calc_spectrum
from radis.io.hitran import fetch_hitran
from radis.misc.warning import LinestrengthCutoffWarning, MissingReferenceWarning
from scipy.interpolate import interp1d


radis.config["DEFAULT_DOWNLOAD_PATH"] = str(RADIS_CACHE_DIR)
radis.config["ALLOW_OVERWRITE"] = True

BANDS_CM_1 = [
    (600, 760),  # bande a 667 cm-1 ~ 15 micrometres
    (1200, 1500),  # bande a 1388 cm-1 ~ 7.2 micrometres
    (2100, 2450),  # bande a 2349 cm-1 ~ 4.3 micrometres
]


def _remove_incomplete_hapi_downloads() -> None:
    """Remove local HAPI headers that were left without data files."""
    download_dir = HITRAN_DIR / "downloads__can_be_deleted" / "CO2"
    if not download_dir.exists():
        return

    for header_path in download_dir.glob("*.header"):
        data_path = header_path.with_suffix(".data")
        if not data_path.exists():
            header_path.unlink()


def _ensure_local_co2_databank(regen_cache: bool = False) -> Path:
    """Return a local RADIS/HITRAN CO2 database path, downloading it if needed."""
    if regen_cache:
        if CO2_DB_PATH.exists():
            CO2_DB_PATH.unlink()
        shutil.rmtree(HITRAN_DIR / "downloads__can_be_deleted", ignore_errors=True)

    if CO2_DB_PATH.exists():
        return CO2_DB_PATH

    _remove_incomplete_hapi_downloads()

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, local_paths = fetch_hitran(
                "CO2",
                isotope="1,2,3",
                local_databases=str(HITRAN_DIR),
                databank_name=LOCAL_DATABANK_NAME,
                cache="regen" if regen_cache else True,
                verbose=False,
                return_local_path=True,
                engine="pytables",
                output="pandas",
            )
    except Exception as exc:
        raise RuntimeError(
            "Impossible de préparer la base HITRAN locale pour CO2. "
            "RADIS télécharge ces raies au premier lancement ; vérifie la connexion "
            "réseau, puis relance avec --regen-cache si un téléchargement a été "
            "interrompu."
        ) from exc

    if not local_paths:
        raise RuntimeError("RADIS n'a retourné aucun fichier local HITRAN pour CO2.")

    return Path(local_paths[0])


def make_cross_section_co2_all_bands(regen_cache: bool = False):
    co2_databank = _ensure_local_co2_databank(regen_cache=regen_cache)

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
                spectrum = calc_spectrum(
                    wmin=wmin,
                    wmax=wmax,
                    molecule="CO2",
                    isotope="1,2,3",
                    Tgas=255,
                    pressure=1.013,
                    mole_fraction=400e-6,  # 400 ppm
                    path_length=100,  # 1 m = 100 cm
                    databank=str(co2_databank),
                    verbose=False,
                )
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
