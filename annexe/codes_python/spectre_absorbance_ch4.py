"""Calcul du spectre d'absorbance infrarouge du CH4 avec RADIS."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def calculer_spectre_ch4(
    ch4_ppm: float,
    pressure_bar: float,
    temperature_k: float,
    path_length_m: float,
) -> dict[str, np.ndarray]:
    """Calcule le spectre d'absorbance avec RADIS.
    
    Retourne un dictionnaire avec les nombres d'onde (cm-1),
    les longueurs d'onde (µm) et l'absorbance.
    """
    # L'importation de RADIS est vérifiée dans la fonction principale
    import radis

    print(
        f"Calcul du spectre RADIS pour {ch4_ppm} ppm de CH4, "
        f"{pressure_bar} bar, {temperature_k} K, sur {path_length_m} m..."
    )
    print(
        "Note : Le premier calcul peut etre long car RADIS doit telecharger "
        "les donnees spectroscopiques (~100 Mo pour HITRAN/CH4)."
    )
    
    # Le CH4 a des bandes d'absorption majeures autour de 3.3 µm (~3000 cm-1) 
    # et 7.6 µm (~1300 cm-1)
    # On inclut les deux principaux isotopologues du methane (12CH4 et 13CH4).
    # RADIS repartira la concentration (mole_fraction) selon leur abondance naturelle.
    spectre = radis.calc_spectrum(
        wavenum_min=500,
        wavenum_max=4500,
        molecule="CH4",
        isotope="1,2",  # Inclut 12CH4 (1) et 13CH4 (2)
        pressure=pressure_bar,
        Tgas=temperature_k,
        mole_fraction=ch4_ppm * 1e-6,
        path_length=path_length_m * 100.0,  # RADIS attend des cm
        databank="hitran",
        # Laisser RADIS gerer les avertissements est plus sur.
        # Une 'AccuracyError' peut indiquer un probleme dans les parametres.
    )

    wavenumber, absorbance = spectre.get("absorbance")
    wavelength_um = 10_000.0 / wavenumber

    return {
        "wavenumber_cm-1": wavenumber,
        "wavelength_um": wavelength_um,
        "absorbance": absorbance,
    }


def build_plot(data: dict[str, np.ndarray], use_file_backend: bool):
    if use_file_backend:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data["wavelength_um"], data["absorbance"], color="darkorange", linewidth=1)
    ax.set_xlabel("Longueur d'onde (µm)")
    ax.set_ylabel("Absorbance")
    ax.set_title("Spectre d'absorbance infrarouge du CH4")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(2.0, 10.0)  # Zoom englobant les bandes principales du CH4

    fig.tight_layout()
    return fig, ax


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule un spectre d'absorbance du CH4 avec RADIS."
    )
    parser.add_argument("--ch4-ppm", type=float, default=1.90)  # ~1900 ppb actuels
    parser.add_argument("--pressure-bar", type=float, default=1.01325)
    parser.add_argument("--temperature-k", type=float, default=288.15)
    parser.add_argument("--path-length-m", type=float, default=1000.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def is_headless_environment() -> bool:
    return bool(
        os.environ.get("CI")
        or os.environ.get("CODEX_CI")
        or os.environ.get("CODEX_SANDBOX")
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
    )


def main(argv: list[str] | None = None) -> int:
    """Fonction principale du script."""
    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Verifie la presence de RADIS avant de continuer
    try:
        import radis  # noqa: F401
    except ImportError:
        print("Erreur : la bibliotheque 'radis' n'est pas installee.", file=sys.stderr)
        print("Installez-la avec : pip install radis", file=sys.stderr)
        return 1

    if args.ch4_ppm <= 0.0:
        raise ValueError("--ch4-ppm doit etre positif.")
    if args.pressure_bar <= 0.0 or args.temperature_k <= 0.0:
        raise ValueError("Pression et temperature doivent etre positives.")

    data = calculer_spectre_ch4(
        args.ch4_ppm, args.pressure_bar, args.temperature_k, args.path_length_m
    )

    if args.csv:
        print(f"Enregistrement des donnees dans {args.csv}...")
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            np.column_stack((data["wavenumber_cm-1"], data["wavelength_um"], data["absorbance"])),
            delimiter=",",
            header="wavenumber_cm-1,wavelength_um,absorbance",
            comments="",
        )
        print("Donnees enregistrees.")

    if args.no_plot and not args.output:
        print("Calcul termine.")
        return 0

    headless = is_headless_environment()
    output_path = args.output
    if headless and output_path is None:
        output_path = SCRIPT_DIR / "spectre_absorbance_ch4.png"
        print(f"Mode non-interactif, le graphique sera enregistre dans : {output_path}")

    # La generation du graphique necessite matplotlib
    import matplotlib.pyplot as plt

    fig, _ = build_plot(data, use_file_backend=headless or args.no_plot or bool(output_path))
    
    if output_path:
        print(f"Enregistrement du graphique dans {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Graphique enregistre : {output_path}")

    if args.no_plot or headless:
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())