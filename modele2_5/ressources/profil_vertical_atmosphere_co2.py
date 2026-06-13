"""Profil vertical de pression, temperature et CO2 pour le modele 2.5."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from profil_temperature_standard import atmosphere_standard


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
K_B = 1.380649e-23  # J K-1


def calculer_profil(
    altitude_m: np.ndarray,
    co2_surface_ppm: float,
    gradient_ppm_par_km: float,
    pression_surface_pa: float,
    temperature_surface_k: float,
) -> dict[str, np.ndarray]:
    """Construit le profil vertical utilise par le modele 2.5."""

    temperature_k, pression_pa = atmosphere_standard(
        altitude_m,
        pression_surface_pa,
        temperature_surface_k,
    )
    co2_ppm = co2_surface_ppm + gradient_ppm_par_km * altitude_m / 1000.0
    if np.any(co2_ppm <= 0.0):
        raise ValueError("Le profil de CO2 devient nul ou negatif.")

    fraction_molaire_co2 = co2_ppm * 1e-6
    return {
        "altitude_km": altitude_m / 1000.0,
        "temperature_k": temperature_k,
        "pression_pa": pression_pa,
        "pression_hpa": pression_pa / 100.0,
        "co2_ppm": co2_ppm,
        "pression_partielle_co2_pa": pression_pa * fraction_molaire_co2,
        "concentration_co2_molecules_m3": (
            pression_pa * fraction_molaire_co2 / (K_B * temperature_k)
        ),
    }


def construire_graphique(profil: dict[str, np.ndarray], sortie_fichier: bool):
    """Construit le graphique de diagnostic du profil vertical CO2."""

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

    if sortie_fichier:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    altitude_km = profil["altitude_km"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 6), sharey=True)
    axes[0].semilogx(profil["pression_hpa"], altitude_km, color="navy")
    axes[0].set_xlabel("Pression (hPa)")
    axes[0].set_ylabel("Altitude (km)")
    axes[1].plot(profil["temperature_k"], altitude_km, color="firebrick")
    axes[1].set_xlabel("Temperature (K)")
    axes[2].plot(profil["co2_ppm"], altitude_km, color="darkgreen")
    axes[2].set_xlabel("Rapport de melange CO2 (ppm)")
    axes[3].semilogx(
        profil["concentration_co2_molecules_m3"],
        altitude_km,
        color="purple",
    )
    axes[3].set_xlabel("Concentration CO2 (molecules/m3)")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("Profil vertical pression-temperature-CO2")
    fig.tight_layout()
    return fig, plt


def analyser_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule le profil vertical de pression, temperature et CO2."
    )
    parser.add_argument(
        "--max-altitude-km",
        type=float,
        default=84.0,
        help="altitude maximale du profil en kilometres",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=100.0,
        help="pas vertical du profil en metres",
    )
    parser.add_argument(
        "--surface-co2-ppm",
        type=float,
        default=420.0,
        help="concentration de CO2 a la surface en ppm",
    )
    parser.add_argument(
        "--co2-gradient-ppm-per-km",
        type=float,
        default=0.0,
        help="gradient vertical lineaire du CO2 en ppm/km",
    )
    parser.add_argument(
        "--surface-pressure-pa",
        type=float,
        default=101_325.0,
        help="pression de surface en pascals",
    )
    parser.add_argument(
        "--surface-temperature-k",
        type=float,
        default=288.15,
        help="temperature de surface en kelvins",
    )
    parser.add_argument("--output", type=Path, help="chemin du graphique produit")
    parser.add_argument("--csv", type=Path, help="chemin du CSV produit")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="calcule sans ouvrir de fenetre graphique",
    )
    return parser.parse_args(argv)


def environnement_sans_interface_graphique() -> bool:
    """Detecte un environnement ou il faut enregistrer l'image."""

    return bool(
        os.environ.get("CI")
        or os.environ.get("CODEX_CI")
        or os.environ.get("CODEX_SANDBOX")
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
    )


def main(argv: list[str] | None = None) -> int:
    args = analyser_arguments(sys.argv[1:] if argv is None else argv)
    if not 0.0 < args.max_altitude_km <= 85.0:
        raise ValueError("--max-altitude-km doit etre entre 0 et 85.")
    if args.step_m <= 0.0:
        raise ValueError("--step-m doit etre strictement positif.")
    if args.surface_co2_ppm <= 0.0:
        raise ValueError("--surface-co2-ppm doit etre strictement positif.")
    if args.surface_pressure_pa <= 0.0 or args.surface_temperature_k <= 0.0:
        raise ValueError(
            "La pression et la temperature de surface doivent etre positives."
        )

    max_altitude_m = args.max_altitude_km * 1000.0
    altitude_m = np.arange(0.0, max_altitude_m + args.step_m, args.step_m)
    altitude_m = altitude_m[altitude_m <= max_altitude_m]
    profil = calculer_profil(
        altitude_m,
        args.surface_co2_ppm,
        args.co2_gradient_ppm_per_km,
        args.surface_pressure_pa,
        args.surface_temperature_k,
    )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            np.column_stack(tuple(profil.values())),
            delimiter=",",
            header=",".join(profil.keys()),
            comments="",
        )

    if args.no_plot and not args.output:
        print("Calcul termine.")
        if args.csv:
            print(f"Donnees enregistrees : {args.csv}")
        return 0

    sans_interface = environnement_sans_interface_graphique()
    chemin_sortie = args.output
    if sans_interface and chemin_sortie is None:
        chemin_sortie = SCRIPT_DIR / "profil_vertical_atmosphere_co2.png"

    fig, plt = construire_graphique(
        profil,
        sortie_fichier=sans_interface or args.no_plot,
    )
    if chemin_sortie:
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(chemin_sortie, dpi=200)

    if args.no_plot or sans_interface:
        plt.close(fig)
        if chemin_sortie:
            print(f"Graphique enregistre : {chemin_sortie}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
