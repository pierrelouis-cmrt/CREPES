"""Profil de temperature de l'atmosphere standard 1976.

Ce script produit un CSV et un graphique du profil de temperature standard
utilise par le modele 2.5. Les equations suivent l'atmosphere standard 1976 :
temperature lineaire par morceaux en altitude geopotentielle, pression obtenue
par hydrostatique et gaz parfait sec.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"

G0 = 9.80665  # m s-2
R_AIR = 287.05287  # J kg-1 K-1, air sec standard
RAYON_TERRE_USSA_M = 6_356_766.0
PRESSION_SURFACE_STANDARD_PA = 101_325.0
TEMPERATURE_SURFACE_STANDARD_K = 288.15

# U.S. Standard Atmosphere 1976, jusqu'a 84,852 km geopotentiels.
BASES_GEOPOTENTIELLES_M = np.array(
    [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
)
GRADIENTS_THERMIQUES_K_M = np.array(
    [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]
)


def geopotentiel_depuis_geometrique(altitude_geometrique_m: np.ndarray) -> np.ndarray:
    """Convertit l'altitude geometrique en altitude geopotentielle."""

    altitude_geometrique_m = np.asarray(altitude_geometrique_m, dtype=float)
    return (
        RAYON_TERRE_USSA_M
        * altitude_geometrique_m
        / (RAYON_TERRE_USSA_M + altitude_geometrique_m)
    )


def geometrique_depuis_geopotentiel(altitude_geopotentielle_m: np.ndarray) -> np.ndarray:
    """Convertit l'altitude geopotentielle en altitude geometrique."""

    altitude_geopotentielle_m = np.asarray(altitude_geopotentielle_m, dtype=float)
    return (
        RAYON_TERRE_USSA_M
        * altitude_geopotentielle_m
        / (RAYON_TERRE_USSA_M - altitude_geopotentielle_m)
    )


def _calculer_bases_standard(pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K) -> tuple[np.ndarray, np.ndarray]:
    """Calcule temperature et pression aux bases des couches standard."""

    "calculer_bases_standard(pression_surface_pa,temperature_de_la_surface_k)->tableau (liste(température),liste(pressions))"
    "float = PRESSION_SURFACE_STANDARD_PA, signifie que par défaut on prend la pression de surface standard"

    #return np.asarray(temperatures), np.asarray(pressions)


def atmosphere_standard(altitude_geometrique_m: np.ndarray,pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,) -> tuple[np.ndarray, np.ndarray]:
    """Retourne temperature (K) et pression (Pa) a une altitude geometrique."""

    "atmosphere_standard(liste(altitude_geometrique_m,pression_surface_pa,temperature_de_la_surface_k)->tableau (température,pression)"

    #return temperature_k, pression_pa


def altitude_depuis_pression(pression_pa: float,pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K) -> float:
    """Inverse le profil standard et retourne l'altitude geometrique en metres."""

    "altitude_depuis_pression(pression_pa,pression_surface_pa,temperature_de_la_surface_k)->altitude_geometrique_m"

    return float(geometrique_depuis_geopotentiel(BASES_GEOPOTENTIELLES_M[-1]))


def temperature_moyenne_altitude(altitude_bas_m: float,altitude_haut_m: float,nombre_points: int = 1_001,) -> float:
    """Moyenne numerique de T(z) entre deux altitudes geometriques."""
    "temperature_moyenne_altitude(altitude_bas_m,altitude_haut_m,nombre_points)->moyenne de la temperature entre les deux altitudes"

    #return float(np.trapezoid(temperatures_k, altitudes_m) / (altitude_haut_m - altitude_bas_m))



def construire_profil_temperature(max_altitude_km: float,step_m: float,) -> dict[str, np.ndarray]:
    """Construit le profil exporte par le script."""
    "construire_profil_temperature(max_altitude_km,step_m)->tableau (altitude_geometrique_km,altitude_geopotentielle_km,temperature_k,temperature_c,pression_pa,pression_hpa)"
    max_altitude_m = max_altitude_km * 1000.0
    altitudes_m = np.arange(0.0, max_altitude_m + step_m, step_m)
    altitudes_m = altitudes_m[altitudes_m <= max_altitude_m]
    temperatures_k, pressions_pa = atmosphere_standard(altitudes_m)
    altitude_geopotentielle_m = geopotentiel_depuis_geometrique(altitudes_m)
    return {
        "altitude_geometrique_km": altitudes_m / 1000.0,
        "altitude_geopotentielle_km": altitude_geopotentielle_m / 1000.0,
        "temperature_k": temperatures_k,
        "temperature_c": temperatures_k - 273.15,
        "pression_pa": pressions_pa,
        "pression_hpa": pressions_pa / 100.0,
    }

"visuel"
def construire_graphique_temperature(profil: dict[str, np.ndarray],sortie_fichier: bool,):
    """Construit le graphique temperature-altitude."""


    #return fig, plt

"creer les chemeins pr acceder aux fichier csv et graphique"
def analyser_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere le profil standard de temperature atmospherique.")
    parser.add_argument("--max-altitude-km",type=float,default=84.0,help="altitude geometrique maximale en kilometres")
    parser.add_argument("--step-m",type=float,default=100.0,help="pas vertical du profil en metres")
    parser.add_argument("--output", type=Path, help="chemin du graphique produit")
    parser.add_argument("--csv", type=Path, help="chemin du CSV produit")
    parser.add_argument("--no-plot",action="store_true",help="calcule sans ouvrir de fenetre graphique")
    return parser.parse_args(argv)

"visuel"
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

    profil = construire_profil_temperature(args.max_altitude_km, args.step_m)
    "permet de mettre les donnees dans un CSV, pour les utiliser dans d'autres logiciels"
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


    "visuel"
    sans_interface = environnement_sans_interface_graphique()
    chemin_sortie = args.output
    if sans_interface and chemin_sortie is None:
        chemin_sortie = SCRIPT_DIR / "profil_temperature_standard.png"

    fig, plt = construire_graphique_temperature(
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
