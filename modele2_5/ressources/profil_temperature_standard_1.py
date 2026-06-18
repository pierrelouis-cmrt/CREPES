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


def _calculer_bases_standard(
    pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,
    temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule temperature et pression aux bases des couches standard."""

    temperatures = [temperature_surface_k]
    pressions = [pression_surface_pa]

    for indice, gradient_thermique in enumerate(GRADIENTS_THERMIQUES_K_M):
        altitude_bas_m = BASES_GEOPOTENTIELLES_M[indice]
        altitude_haut_m = BASES_GEOPOTENTIELLES_M[indice + 1]
        temperature_bas_k = temperatures[-1]
        pression_bas_pa = pressions[-1]
        epaisseur_m = altitude_haut_m - altitude_bas_m
        temperature_haut_k = temperature_bas_k + gradient_thermique * epaisseur_m

        if gradient_thermique == 0.0:
            pression_haut_pa = pression_bas_pa * np.exp(
                -G0 * epaisseur_m / (R_AIR * temperature_bas_k)
            )
        else:
            pression_haut_pa = pression_bas_pa * (
                temperature_haut_k / temperature_bas_k
            ) ** (-G0 / (R_AIR * gradient_thermique))

        temperatures.append(temperature_haut_k)
        pressions.append(pression_haut_pa)

    return np.asarray(temperatures), np.asarray(pressions)


def atmosphere_standard(
    altitude_geometrique_m: np.ndarray,
    pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,
    temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne temperature (K) et pression (Pa) a une altitude geometrique."""

    altitude_geometrique_m = np.asarray(altitude_geometrique_m, dtype=float)
    altitude_geopotentielle_m = geopotentiel_depuis_geometrique(altitude_geometrique_m)
    if np.any(
        (altitude_geopotentielle_m < 0.0)
        | (altitude_geopotentielle_m > BASES_GEOPOTENTIELLES_M[-1])
    ):
        raise ValueError("L'altitude doit rester entre 0 et 84,852 km geopotentiels.")

    temperatures_bases, pressions_bases = _calculer_bases_standard(
        pression_surface_pa,
        temperature_surface_k,
    )
    temperature_k = np.empty_like(altitude_geopotentielle_m)
    pression_pa = np.empty_like(altitude_geopotentielle_m)
    indices_couches = np.searchsorted(
        BASES_GEOPOTENTIELLES_M[1:],
        altitude_geopotentielle_m,
        side="right",
    )
    indices_couches = np.minimum(indices_couches, len(GRADIENTS_THERMIQUES_K_M) - 1)

    for indice_couche, gradient_thermique in enumerate(GRADIENTS_THERMIQUES_K_M):
        masque = indices_couches == indice_couche
        if not np.any(masque):
            continue

        altitude_base_m = BASES_GEOPOTENTIELLES_M[indice_couche]
        temperature_base_k = temperatures_bases[indice_couche]
        pression_base_pa = pressions_bases[indice_couche]
        delta_altitude_m = altitude_geopotentielle_m[masque] - altitude_base_m
        temperature_k[masque] = (
            temperature_base_k + gradient_thermique * delta_altitude_m
        )

        if gradient_thermique == 0.0:
            pression_pa[masque] = pression_base_pa * np.exp(
                -G0 * delta_altitude_m / (R_AIR * temperature_base_k)
            )
        else:
            pression_pa[masque] = pression_base_pa * (
                temperature_k[masque] / temperature_base_k
            ) ** (-G0 / (R_AIR * gradient_thermique))

    return temperature_k, pression_pa


def altitude_depuis_pression(pression_pa: float,pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,) -> float:
    """Inverse le profil standard et retourne l'altitude geometrique en metres."""

    if pression_pa <= 0.0:
        raise ValueError("La pression doit etre strictement positive.")

    temperatures_bases, pressions_bases = _calculer_bases_standard(
        pression_surface_pa,
        temperature_surface_k,
    )
    pression_min_pa = pressions_bases[-1]
    if pression_pa > pression_surface_pa or pression_pa < pression_min_pa:
        raise ValueError(
            "La pression doit rester dans le domaine de l'atmosphere standard."
        )

    for indice, gradient_thermique in enumerate(GRADIENTS_THERMIQUES_K_M):
        pression_bas_pa = pressions_bases[indice]
        pression_haut_pa = pressions_bases[indice + 1]
        if pression_bas_pa >= pression_pa >= pression_haut_pa:
            altitude_base_m = BASES_GEOPOTENTIELLES_M[indice]
            temperature_base_k = temperatures_bases[indice]
            if gradient_thermique == 0.0:
                delta_h_m = (
                    -R_AIR
                    * temperature_base_k
                    / G0
                    * np.log(pression_pa / pression_bas_pa)
                )
            else:
                rapport_temperature = (pression_pa / pression_bas_pa) ** (
                    -R_AIR * gradient_thermique / G0
                )
                delta_h_m = (
                    temperature_base_k * (rapport_temperature - 1.0)
                    / gradient_thermique
                )
            altitude_geopotentielle_m = altitude_base_m + delta_h_m
            return float(geometrique_depuis_geopotentiel(altitude_geopotentielle_m))

    return float(geometrique_depuis_geopotentiel(BASES_GEOPOTENTIELLES_M[-1]))


def temperature_moyenne_altitude(
    altitude_bas_m: float,
    altitude_haut_m: float,
    nombre_points: int = 1_001,
) -> float:
    """Moyenne numerique de T(z) entre deux altitudes geometriques."""

    if altitude_haut_m < altitude_bas_m:
        raise ValueError("L'altitude haute doit etre superieure a l'altitude basse.")
    if altitude_haut_m == altitude_bas_m:
        return float(atmosphere_standard(np.array([altitude_bas_m]))[0][0])

    altitudes_m = np.linspace(altitude_bas_m, altitude_haut_m, nombre_points)
    temperatures_k, _ = atmosphere_standard(altitudes_m)
    return float(np.trapezoid(temperatures_k, altitudes_m) / (altitude_haut_m - altitude_bas_m))


def construire_profil_temperature(max_altitude_km: float,step_m: float) -> dict[str, np.ndarray]:
    """Construit le profil exporte par le script."""

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


def construire_graphique_temperature(
    profil: dict[str, np.ndarray],
    sortie_fichier: bool,
):
    """Construit le graphique temperature-altitude."""

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

    if sortie_fichier:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    altitude_km = profil["altitude_geometrique_km"]
    temperature_k = profil["temperature_k"]
    fig, axis = plt.subplots(figsize=(7, 8))
    axis.plot(temperature_k, altitude_km, color="firebrick", linewidth=2.2)

    for base_geopotentielle_m in BASES_GEOPOTENTIELLES_M[1:-1]:
        base_geometrique_km = (
            geometrique_depuis_geopotentiel(np.array([base_geopotentielle_m]))[0]
            / 1000.0
        )
        if base_geometrique_km <= altitude_km[-1]:
            axis.axhline(base_geometrique_km, color="0.75", linewidth=0.8, alpha=0.7)

    axis.set_xlabel("Temperature standard (K)")
    axis.set_ylabel("Altitude geometrique (km)")
    axis.set_title("Profil standard de temperature - atmosphere 1976")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, plt


def analyser_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genere le profil standard de temperature atmospherique."
    )
    parser.add_argument(
        "--max-altitude-km",
        type=float,
        default=84.0,
        help="altitude geometrique maximale en kilometres",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=100.0,
        help="pas vertical du profil en metres",
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

    profil = construire_profil_temperature(args.max_altitude_km, args.step_m)

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
