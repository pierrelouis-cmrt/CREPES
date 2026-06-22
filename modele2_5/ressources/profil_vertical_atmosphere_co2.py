"""Profil vertical de pression, temperature et CO2 pour le modele 2.5."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DONNEES_DIR = SCRIPT_DIR.parent / "sorties"
CSV_DEFAUT = DONNEES_DIR / "profil_vertical_atmosphere_co2.csv"
GRAPHIQUE_DEFAUT = DONNEES_DIR / "profil_vertical_atmosphere_co2.png"
CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
K_B = 1.380649e-23  # J K-1
G0 = 9.80665  # m s-2
R_AIR = 287.05287  # J kg-1 K-1, air sec standard
RAYON_TERRE_USSA_M = 6_356_766.0
PRESSION_SURFACE_STANDARD_PA = 101_325.0
TEMPERATURE_SURFACE_STANDARD_K = 288.15
BASES_GEOPOTENTIELLES_M = np.array(
    [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
)
GRADIENTS_THERMIQUES_K_M = np.array(
    [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]
)

# Interface unique du profil atmospherique pour le modele 2.5 : pression,
# temperature et CO2 sont calcules dans ce module.
__all__ = [
    "altitude_depuis_pression",
    "atmosphere_standard",
    "temperature_moyenne_altitude",
    "calculer_profil",
]


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
    """Calcule temperature et pression aux bases USSA 1976."""

    temperatures = [temperature_surface_k]
    pressions = [pression_surface_pa]
    for indice, gradient in enumerate(GRADIENTS_THERMIQUES_K_M):
        epaisseur_m = BASES_GEOPOTENTIELLES_M[indice + 1] - BASES_GEOPOTENTIELLES_M[indice]
        temperature_bas = temperatures[-1]
        pression_bas = pressions[-1]
        temperature_haut = temperature_bas + gradient * epaisseur_m
        if gradient == 0.0:
            pression_haut = pression_bas * np.exp(-G0 * epaisseur_m / (R_AIR * temperature_bas))
        else:
            pression_haut = pression_bas * (temperature_haut / temperature_bas) ** (-G0 / (R_AIR * gradient))
        temperatures.append(temperature_haut)
        pressions.append(pression_haut)
    return np.asarray(temperatures), np.asarray(pressions)


def atmosphere_standard(
    altitude_geometrique_m: np.ndarray,
    pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,
    temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne temperature (K) et pression (Pa) du profil USSA 1976."""

    altitude_geometrique_m = np.asarray(altitude_geometrique_m, dtype=float)
    altitude_geopotentielle_m = geopotentiel_depuis_geometrique(altitude_geometrique_m)
    if np.any((altitude_geopotentielle_m < 0.0) | (altitude_geopotentielle_m > BASES_GEOPOTENTIELLES_M[-1])):
        raise ValueError("L'altitude doit rester dans le domaine USSA 1976.")
    temperatures_bases, pressions_bases = _calculer_bases_standard(pression_surface_pa, temperature_surface_k)
    temperature_k = np.empty_like(altitude_geopotentielle_m)
    pression_pa = np.empty_like(altitude_geopotentielle_m)
    indices_couches = np.searchsorted(BASES_GEOPOTENTIELLES_M[1:], altitude_geopotentielle_m, side="right")
    indices_couches = np.minimum(indices_couches, len(GRADIENTS_THERMIQUES_K_M) - 1)
    for indice, gradient in enumerate(GRADIENTS_THERMIQUES_K_M):
        masque = indices_couches == indice
        if not np.any(masque):
            continue
        delta_h_m = altitude_geopotentielle_m[masque] - BASES_GEOPOTENTIELLES_M[indice]
        temperature_base = temperatures_bases[indice]
        pression_base = pressions_bases[indice]
        temperature_k[masque] = temperature_base + gradient * delta_h_m
        if gradient == 0.0:
            pression_pa[masque] = pression_base * np.exp(-G0 * delta_h_m / (R_AIR * temperature_base))
        else:
            pression_pa[masque] = pression_base * (temperature_k[masque] / temperature_base) ** (-G0 / (R_AIR * gradient))
    return temperature_k, pression_pa


def altitude_depuis_pression(
    pression_pa: float,
    pression_surface_pa: float = PRESSION_SURFACE_STANDARD_PA,
    temperature_surface_k: float = TEMPERATURE_SURFACE_STANDARD_K,
) -> float:
    """Inverse le profil USSA 1976 et retourne l'altitude geometrique."""

    if pression_pa <= 0.0:
        raise ValueError("La pression doit etre strictement positive.")
    temperatures_bases, pressions_bases = _calculer_bases_standard(pression_surface_pa, temperature_surface_k)
    if pression_pa > pression_surface_pa or pression_pa < pressions_bases[-1]:
        raise ValueError("La pression doit rester dans le domaine USSA 1976.")
    for indice, gradient in enumerate(GRADIENTS_THERMIQUES_K_M):
        pression_bas, pression_haut = pressions_bases[indice : indice + 2]
        if pression_bas >= pression_pa >= pression_haut:
            temperature_base = temperatures_bases[indice]
            if gradient == 0.0:
                delta_h_m = -R_AIR * temperature_base / G0 * np.log(pression_pa / pression_bas)
            else:
                rapport_temperature = (pression_pa / pression_bas) ** (-R_AIR * gradient / G0)
                delta_h_m = temperature_base * (rapport_temperature - 1.0) / gradient
            altitude_h_m = BASES_GEOPOTENTIELLES_M[indice] + delta_h_m
            return float(geometrique_depuis_geopotentiel(altitude_h_m))
    raise RuntimeError("Pression non associee a une couche standard.")


def temperature_moyenne_altitude(altitude_bas_m: float, altitude_haut_m: float, nombre_points: int = 1_001) -> float:
    """Calcule la moyenne numerique de temperature entre deux altitudes."""

    if altitude_haut_m < altitude_bas_m:
        raise ValueError("L'altitude haute doit etre superieure a l'altitude basse.")
    if altitude_haut_m == altitude_bas_m:
        return float(atmosphere_standard(np.array([altitude_bas_m]))[0][0])
    altitudes_m = np.linspace(altitude_bas_m, altitude_haut_m, nombre_points)
    temperatures_k, _ = atmosphere_standard(altitudes_m)
    return float(np.trapezoid(temperatures_k, altitudes_m) / (altitude_haut_m - altitude_bas_m))


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
    # Le CO2 suit ici un profil simple : valeur de surface plus gradient lineaire.
    co2_ppm = co2_surface_ppm + gradient_ppm_par_km * altitude_m / 1000.0
    if np.any(co2_ppm <= 0.0):
        raise ValueError("Le profil de CO2 devient nul ou negatif.")

    fraction_molaire_co2 = co2_ppm * 1e-6
    # La concentration en molecules vient de p = n kB T sur la pression partielle.
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
    parser.add_argument(
        "--output",
        type=Path,
        default=GRAPHIQUE_DEFAUT,
        help="chemin du graphique produit",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=CSV_DEFAUT,
        help="chemin du CSV produit",
    )
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
