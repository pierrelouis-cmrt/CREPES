"""Profil vertical simplifié de pression, température et CO2.

Ce script sert deux rôles :

- fournir au modèle 2 les pressions aux interfaces des couches ;
- produire un CSV et un graphique de diagnostic du profil vertical utilisé.

Le profil de pression et de température suit l'atmosphère standard 1976 jusqu'à
84,852 km. Le profil de CO2 est volontairement simple : une valeur de surface et
un gradient linéaire optionnel en ppm/km.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
SORTIES_DIR = MODEL_DIR / "sorties"
CSV_DEFAUT = SORTIES_DIR / "profil_vertical_atmosphere_co2.csv"
GRAPHIQUE_DEFAUT = SORTIES_DIR / "profil_vertical_atmosphere_co2.png"
CACHE_DIR = SCRIPT_DIR / ".cache"
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
G0 = 9.80665  # m s-2, acceleration de la pesanteur standard au niveau de la mer
R_AIR = 287.05287  # J kg-1 K-1, constante de gaz pour l'air sec avec R=8.314 J mol-1 K-1 et Mair=28.9647 g mol-1
K_B = 1.380649e-23 # J K-1, constante de Boltzmann


# Atmosphère standard 1976, jusqu'à 84,852 km géopotentiels.
# Ce tableau contient les altitudes (en mètres) des bases des couches atmosphériques défénies dans notre modèle.
BASES_COUCHES_M = np.array(
    [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
)

#Ce tableau contient les gradients thermiques (en kelvins par mètre, K/m) pour chaque couche atmosphérique définie par BASES_COUCHES_M.
GRADIENTS_THERMIQUES_K_M = np.array(
    [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]
)

#renvoie des tableaux numpy contenant les températures et pressions aux bases des couches, calculées à partir de la pression et de la température de surface.
def _calculer_bases_couches_standard(pression_surface_pa: float,temperature_surface_k: float) -> tuple[np.ndarray, np.ndarray]:
    """Calcule température et pression aux bases des couches standard."""

    temperatures = [temperature_surface_k]
    pressions = [pression_surface_pa]

    for index, gradient_thermique in enumerate(GRADIENTS_THERMIQUES_K_M):
        altitude_bas_m = BASES_COUCHES_M[index]
        altitude_haut_m = BASES_COUCHES_M[index + 1]
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


def atmosphere_standard(altitude_m: np.ndarray,pression_surface_pa: float = 101_325.0,temperature_surface_k: float = 288.15) -> tuple[np.ndarray, np.ndarray]:
    """Retourne température (K) et pression (Pa) de l'atmosphère standard."""

    altitude_m = np.asarray(altitude_m, dtype=float)
    if np.any((altitude_m < 0.0) | (altitude_m > BASES_COUCHES_M[-1])):
        raise ValueError("L'altitude doit rester entre 0 et 84.852 km.")

    temperatures_bases, pressions_bases = _calculer_bases_couches_standard(
        pression_surface_pa, temperature_surface_k
    )
    temperature_k = np.empty_like(altitude_m)
    pression_pa = np.empty_like(altitude_m)
    # Chaque altitude est rattachée à la couche standard qui la contient.
    indices_couches = np.searchsorted(BASES_COUCHES_M[1:], altitude_m, side="right")
    indices_couches = np.minimum(indices_couches, len(GRADIENTS_THERMIQUES_K_M) - 1)

    for indice_couche, gradient_thermique in enumerate(GRADIENTS_THERMIQUES_K_M):
        masque = indices_couches == indice_couche
        if not np.any(masque):
            continue

        altitude_base_m = BASES_COUCHES_M[indice_couche]
        temperature_base_k = temperatures_bases[indice_couche]
        pression_base_pa = pressions_bases[indice_couche]
        delta_altitude_m = altitude_m[masque] - altitude_base_m
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


def calculer_profil(
    altitude_m: np.ndarray,
    co2_surface_ppm: float,
    gradient_ppm_par_km: float,
    pression_surface_pa: float,
    temperature_surface_k: float,
) -> dict[str, np.ndarray]:
    """Construit le profil vertical complet utilisé par le modèle 2."""

    # On part du profil atmosphérique, puis on ajoute le CO2 choisi par l'utilisateur.
    temperature_k, pression_pa = atmosphere_standard(
        altitude_m, pression_surface_pa, temperature_surface_k
    )
    co2_ppm = co2_surface_ppm + gradient_ppm_par_km * altitude_m / 1000.0
    if np.any(co2_ppm <= 0.0):
        raise ValueError("Le profil de CO2 devient nul ou négatif.")

    fraction_molaire_co2 = co2_ppm * 1e-6
    # Les grandeurs dérivées servent aux diagnostics et aux graphiques.
    return {
        "altitude_km": altitude_m / 1000.0,
        "temperature_k": temperature_k,
        "pression_pa": pression_pa,
        "pression_bar": pression_pa / 100_000.0,
        "co2_ppm": co2_ppm,
        "pression_partielle_co2_pa": pression_pa * fraction_molaire_co2,
        "concentration_co2_molecules_m3": pression_pa
        * fraction_molaire_co2
        / (K_B * temperature_k),
    }


def construire_graphique(profil: dict[str, np.ndarray], sortie_fichier: bool):
    """Construit le graphique de diagnostic du profil vertical."""

    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

    if sortie_fichier:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    altitude_km = profil["altitude_km"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    axes[0].semilogx(profil["pression_pa"] / 100.0, altitude_km, color="navy")
    axes[0].set_xlabel("Pression atmosphérique (hPa)")
    axes[0].set_ylabel("Altitude (km)")
    axes[1].plot(profil["co2_ppm"], altitude_km, color="darkgreen")
    axes[1].set_xlabel("Rapport de mélange CO2 (ppm)")
    axes[2].semilogx(
        profil["concentration_co2_molecules_m3"],
        altitude_km,
        color="firebrick",
    )
    axes[2].set_xlabel("Concentration CO2 (molécules/m3)")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("Évolution verticale de la pression et du CO2")
    fig.tight_layout()
    return fig, plt


def analyser_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcule le profil vertical de pression, température et CO2."
    )
    parser.add_argument(
        "--max-altitude-km",
        type=float,
        default=50.0,
        help="altitude maximale du profil en kilomètres",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=100.0,
        help="pas vertical du profil en mètres",
    )
    parser.add_argument(
        "--surface-co2-ppm",
        type=float,
        default=420.0,
        help="concentration de CO2 à la surface en ppm",
    )
    parser.add_argument(
        "--co2-gradient-ppm-per-km",
        type=float,
        default=0.0,
        help="gradient vertical linéaire du CO2 en ppm/km",
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
        help="température de surface en kelvins",
    )
    parser.add_argument(
        "--output", type=Path, default=GRAPHIQUE_DEFAUT, help="chemin du graphique produit"
    )
    parser.add_argument(
        "--csv", type=Path, default=CSV_DEFAUT, help="chemin du CSV produit"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="calcule sans ouvrir de fenêtre graphique",
    )
    return parser.parse_args(argv)


def environnement_sans_interface_graphique() -> bool:
    """Détecte un environnement où il faut écrire l'image au lieu de l'afficher."""

    return bool(
        os.environ.get("CI")
        or os.environ.get("CODEX_CI")
        or os.environ.get("CODEX_SANDBOX")
        or os.environ.get("MPLBACKEND", "").lower() == "agg"
    )


def main(argv: list[str] | None = None) -> int:
    args = analyser_arguments(sys.argv[1:] if argv is None else argv)
    if not 0.0 < args.max_altitude_km <= 84.852:
        raise ValueError("--max-altitude-km doit être entre 0 et 84.852.")
    if args.step_m <= 0.0:
        raise ValueError("--step-m doit être strictement positif.")
    if args.surface_co2_ppm <= 0.0:
        raise ValueError("--surface-co2-ppm doit être strictement positif.")
    if args.surface_pressure_pa <= 0.0 or args.surface_temperature_k <= 0.0:
        raise ValueError(
            "La pression et la température de surface doivent être positives."
        )

    max_altitude_m = args.max_altitude_km * 1000.0
    # Génère les points du profil vertical jusqu'à l'altitude demandée.
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
        print("Calcul terminé.")
        if args.csv:
            print(f"Données enregistrées : {args.csv}")
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
            print(f"Graphique enregistré : {chemin_sortie}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
