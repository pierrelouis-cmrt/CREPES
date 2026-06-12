"""Profil vertical simplifie de la pression atmospherique et du CO2."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
G0 = 9.80665
R_AIR = 287.05287
K_B = 1.380649e-23

# Atmosphere standard 1976, jusqu'a 84.852 km geopotentiels.
LAYER_BASES_M = np.array(
    [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0]
)
LAPSE_RATES_K_M = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020])


def _standard_layer_bases(
    sea_level_pressure_pa: float,
    sea_level_temperature_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    temperatures = [sea_level_temperature_k]
    pressures = [sea_level_pressure_pa]

    for index, lapse_rate in enumerate(LAPSE_RATES_K_M):
        h0 = LAYER_BASES_M[index]
        h1 = LAYER_BASES_M[index + 1]
        t0 = temperatures[-1]
        p0 = pressures[-1]
        t1 = t0 + lapse_rate * (h1 - h0)

        if lapse_rate == 0.0:
            p1 = p0 * np.exp(-G0 * (h1 - h0) / (R_AIR * t0))
        else:
            p1 = p0 * (t1 / t0) ** (-G0 / (R_AIR * lapse_rate))

        temperatures.append(t1)
        pressures.append(p1)

    return np.asarray(temperatures), np.asarray(pressures)


def standard_atmosphere(
    altitude_m: np.ndarray,
    sea_level_pressure_pa: float = 101_325.0,
    sea_level_temperature_k: float = 288.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Retourne temperature (K) et pression (Pa) de l'atmosphere standard."""
    altitude_m = np.asarray(altitude_m, dtype=float)
    if np.any((altitude_m < 0.0) | (altitude_m > LAYER_BASES_M[-1])):
        raise ValueError("L'altitude doit rester entre 0 et 84.852 km.")

    base_temperatures, base_pressures = _standard_layer_bases(
        sea_level_pressure_pa, sea_level_temperature_k
    )
    temperature_k = np.empty_like(altitude_m)
    pressure_pa = np.empty_like(altitude_m)
    layer_indices = np.searchsorted(LAYER_BASES_M[1:], altitude_m, side="right")
    layer_indices = np.minimum(layer_indices, len(LAPSE_RATES_K_M) - 1)

    for layer_index, lapse_rate in enumerate(LAPSE_RATES_K_M):
        mask = layer_indices == layer_index
        if not np.any(mask):
            continue

        h0 = LAYER_BASES_M[layer_index]
        t0 = base_temperatures[layer_index]
        p0 = base_pressures[layer_index]
        delta_h = altitude_m[mask] - h0
        temperature_k[mask] = t0 + lapse_rate * delta_h

        if lapse_rate == 0.0:
            pressure_pa[mask] = p0 * np.exp(-G0 * delta_h / (R_AIR * t0))
        else:
            pressure_pa[mask] = p0 * (
                temperature_k[mask] / t0
            ) ** (-G0 / (R_AIR * lapse_rate))

    return temperature_k, pressure_pa


def calculate_profile(
    altitude_m: np.ndarray,
    surface_co2_ppm: float,
    gradient_ppm_per_km: float,
    sea_level_pressure_pa: float,
    sea_level_temperature_k: float,
) -> dict[str, np.ndarray]:
    temperature_k, pressure_pa = standard_atmosphere(
        altitude_m, sea_level_pressure_pa, sea_level_temperature_k
    )
    co2_ppm = surface_co2_ppm + gradient_ppm_per_km * altitude_m / 1000.0
    if np.any(co2_ppm <= 0.0):
        raise ValueError("Le profil de CO2 devient nul ou negatif.")

    co2_mole_fraction = co2_ppm * 1e-6
    return {
        "altitude_km": altitude_m / 1000.0,
        "temperature_k": temperature_k,
        "pressure_pa": pressure_pa,
        "pressure_bar": pressure_pa / 100_000.0,
        "co2_ppm": co2_ppm,
        "co2_partial_pressure_pa": pressure_pa * co2_mole_fraction,
        "co2_number_density_m3": pressure_pa
        * co2_mole_fraction
        / (K_B * temperature_k),
    }


def build_plot(profile: dict[str, np.ndarray], use_file_backend: bool):
    if use_file_backend:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    altitude_km = profile["altitude_km"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    axes[0].semilogx(profile["pressure_pa"] / 100.0, altitude_km, color="navy")
    axes[0].set_xlabel("Pression atmospherique (hPa)")
    axes[0].set_ylabel("Altitude (km)")
    axes[1].plot(profile["co2_ppm"], altitude_km, color="darkgreen")
    axes[1].set_xlabel("Rapport de melange CO2 (ppm)")
    axes[2].semilogx(profile["co2_number_density_m3"], altitude_km, color="firebrick")
    axes[2].set_xlabel("Concentration CO2 (molecules/m3)")

    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)

    fig.suptitle("Evolution verticale de la pression et du CO2")
    fig.tight_layout()
    return fig, plt


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modele le profil vertical de la pression et du CO2."
    )
    parser.add_argument("--max-altitude-km", type=float, default=50.0)
    parser.add_argument("--step-m", type=float, default=100.0)
    parser.add_argument("--surface-co2-ppm", type=float, default=420.0)
    parser.add_argument("--co2-gradient-ppm-per-km", type=float, default=0.0)
    parser.add_argument("--surface-pressure-pa", type=float, default=101_325.0)
    parser.add_argument("--surface-temperature-k", type=float, default=288.15)
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
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not 0.0 < args.max_altitude_km <= 84.852:
        raise ValueError("--max-altitude-km doit etre entre 0 et 84.852.")
    if args.step_m <= 0.0:
        raise ValueError("--step-m doit etre strictement positif.")
    if args.surface_co2_ppm <= 0.0:
        raise ValueError("--surface-co2-ppm doit etre strictement positif.")
    if args.surface_pressure_pa <= 0.0 or args.surface_temperature_k <= 0.0:
        raise ValueError("La pression et la temperature de surface doivent etre positives.")

    max_altitude_m = args.max_altitude_km * 1000.0
    altitude_m = np.arange(0.0, max_altitude_m + args.step_m, args.step_m)
    altitude_m = altitude_m[altitude_m <= max_altitude_m]
    profile = calculate_profile(
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
            np.column_stack(tuple(profile.values())),
            delimiter=",",
            header=",".join(profile.keys()),
            comments="",
        )

    if args.no_plot and not args.output:
        print("Calcul termine.")
        if args.csv:
            print(f"Donnees enregistrees : {args.csv}")
        return 0

    headless = is_headless_environment()
    output_path = args.output
    if headless and output_path is None:
        output_path = SCRIPT_DIR / "profil_atmosphere_co2.png"

    fig, plt = build_plot(profile, use_file_backend=headless or args.no_plot)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    if args.no_plot or headless:
        plt.close(fig)
        if output_path:
            print(f"Graphique enregistre : {output_path}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
