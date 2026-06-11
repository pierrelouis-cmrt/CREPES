"""Modèle 1 CO₂ : colonne atmosphérique simple à 3 couches.

Le script est autonome et ne dépend pas du modèle 0. Il calcule deux flux
radiatifs infrarouges :
- le flux montant sortant au sommet de l'atmosphère ;
- le flux descendant reçu par la surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, pi


# =========================
# Constantes physiques
# =========================

STEFAN_BOLTZMANN_CONSTANT = 5.670374419e-8  # W m-2 K-4
PLANCK_CONSTANT = 6.62607015e-34  # J s
SPEED_OF_LIGHT = 299_792_458.0  # m s-1
BOLTZMANN_CONSTANT = 1.380649e-23  # J K-1


# =========================
# Paramètres du modèle 1
# =========================

SOLAR_IRRADIANCE = 1360.0  # W m-2, symbole S_0 dans le README
SURFACE_ALBEDO = 0.30
GLOBAL_MEAN_ABSORBED_SOLAR_FLUX = (
    SOLAR_IRRADIANCE * (1.0 - SURFACE_ALBEDO) / 4.0
)  # W m-2

SURFACE_TEMPERATURE_K = 288.15  # K, 15 °C
ATMOSPHERE_TEMPERATURE_K = 253.15  # K, -20 °C
CO2_CONCENTRATION_PPM = 425.65
LAYER_COUNT = 3


@dataclass(frozen=True)
class AtmosphericLayer:
    """Une couche verticale du modèle, uniforme en température et CO₂."""

    name: str
    altitude_bottom_km: float
    altitude_top_km: float
    temperature_k: float
    co2_ppm: float


@dataclass(frozen=True)
class SpectralBand:
    """Bande CO₂ avec absorbance moyenne issue de la convention RADIS."""

    name: str
    wavelength_min_um: float
    wavelength_max_um: float
    absorbance: float


ATMOSPHERE_LAYERS = (
    AtmosphericLayer(
        "layer_1",
        0.0,
        5.0,
        ATMOSPHERE_TEMPERATURE_K,
        CO2_CONCENTRATION_PPM,
    ),
    AtmosphericLayer(
        "layer_2",
        5.0,
        10.0,
        ATMOSPHERE_TEMPERATURE_K,
        CO2_CONCENTRATION_PPM,
    ),
    AtmosphericLayer(
        "layer_3",
        10.0,
        20.0,
        ATMOSPHERE_TEMPERATURE_K,
        CO2_CONCENTRATION_PPM,
    ),
)

CO2_BANDS = (
    SpectralBand("CO2_15um", 14.25, 15.75, 1.0),
    SpectralBand("CO2_4_3um", 4.2, 4.35, 3.25),
)


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """Luminance spectrale de Planck B_lambda en W m-3 sr-1."""

    exponent = (
        PLANCK_CONSTANT
        * SPEED_OF_LIGHT
        / (wavelength_m * BOLTZMANN_CONSTANT * temperature_k)
    )
    return (
        2.0
        * PLANCK_CONSTANT
        * SPEED_OF_LIGHT**2
        / wavelength_m**5
        / (exp(exponent) - 1.0)
    )


def blackbody_band_flux(
    temperature_k: float,
    wavelength_min_um: float,
    wavelength_max_um: float,
    steps: int = 2_000,
) -> float:
    """Flux hémisphérique de corps noir intégré dans une bande spectrale."""

    wavelength_min_m = wavelength_min_um * 1e-6
    wavelength_max_m = wavelength_max_um * 1e-6
    step_m = (wavelength_max_m - wavelength_min_m) / steps

    total = 0.0
    for index in range(steps):
        wavelength_m = wavelength_min_m + (index + 0.5) * step_m
        total += pi * planck_spectral_radiance(wavelength_m, temperature_k) * step_m

    return total


def transmission_from_absorbance(absorbance: float) -> float:
    """Convention RADIS : transmission = exp(-absorbance)."""

    return exp(-absorbance)


def emissivity_from_absorbance(absorbance: float) -> float:
    """À l'équilibre, émissivité = absorptivité = 1 - transmission."""

    return -expm1(-absorbance)


def propagate_upward_flux(
    incoming_flux: float,
    layer_emission: float,
    transmission: float,
    emissivity: float,
    layers: tuple[AtmosphericLayer, ...],
) -> float:
    """Propage un flux IR montant à travers toutes les couches."""

    flux = incoming_flux
    for _layer in layers:
        flux = transmission * flux + emissivity * layer_emission
    return flux


def propagate_downward_flux(
    layer_emission: float,
    transmission: float,
    emissivity: float,
    layers: tuple[AtmosphericLayer, ...],
) -> float:
    """Propage le flux IR descendant depuis le sommet de l'atmosphère."""

    flux = 0.0
    for _layer in reversed(layers):
        flux = transmission * flux + emissivity * layer_emission
    return flux


def calculate_fluxes(
    layers: tuple[AtmosphericLayer, ...] = ATMOSPHERE_LAYERS,
    bands: tuple[SpectralBand, ...] = CO2_BANDS,
) -> tuple[float, float]:
    """Retourne l'OLR au sommet et le flux IR descendant à la surface."""

    if len(layers) != LAYER_COUNT:
        raise ValueError(f"Le modèle 1 attend exactement {LAYER_COUNT} couches.")

    surface_total_flux = STEFAN_BOLTZMANN_CONSTANT * SURFACE_TEMPERATURE_K**4
    surface_absorbing_band_flux = 0.0
    top_absorbing_band_flux = 0.0
    surface_downward_flux = 0.0

    for band in bands:
        surface_band_flux = blackbody_band_flux(
            SURFACE_TEMPERATURE_K,
            band.wavelength_min_um,
            band.wavelength_max_um,
        )
        layer_band_flux = blackbody_band_flux(
            ATMOSPHERE_TEMPERATURE_K,
            band.wavelength_min_um,
            band.wavelength_max_um,
        )

        transmission = transmission_from_absorbance(band.absorbance)
        emissivity = emissivity_from_absorbance(band.absorbance)

        surface_absorbing_band_flux += surface_band_flux
        top_absorbing_band_flux += propagate_upward_flux(
            surface_band_flux,
            layer_band_flux,
            transmission,
            emissivity,
            layers,
        )
        surface_downward_flux += propagate_downward_flux(
            layer_band_flux,
            transmission,
            emissivity,
            layers,
        )

    transparent_surface_flux = surface_total_flux - surface_absorbing_band_flux
    top_atmosphere_flux = transparent_surface_flux + top_absorbing_band_flux

    return top_atmosphere_flux, surface_downward_flux


def main() -> None:
    top_atmosphere_flux, surface_downward_flux = calculate_fluxes()

    print(f"outgoing_longwave_flux_top_atmosphere_W_m2 = {top_atmosphere_flux:.6f}")
    print(f"downward_longwave_flux_surface_W_m2 = {surface_downward_flux:.6f}")


if __name__ == "__main__":
    main()
