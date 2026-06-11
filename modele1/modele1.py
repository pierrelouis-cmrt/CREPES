"""Modele 1 CO2 : colonne atmospherique simple a 3 couches.

Le script est autonome et ne depend pas du modele 0. Il calcule deux flux
radiatifs infrarouges :
- le flux montant sortant au sommet de l'atmosphere ;
- le flux descendant recu par la surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, pi


# =========================
# Constantes physiques
# =========================

STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
PLANCK = 6.62607015e-34  # J s
LIGHT_SPEED = 299_792_458.0  # m s-1
BOLTZMANN = 1.380649e-23  # J K-1


# =========================
# Parametres du modele 1
# =========================

S0 = 1360.0  # W m-2
ALBEDO = 0.30
SOLAR_ABSORBED_GLOBAL_MEAN = S0 * (1.0 - ALBEDO) / 4.0  # W m-2

T_SURFACE = 288.15  # K, 15 degC
T_ATMOSPHERE = 253.15  # K, -20 degC
CO2_PPM = 420.0
N_LAYERS = 3


@dataclass(frozen=True)
class AtmosphericLayer:
    """Une couche verticale du modele, uniforme en temperature et CO2."""

    name: str
    altitude_bottom_km: float
    altitude_top_km: float
    temperature_k: float
    co2_ppm: float


@dataclass(frozen=True)
class SpectralBand:
    """Bande CO2 avec absorbance moyenne issue de la convention RADIS."""

    name: str
    wavelength_min_um: float
    wavelength_max_um: float
    absorbance: float


ATMOSPHERE_LAYERS = (
    AtmosphericLayer("couche_1", 0.0, 5.0, T_ATMOSPHERE, CO2_PPM),
    AtmosphericLayer("couche_2", 5.0, 10.0, T_ATMOSPHERE, CO2_PPM),
    AtmosphericLayer("couche_3", 10.0, 20.0, T_ATMOSPHERE, CO2_PPM),
)

CO2_BANDS = (
    SpectralBand("CO2_15um", 14.25, 15.75, 1.0),
    SpectralBand("CO2_4_3um", 4.2, 4.35, 3.25),
)


def planck_spectral_radiance(wavelength_m: float, temperature_k: float) -> float:
    """Luminance spectrale de Planck B_lambda en W m-3 sr-1."""

    exponent = PLANCK * LIGHT_SPEED / (wavelength_m * BOLTZMANN * temperature_k)
    return (
        2.0
        * PLANCK
        * LIGHT_SPEED**2
        / wavelength_m**5
        / (exp(exponent) - 1.0)
    )


def blackbody_band_flux(
    temperature_k: float,
    wavelength_min_um: float,
    wavelength_max_um: float,
    steps: int = 2_000,
) -> float:
    """Flux hemispherique de corps noir integre dans une bande spectrale."""

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
    """A l'equilibre, emissivite = absorptivite = 1 - transmission."""

    return -expm1(-absorbance)


def propagate_upward_flux(
    incoming_flux: float,
    layer_emission: float,
    transmission: float,
    emissivity: float,
    layers: tuple[AtmosphericLayer, ...],
) -> float:
    """Propage un flux IR montant a travers toutes les couches."""

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
    """Propage le flux IR descendant depuis le sommet de l'atmosphere."""

    flux = 0.0
    for _layer in reversed(layers):
        flux = transmission * flux + emissivity * layer_emission
    return flux


def calculate_fluxes(
    layers: tuple[AtmosphericLayer, ...] = ATMOSPHERE_LAYERS,
    bands: tuple[SpectralBand, ...] = CO2_BANDS,
) -> tuple[float, float]:
    """Retourne OLR au sommet et flux IR descendant a la surface."""

    if len(layers) != N_LAYERS:
        raise ValueError(f"Le modele 1 attend exactement {N_LAYERS} couches.")

    surface_total_flux = STEFAN_BOLTZMANN * T_SURFACE**4
    surface_absorbing_band_flux = 0.0
    top_absorbing_band_flux = 0.0
    down_surface_flux = 0.0

    for band in bands:
        surface_band_flux = blackbody_band_flux(
            T_SURFACE,
            band.wavelength_min_um,
            band.wavelength_max_um,
        )
        layer_band_flux = blackbody_band_flux(
            T_ATMOSPHERE,
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
        down_surface_flux += propagate_downward_flux(
            layer_band_flux,
            transmission,
            emissivity,
            layers,
        )

    transparent_surface_flux = surface_total_flux - surface_absorbing_band_flux
    top_atmosphere_flux = transparent_surface_flux + top_absorbing_band_flux

    return top_atmosphere_flux, down_surface_flux


def main() -> None:
    top_atmosphere_flux, down_surface_flux = calculate_fluxes()

    print(f"flux_ascendant_haut_atmosphere_W_m2 = {top_atmosphere_flux:.6f}")
    print(f"flux_descendant_surface_W_m2 = {down_surface_flux:.6f}")


if __name__ == "__main__":
    main()
