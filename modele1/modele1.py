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

CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
CONSTANTE_PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
CONSTANTE_BOLTZMANN = 1.380649e-23  # J K-1


# =========================
# Paramètres du modèle 1
# =========================

IRRADIANCE_SOLAIRE = 1360.0  # W m-2, symbole S_0 dans le README
ALBEDO_SURFACE = 0.30
FLUX_SOLAIRE_MOYEN_GLOBAL_ABSORBE = (
    IRRADIANCE_SOLAIRE * (1.0 - ALBEDO_SURFACE) / 4.0
)  # W m-2

TEMPERATURE_SURFACE_K = 288.15  # K, 15 °C
TEMPERATURE_ATMOSPHERE_K = 253.15  # K, -20 °C
CONCENTRATION_CO2_PPM = 425.65
NOMBRE_COUCHES = 3


@dataclass(frozen=True)
class CoucheAtmospherique:
    """Une couche verticale du modèle, uniforme en température et CO₂."""

    nom: str
    altitude_basse_km: float
    altitude_haute_km: float
    temperature_k: float
    co2_ppm: float


@dataclass(frozen=True)
class BandeSpectrale:
    """Bande CO₂ avec absorbance moyenne issue de la convention RADIS."""

    nom: str
    longueur_onde_min_um: float
    longueur_onde_max_um: float
    absorbance: float


COUCHES_ATMOSPHERE = (
    CoucheAtmospherique(
        "couche_1",
        0.0,
        5.0,
        TEMPERATURE_ATMOSPHERE_K,
        CONCENTRATION_CO2_PPM,
    ),
    CoucheAtmospherique(
        "couche_2",
        5.0,
        10.0,
        TEMPERATURE_ATMOSPHERE_K,
        CONCENTRATION_CO2_PPM,
    ),
    CoucheAtmospherique(
        "couche_3",
        10.0,
        20.0,
        TEMPERATURE_ATMOSPHERE_K,
        CONCENTRATION_CO2_PPM,
    ),
)

BANDES_CO2 = (
    BandeSpectrale("CO2_15um", 14.25, 15.75, 1.0),
    BandeSpectrale("CO2_4_3um", 4.2, 4.35, 3.25),
)


def luminance_spectrale_planck(longueur_onde_m: float, temperature_k: float) -> float: #details par bande d'absorption de sigmaT4 => loi de Planck
    """Luminance spectrale de Planck B_lambda par unité d'angle solide."""

    exposant = (
        CONSTANTE_PLANCK
        * VITESSE_LUMIERE
        / (longueur_onde_m * CONSTANTE_BOLTZMANN * temperature_k)
    )
    return (
        2.0
        * CONSTANTE_PLANCK
        * VITESSE_LUMIERE**2
        / longueur_onde_m**5
        / (exp(exposant) - 1.0)
    )# sortie fct = luminance specteale de Planck en W m-3 par angle solide, pour une longueur d'onde et une température données


def flux_bande_corps_noir(
    temperature_k: float,
    longueur_onde_min_um: float,
    longueur_onde_max_um: float,
    nombre_pas: int = 2_000,
) -> float:
    """Flux hémisphérique de corps noir intégré dans une bande spectrale."""

    longueur_onde_min_m = longueur_onde_min_um * 1e-6
    longueur_onde_max_m = longueur_onde_max_um * 1e-6
    pas_m = (longueur_onde_max_m - longueur_onde_min_m) / nombre_pas # pas d'intégration en mètre

    somme = 0.0
    for indice in range(nombre_pas):
        longueur_onde_m = longueur_onde_min_m + (indice + 0.5) * pas_m # flux hémisphérique = intégrale de la luminance spectrale de Planck sur les longueurs d'ondes de la bande, multipliée par pi pour intégrer sur les angles solides
        somme += pi * luminance_spectrale_planck(longueur_onde_m, temperature_k) * pas_m 

    return somme # retourne le flux hémisphérique de corps noir en W m-2 pour une température et une bande spectrale données (ici le flux d'une couche)(appeler plus tard layer_emession)


def transmission_depuis_absorbance(absorbance: float) -> float: # fct de transmission à partir de l'absorbance, selon la convention RADIS, prepare pr la formule d'émissivitée
    """Convention RADIS : transmission = exp(-absorbance)."""

    return exp(-absorbance)


def emissivite_depuis_absorbance(absorbance: float) -> float:
    """À l'équilibre, émissivité = absorptivité = 1 - transmission."""

    return -expm1(-absorbance) # expm1(x) = exp(x) - 1, donc 1 - exp(-absorbance) = 1 - transmission = émissivité


def propager_flux_montant(
    flux_entrant: float,
    emission_couche: float,
    transmission: float,
    emissivite: float,
    couches: tuple[CoucheAtmospherique, ...],
) -> float: #  layers est un tuple (une structure de données immuable et ordonnée en Python)(ici les 3 couches)
    """Propage un flux IR montant à travers toutes les couches."""

    flux = flux_entrant
    for _couche in couches:
        flux = transmission * flux + emissivite * emission_couche
    return flux # retourne le flux montant après avoir traversé toutes les couches


def propager_flux_descendant(
    emission_couche: float,
    transmission: float,
    emissivite: float,
    couches: tuple[CoucheAtmospherique, ...],
) -> float:
    """Propage le flux IR descendant depuis le sommet de l'atmosphère."""

    flux = 0.0
    for _couche in reversed(couches):
        flux = transmission * flux + emissivite * emission_couche
    return flux # retourne le flux descendant à la surface après avoir traversé toutes les couches

#ici calcule du bilan des flux de chaque bande avec : la prod du corps noir, l'emissivité de la bande et la transsmission 
def calculer_flux(
    couches: tuple[CoucheAtmospherique, ...] = COUCHES_ATMOSPHERE,
    bandes: tuple[BandeSpectrale, ...] = BANDES_CO2,
) -> tuple[float, float]: 
    """Retourne l'OLR au sommet et le flux IR descendant à la surface."""

    if len(couches) != NOMBRE_COUCHES:
        raise ValueError(f"Le modèle 1 attend exactement {NOMBRE_COUCHES} couches.")

    flux_total_surface = CONSTANTE_STEFAN_BOLTZMANN * TEMPERATURE_SURFACE_K**4
    flux_bandes_absorbantes_surface = 0.0
    flux_bandes_absorbantes_sommet = 0.0
    flux_descendant_surface = 0.0

    for bande in bandes:
        flux_bande_surface = flux_bande_corps_noir(
            TEMPERATURE_SURFACE_K,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux_bande_couche = flux_bande_corps_noir(
            TEMPERATURE_ATMOSPHERE_K,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )

        transmission = transmission_depuis_absorbance(bande.absorbance)
        emissivite = emissivite_depuis_absorbance(bande.absorbance)

        flux_bandes_absorbantes_surface += flux_bande_surface
        flux_bandes_absorbantes_sommet += propager_flux_montant(
            flux_bande_surface,
            flux_bande_couche,
            transmission,
            emissivite,
            couches,
        )
        flux_descendant_surface += propager_flux_descendant(
            flux_bande_couche,
            transmission,
            emissivite,
            couches,
        )

    flux_transparent_surface = flux_total_surface - flux_bandes_absorbantes_surface
    flux_sommet_atmosphere = flux_transparent_surface + flux_bandes_absorbantes_sommet

    return flux_sommet_atmosphere, flux_descendant_surface


def principal() -> None:
    flux_sommet_atmosphere, flux_descendant_surface = calculer_flux()

    print(
        "flux_infrarouge_sortant_sommet_atmosphere_W_m2 "
        f"= {flux_sommet_atmosphere:.6f}"
    )
    print(f"flux_infrarouge_descendant_surface_W_m2 = {flux_descendant_surface:.6f}")


if __name__ == "__main__":
    principal()
