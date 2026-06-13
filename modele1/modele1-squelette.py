"""Squelette bare bones du modèle 1.

Même chemin logique que ``modele1.py`` :
surface -> bandes CO2 -> transmission/emission des couches -> deux flux finaux.
Les formules sont volontairement grossières pour garder seulement l'ossature.
"""

from math import exp


CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8

TEMPERATURE_SURFACE_K = 288.15
TEMPERATURE_ATMOSPHERE_K = 253.15
NOMBRE_COUCHES = 3

COUCHES_ATMOSPHERE = (
    ("couche_1", 0.0, 5.0),
    ("couche_2", 5.0, 10.0),
    ("couche_3", 10.0, 20.0),
)

BANDES_CO2 = (
    ("CO2_15um", 14.25, 15.75, 1.0),
    ("CO2_4_3um", 4.2, 4.35, 3.25),
)


def flux_bande_corps_noir(temperature_k, longueur_onde_min_um, longueur_onde_max_um):
    """Remplace l'intégration de Planck du vrai script par une approximation."""

    largeur_bande_um = longueur_onde_max_um - longueur_onde_min_um
    flux_total = CONSTANTE_STEFAN_BOLTZMANN * temperature_k**4
    return flux_total * largeur_bande_um / 100.0


def transmission_depuis_absorbance(absorbance):
    return exp(-absorbance)


def emissivite_depuis_absorbance(absorbance):
    return 1.0 - transmission_depuis_absorbance(absorbance)


def propager_flux(flux_initial, emission_couche, transmission, emissivite, couches):
    flux = flux_initial
    for _couche in couches:
        flux = transmission * flux + emissivite * emission_couche
    return flux


def calculer_flux(couches=COUCHES_ATMOSPHERE, bandes=BANDES_CO2):
    if len(couches) != NOMBRE_COUCHES:
        raise ValueError(f"Le modèle 1 attend exactement {NOMBRE_COUCHES} couches.")

    flux_total_surface = CONSTANTE_STEFAN_BOLTZMANN * TEMPERATURE_SURFACE_K**4
    flux_bandes_absorbantes_surface = 0.0
    flux_bandes_absorbantes_sommet = 0.0
    flux_descendant_surface = 0.0

    for _nom, longueur_onde_min_um, longueur_onde_max_um, absorbance in bandes:
        flux_bande_surface = flux_bande_corps_noir(
            TEMPERATURE_SURFACE_K,
            longueur_onde_min_um,
            longueur_onde_max_um,
        )
        flux_bande_couche = flux_bande_corps_noir(
            TEMPERATURE_ATMOSPHERE_K,
            longueur_onde_min_um,
            longueur_onde_max_um,
        )

        transmission = transmission_depuis_absorbance(absorbance)
        emissivite = emissivite_depuis_absorbance(absorbance)

        flux_bandes_absorbantes_surface += flux_bande_surface
        flux_bandes_absorbantes_sommet += propager_flux(
            flux_bande_surface,
            flux_bande_couche,
            transmission,
            emissivite,
            couches,
        )
        flux_descendant_surface += propager_flux(
            0.0,
            flux_bande_couche,
            transmission,
            emissivite,
            reversed(couches),
        )

    flux_transparent_surface = flux_total_surface - flux_bandes_absorbantes_surface
    flux_sommet_atmosphere = flux_transparent_surface + flux_bandes_absorbantes_sommet

    return flux_sommet_atmosphere, flux_descendant_surface


def principal():
    flux_sommet_atmosphere, flux_descendant_surface = calculer_flux()

    print(
        "flux_infrarouge_sortant_sommet_atmosphere_W_m2 "
        f"= {flux_sommet_atmosphere:.6f}"
    )
    print(f"flux_infrarouge_descendant_surface_W_m2 = {flux_descendant_surface:.6f}")


if __name__ == "__main__":
    principal()
