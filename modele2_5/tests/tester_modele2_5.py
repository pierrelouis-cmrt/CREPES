"""Tests numeriques simples du modele 2.5.

Ce script reste volontairement separe du modele principal. Il est fait pour etre
lance a la main et pour afficher des valeurs comparables d'une iteration a
l'autre.
"""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from pathlib import Path
import sys


RACINE_PROJET = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(RACINE_PROJET))
except ValueError:
    pass
sys.path.insert(0, str(RACINE_PROJET))

from modele2_5.codes_python.modele2_5 import (
    BANDES_CO2,
    CONSTANTE_STEFAN_BOLTZMANN,
    TEMPERATURE_SURFACE_K,
    calculer_flux_colonne,
    calculer_forcage_doublement_co2,
    construire_bandes_co2,
    creer_couches_atmospheriques,
)


def assert_presque_egal(
    valeur: float,
    attendu: float,
    tolerance_absolue: float,
    message: str,
) -> None:
    """Assertion lisible pour les grandeurs physiques."""

    if not isclose(valeur, attendu, abs_tol=tolerance_absolue):
        raise AssertionError(
            f"{message}: valeur={valeur:.9f}, attendu={attendu:.9f}, "
            f"tolerance={tolerance_absolue:.9f}"
        )


def test_opacite_nulle() -> None:
    """Sans opacite, OLR = sigma Ts^4 et LW descendant = 0."""

    couches = creer_couches_atmospheriques(co2_surface_ppm=420.0)
    bandes_transparentes = tuple(
        replace(bande, coefficient_opacite=0.0) for bande in BANDES_CO2
    )
    olr, flux_descendant = calculer_flux_colonne(couches, bandes_transparentes)
    flux_surface = CONSTANTE_STEFAN_BOLTZMANN * TEMPERATURE_SURFACE_K**4
    assert_presque_egal(olr, flux_surface, 1e-8, "OLR transparent")
    assert_presque_egal(flux_descendant, 0.0, 1e-10, "LW descendant transparent")


def test_grille_pression_temperature() -> None:
    """La grille doit avoir 10 couches, pression descendante et altitude montante."""

    couches = creer_couches_atmospheriques()
    if len(couches) != 10:
        raise AssertionError(f"Nombre de couches inattendu: {len(couches)}")
    for couche in couches:
        if couche.pression_bas_pa <= couche.pression_haut_pa:
            raise AssertionError(f"Pressions non descendantes: {couche.nom}")
        if couche.altitude_bas_km >= couche.altitude_haut_km:
            raise AssertionError(f"Altitudes non montantes: {couche.nom}")
        if not 180.0 <= couche.temperature_k <= 300.0:
            raise AssertionError(
                f"Temperature moyenne hors domaine plausible: {couche.nom}"
            )


def test_monotonie_co2() -> None:
    """A temperatures fixees, augmenter le CO2 doit diminuer l'OLR."""

    olr_280, _ = calculer_flux_colonne(creer_couches_atmospheriques(280.0))
    olr_420, _ = calculer_flux_colonne(creer_couches_atmospheriques(420.0))
    olr_560, _ = calculer_flux_colonne(creer_couches_atmospheriques(560.0))
    if not (olr_280 > olr_420 > olr_560):
        raise AssertionError(
            "OLR non monotone avec le CO2: "
            f"280={olr_280:.6f}, 420={olr_420:.6f}, 560={olr_560:.6f}"
        )


def test_forcage_doublement() -> None:
    """Le doublement 280->560 ppm doit rester proche de la cible IPCC AR6."""

    forcage = calculer_forcage_doublement_co2()
    if not 3.70 <= forcage <= 4.10:
        raise AssertionError(f"Forcage 2xCO2 hors cible: {forcage:.6f} W/m2")


def test_comportement_logarithmique() -> None:
    """Deux doublements successifs doivent donner des forçages proches."""

    bandes = construire_bandes_co2()
    forcage_280_560 = calculer_forcage_doublement_co2(280.0, 560.0, bandes)
    forcage_560_1120 = calculer_forcage_doublement_co2(560.0, 1120.0, bandes)
    ecart_relatif = abs(forcage_560_1120 - forcage_280_560) / forcage_280_560
    if ecart_relatif > 0.35:
        raise AssertionError(
            "Comportement trop eloigne d'une loi logarithmique: "
            f"F280-560={forcage_280_560:.6f}, "
            f"F560-1120={forcage_560_1120:.6f}"
        )


def afficher_cas_reference() -> None:
    """Affiche quelques sorties numeriques pour comparaison future."""

    print("cas_reference")
    for co2_ppm in (280.0, 420.0, 560.0, 1120.0):
        olr, flux_descendant = calculer_flux_colonne(
            creer_couches_atmospheriques(co2_surface_ppm=co2_ppm)
        )
        print(
            f"CO2={co2_ppm:.0f} ppm, "
            f"OLR={olr:.6f} W/m2, "
            f"LW_down_surface={flux_descendant:.6f} W/m2"
        )
    print(
        "forcage_280_560_ppm="
        f"{calculer_forcage_doublement_co2():.6f} W/m2"
    )


def main() -> None:
    tests = (
        test_opacite_nulle,
        test_grille_pression_temperature,
        test_monotonie_co2,
        test_forcage_doublement,
        test_comportement_logarithmique,
    )
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    afficher_cas_reference()


if __name__ == "__main__":
    main()
