"""Coefficients CO2 calibres par modele3.codes_python.calibrer_coefficients_co2."""

COEFFICIENTS_CO2_CALIBRES = {
    "CO2_15um_aile_droite_externe": 0.0200755830526,
    "CO2_15um_aile_droite_interne": 0.225559137603,
    "CO2_15um_aile_gauche_externe": 0.0398669092194,
    "CO2_15um_aile_gauche_interne": 0.234672284191,
    "CO2_15um_coeur_sature": 1.03576529886,
    "CO2_4_3um_aile_droite": 0.0332982444001,
    "CO2_4_3um_aile_gauche": 0.0087188685765,
    "CO2_4_3um_coeur_sature": 0.751487516478,
}


def bandes_calibrees():
    from modele3.codes_python.physique import bandes_avec_coefficients_co2

    return bandes_avec_coefficients_co2(COEFFICIENTS_CO2_CALIBRES)
