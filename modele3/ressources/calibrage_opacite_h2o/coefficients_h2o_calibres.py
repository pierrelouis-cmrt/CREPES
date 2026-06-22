"""Coefficients H2O calibres par modele3.codes_python.calibrer_coefficients_h2o."""

COEFFICIENTS_H2O_CALIBRES = {
    "CO2_15um_aile_droite_externe": 11.3675437396,
    "CO2_15um_aile_droite_interne": 5.12229467116,
    "CO2_15um_aile_gauche_externe": 0.682578026891,
    "CO2_15um_aile_gauche_interne": 3.67789043263,
    "CO2_15um_coeur_sature": 2.5275765234,
    "CO2_4_3um_aile_droite": 0.0907878105524,
    "CO2_4_3um_aile_gauche": 0.000468561067285,
    "CO2_4_3um_coeur_sature": 0.0100011366944,
    "H2O_6_3um": 137.339196696,
    "H2O_fenetre_8_13um": 0.334692205631,
    "H2O_rotation_18_80um": 122.166917023,
}


def bandes_calibrees():
    from modele3.codes_python.physique import bandes_avec_coefficients_h2o

    return bandes_avec_coefficients_h2o(COEFFICIENTS_H2O_CALIBRES)
