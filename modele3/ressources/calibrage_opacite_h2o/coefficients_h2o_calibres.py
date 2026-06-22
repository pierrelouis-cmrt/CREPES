"""Coefficients H2O calibres par modele3.codes_python.calibrer_coefficients_h2o."""

COEFFICIENTS_H2O_CALIBRES = {
    "CO2_15um_aile_droite_externe": 1.4496770952,
    "CO2_15um_aile_droite_interne": 0.6532346327,
    "CO2_15um_aile_gauche_externe": 0.0870476291,
    "CO2_15um_aile_gauche_interne": 0.4690330331,
    "CO2_15um_coeur_sature": 0.3223361068,
    "CO2_4_3um_aile_droite": 0.0115779638,
    "CO2_4_3um_aile_gauche": 0.0000597545,
    "CO2_4_3um_coeur_sature": 0.0012754223,
    "H2O_6_3um": 17.5145565554,
    "H2O_fenetre_8_13um": 0.0426825386,
    "H2O_rotation_18_80um": 15.5796701079,
}


def bandes_calibrees():
    from modele3.codes_python.physique import bandes_avec_coefficients_h2o

    return bandes_avec_coefficients_h2o(COEFFICIENTS_H2O_CALIBRES)
