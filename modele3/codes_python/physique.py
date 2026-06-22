"""Formules physiques elementaires du modele 3.

Ce module ne lit aucun fichier. Il contient les constantes, la geometrie
solaire, Planck, les masses colonne et les opacites infrarouges CO2 + H2O.
Les nuages ne creent pas d'opacite longwave implicite.
"""

from __future__ import annotations

from functools import lru_cache
from math import cos, exp, isfinite, pi, radians, sin


SIGMA = 5.670374419e-8  # W m-2 K-4
PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
BOLTZMANN = 1.380649e-23  # J K-1
GRAVITE = 9.80665  # m s-2

CONSTANTE_SOLAIRE = 1361.0  # W m-2
TEMPERATURE_SURFACE_DEFAUT_K = 288.15
EMISSIVITE_SURFACE_CONSTANTE = 0.98
CO2_REFERENCE_PPM = 280.0
CO2_DEFAUT_PPM = 420.0
PRESSION_REFERENCE_PA = 101_325.0

FACTEUR_DIFFUSIF = 1.66
ECHELLE_OPACITE_CO2 = 0.0327228010
MASSE_H2O_REFERENCE_KG_M2 = 10.0
ALBEDO_SURFACE_SECOURS = 0.30
ALBEDO_NEIGE_GLACE_SECOURS = 0.65
SEUIL_FRACTION_NEIGE_GLACE_ALBEDO = 0.05

COEFFICIENTS_OPACITE_EFFECTIFS = {
    "statut": "coefficients effectifs pedagogiques, pas des sections efficaces spectrales",
    "origine": (
        "noyau long-onde du modele 2.5 pour CO2; recalibrage possible avec "
        "modele3/codes_python/calibrer_coefficients_co2.py; ajout modele 3 pour H2O a "
        "partir des masses colonne ERA5"
    ),
    "cible_co2": "ordre de grandeur du forcage relatif 280 -> 560 ppm conserve du modele 2.5",
    "unite_a_co2": (
        "profondeur optique effective sans dimension pour CO2=280 ppm "
        f"et delta_p={PRESSION_REFERENCE_PA:g} Pa"
    ),
    "unite_a_h2o": (
        "profondeur optique effective sans dimension pour "
        f"{MASSE_H2O_REFERENCE_KG_M2:g} kg m-2 de vapeur d'eau"
    ),
    "limites": (
        "valable pour comparer des colonnes pedagogiques CO2 + H2O; ne remplace pas "
        "HITRAN, correlated-k, ni une dependance fine en temperature/pression"
    ),
}

PRESSION_BORDS_REFERENCE_HPA = [
    850.0,
    700.0,
    500.0,
    300.0,
    200.0,
    100.0,
    50.0,
    20.0,
    10.0,
    1.0,
]
JOURS_CUMULES_MOIS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
JOURS_MILIEU_MOIS = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]


def creer_bande(nom, lambda_min_um, lambda_max_um, a_co2, a_h2o, famille, role):
    return {
        "nom": nom,
        "lambda_min_um": lambda_min_um,
        "lambda_max_um": lambda_max_um,
        "a_co2": a_co2,
        "a_h2o": a_h2o,
        "famille": famille,
        "role": role,
    }


BANDES_INFRAROUGES = [
    creer_bande(
        "CO2_15um_aile_gauche_externe",
        13.00,
        14.00,
        0.32 * ECHELLE_OPACITE_CO2,
        0.64,
        "CO2",
        "aile",
    ),
    creer_bande(
        "CO2_15um_aile_gauche_interne",
        14.00,
        14.60,
        3.50 * ECHELLE_OPACITE_CO2,
        0.40,
        "CO2",
        "aile",
    ),
    creer_bande(
        "CO2_15um_coeur_sature",
        14.60,
        15.40,
        40.0 * ECHELLE_OPACITE_CO2,
        0.16,
        "CO2",
        "coeur sature",
    ),
    creer_bande(
        "CO2_15um_aile_droite_interne",
        15.40,
        16.20,
        4.00 * ECHELLE_OPACITE_CO2,
        0.48,
        "CO2",
        "aile",
    ),
    creer_bande(
        "CO2_15um_aile_droite_externe",
        16.20,
        18.00,
        0.48 * ECHELLE_OPACITE_CO2,
        2.00,
        "CO2",
        "aile",
    ),
    creer_bande(
        "CO2_4_3um_aile_gauche",
        4.00,
        4.20,
        0.20 * ECHELLE_OPACITE_CO2,
        0.16,
        "CO2",
        "aile",
    ),
    creer_bande(
        "CO2_4_3um_coeur_sature",
        4.20,
        4.40,
        15.0 * ECHELLE_OPACITE_CO2,
        0.08,
        "CO2",
        "coeur sature",
    ),
    creer_bande(
        "CO2_4_3um_aile_droite",
        4.40,
        4.60,
        0.20 * ECHELLE_OPACITE_CO2,
        0.16,
        "CO2",
        "aile",
    ),
    creer_bande("H2O_6_3um", 5.50, 7.50, 0.0, 25.60, "H2O", "bande vibration-rotation"),
    creer_bande("H2O_fenetre_8_13um", 8.00, 13.00, 0.0, 0.48, "H2O", "continuum fenetre"),
    creer_bande("H2O_rotation_18_80um", 18.00, 80.00, 0.0, 14.40, "H2O", "rotation loin IR"),
]


def copier_bandes_infrarouges():
    """Retourne une copie mutable des bandes de production."""

    return [dict(bande) for bande in BANDES_INFRAROUGES]


def bandes_co2():
    """Retourne les bandes qui portent une opacite CO2 effective."""

    return [
        dict(bande)
        for bande in BANDES_INFRAROUGES
        if float(bande.get("a_co2", 0.0)) > 0.0
    ]


def bandes_h2o():
    """Retourne les bandes qui portent une opacite H2O effective."""

    return [
        dict(bande)
        for bande in BANDES_INFRAROUGES
        if float(bande.get("a_h2o", 0.0)) > 0.0
    ]


def bandes_avec_coefficients_co2(coefficients_co2, facteur=1.0, zero_h2o=False):
    """Construit les bandes du modele avec des coefficients CO2 remplaces.

    `coefficients_co2` est un dictionnaire `{nom_bande: a_co2}`. Les bandes non
    presentes gardent leur valeur de production. La fonction ne mute jamais
    `BANDES_INFRAROUGES`, ce qui permet au script de calibrage de tester des
    jeux de coefficients sans changer le runtime par effet de bord.
    """

    coefficients = {nom: float(valeur) for nom, valeur in dict(coefficients_co2).items()}
    bandes = copier_bandes_infrarouges()
    for bande in bandes:
        if bande["nom"] in coefficients:
            bande["a_co2"] = max(0.0, coefficients[bande["nom"]] * float(facteur))
        if zero_h2o:
            bande["a_h2o"] = 0.0
    return bandes


def bandes_avec_coefficients_h2o(coefficients_h2o, facteur=1.0, zero_co2=False):
    """Construit les bandes du modele avec des coefficients H2O remplaces.

    `coefficients_h2o` est un dictionnaire `{nom_bande: a_h2o}`. Les bandes non
    presentes gardent leur valeur de production. La fonction suit le meme
    contrat que `bandes_avec_coefficients_co2` pour faciliter les calibrages.
    """

    coefficients = {nom: float(valeur) for nom, valeur in dict(coefficients_h2o).items()}
    bandes = copier_bandes_infrarouges()
    for bande in bandes:
        if bande["nom"] in coefficients:
            bande["a_h2o"] = max(0.0, coefficients[bande["nom"]] * float(facteur))
        if zero_co2:
            bande["a_co2"] = 0.0
    return bandes


def borner(valeur, minimum, maximum):
    return max(minimum, min(maximum, valeur))


def valeur_finie(valeur, defaut=None):
    if valeur is None:
        return defaut
    try:
        valeur = float(valeur)
    except (TypeError, ValueError):
        return defaut
    if not isfinite(valeur):
        return defaut
    return valeur


def fraction(valeur, defaut=0.0, maximum=1.0):
    valeur = valeur_finie(valeur, defaut)
    if valeur is None:
        valeur = defaut
    return borner(valeur, 0.0, maximum)


def albedo_surface_corrige_neige_glace(albedo_surface, snow_ice_fraction=None):
    """Corrige le cas source non observable: albedo nul sur neige/glace.

    Les CSV historiques viennent d'un rapport SW_UP/SW_DOWN. En nuit polaire,
    le rapport peut produire 0 alors que l'albedo physique d'une surface
    neigeuse ou glacee reste eleve. La correction reste limitee aux mailles ou
    la fraction neige/glace est explicitement non nulle.
    """

    albedo = fraction(albedo_surface, defaut=ALBEDO_SURFACE_SECOURS)
    neige_glace = fraction(snow_ice_fraction, defaut=0.0)
    if albedo > 0.0 or neige_glace <= SEUIL_FRACTION_NEIGE_GLACE_ALBEDO:
        return albedo
    return ALBEDO_SURFACE_SECOURS + neige_glace * (
        ALBEDO_NEIGE_GLACE_SECOURS - ALBEDO_SURFACE_SECOURS
    )


def mois_depuis_jour_annee(jour_annee):
    if not 1 <= jour_annee <= 365:
        raise ValueError("jour_annee doit etre entre 1 et 365.")
    mois = 1
    for seuil in JOURS_CUMULES_MOIS[1:]:
        if jour_annee > seuil:
            mois += 1
    return min(mois, 12)


def jour_milieu_mois(mois):
    if not 1 <= mois <= 12:
        raise ValueError("mois doit etre entre 1 et 12.")
    return JOURS_MILIEU_MOIS[mois - 1]


def poids_interpolation_mensuelle(jour_annee):
    """Retourne deux mois voisins et le poids du second mois.

    Les mois sont donnes en indices 0..11. L'interpolation est cyclique autour
    des milieux de mois pour eviter une rupture au 1er janvier.
    """

    if not 1 <= jour_annee <= 365:
        raise ValueError("jour_annee doit etre entre 1 et 365.")

    jours = JOURS_MILIEU_MOIS
    if jour_annee < jours[0]:
        jour_precedent = jours[-1] - 365
        poids = (jour_annee - jour_precedent) / (jours[0] - jour_precedent)
        return 11, 0, poids
    for indice in range(11):
        if jours[indice] <= jour_annee <= jours[indice + 1]:
            poids = (jour_annee - jours[indice]) / (jours[indice + 1] - jours[indice])
            return indice, indice + 1, poids
    jour_suivant = jours[0] + 365
    poids = (jour_annee - jours[-1]) / (jour_suivant - jours[-1])
    return 11, 0, poids


def declinaison_solaire_rad(jour_annee):
    return radians(23.44) * sin(2.0 * pi * (284 + jour_annee) / 365.0)


def cosinus_incidence_solaire(latitude_deg, jour_annee, heure_solaire):
    latitude_rad = radians(latitude_deg)
    declinaison = declinaison_solaire_rad(jour_annee)
    angle_horaire = radians(15.0 * (heure_solaire - 12.0))
    cosinus = (
        sin(latitude_rad) * sin(declinaison)
        + cos(latitude_rad) * cos(declinaison) * cos(angle_horaire)
    )
    return max(cosinus, 0.0)


def flux_solaire_incident(latitude_deg, jour_annee, heure_solaire):
    return CONSTANTE_SOLAIRE * cosinus_incidence_solaire(
        latitude_deg,
        jour_annee,
        heure_solaire,
    )


def flux_solaire_moyen_journalier(latitude_deg, jour_annee):
    total = 0.0
    nombre_pas = 96
    for indice in range(nombre_pas):
        heure = 24.0 * (indice + 0.5) / nombre_pas
        total += flux_solaire_incident(latitude_deg, jour_annee, heure)
    return total / nombre_pas


def masse_air_depuis_delta_p(delta_p_pa):
    return delta_p_pa / GRAVITE


def masse_h2o_colonne(humidite_specifique_kgkg, masse_air_kg_m2):
    return max(0.0, humidite_specifique_kgkg) * masse_air_kg_m2


def luminance_spectrale_planck(longueur_onde_m, temperature_k):
    exposant = PLANCK * VITESSE_LUMIERE / (longueur_onde_m * BOLTZMANN * temperature_k)
    if exposant > 700.0:
        return 0.0
    return (
        2.0
        * PLANCK
        * VITESSE_LUMIERE**2
        / longueur_onde_m**5
        / (exp(exposant) - 1.0)
    )


def flux_corps_noir_dans_bande_direct(
    temperature_k,
    lambda_min_um,
    lambda_max_um,
    nombre_pas=2000,
):
    lambda_min_m = lambda_min_um * 1e-6
    lambda_max_m = lambda_max_um * 1e-6
    pas = (lambda_max_m - lambda_min_m) / nombre_pas
    total = 0.0
    for indice in range(nombre_pas):
        longueur_onde_m = lambda_min_m + (indice + 0.5) * pas
        total += pi * luminance_spectrale_planck(longueur_onde_m, temperature_k) * pas
    return total


@lru_cache(maxsize=50000)
def _flux_corps_noir_dans_bande_cache(temperature_k, lambda_min_um, lambda_max_um):
    return flux_corps_noir_dans_bande_direct(
        temperature_k,
        lambda_min_um,
        lambda_max_um,
    )


def flux_corps_noir_dans_bande(temperature_k, lambda_min_um, lambda_max_um, nombre_pas=2000):
    if nombre_pas != 2000:
        return flux_corps_noir_dans_bande_direct(
            temperature_k,
            lambda_min_um,
            lambda_max_um,
            nombre_pas=nombre_pas,
        )
    return _flux_corps_noir_dans_bande_cache(
        round(float(temperature_k), 3),
        float(lambda_min_um),
        float(lambda_max_um),
    )


def flux_lw_surface(temperature_surface_k, emissivite_surface=EMISSIVITE_SURFACE_CONSTANTE):
    return emissivite_surface * SIGMA * temperature_surface_k**4


def tau_co2(couche, bande):
    # Le CO2 depend de la concentration et de l'epaisseur de la couche.
    return (
        bande["a_co2"]
        * (couche["co2_ppm"] / CO2_REFERENCE_PPM)
        * ((couche["pression_bas_pa"] - couche["pression_haut_pa"]) / PRESSION_REFERENCE_PA)
    )


def tau_h2o(couche, bande):
    # La vapeur d'eau depend surtout de la masse d'eau presente dans la colonne.
    return bande["a_h2o"] * (couche["masse_h2o_kg_m2"] / MASSE_H2O_REFERENCE_KG_M2)


def transmission_depuis_tau(tau_total):
    return exp(-FACTEUR_DIFFUSIF * max(tau_total, 0.0))


def opacites_couche_bande(couche, bande):
    opacite_co2 = tau_co2(couche, bande)
    opacite_h2o = tau_h2o(couche, bande)
    # Les deux opacites sont additionnees avant de calculer la transmission.
    tau_total = opacite_co2 + opacite_h2o
    transmission = transmission_depuis_tau(tau_total)
    return {
        "couche": couche["nom"],
        "bande": bande["nom"],
        "tau_co2": opacite_co2,
        "tau_h2o": opacite_h2o,
        "tau_total": tau_total,
        "transmission": transmission,
        "emissivite": 1.0 - transmission,
    }
