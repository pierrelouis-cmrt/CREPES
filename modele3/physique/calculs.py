"""Calculs physiques elementaires du modele 3.

Ce module ne lit pas de fichiers et ne pilote pas la colonne complete. Il
contient seulement des formules reutilisables : solaire, Planck, masses colonne,
opacites, albedo nuage et flux radiatifs simples.
"""

from __future__ import annotations

from math import cos, exp, isfinite, pi, radians, sin


# =============================================================================
# Constantes physiques et reglages effectifs
# =============================================================================

SIGMA = 5.670374419e-8  # W m-2 K-4
PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
BOLTZMANN = 1.380649e-23  # J K-1
GRAVITE = 9.80665  # m s-2

CONSTANTE_SOLAIRE = 1361.0  # W m-2
TEMPERATURE_SURFACE_DEFAUT_K = 288.15
CO2_REFERENCE_PPM = 280.0
CO2_DEFAUT_PPM = 420.0

FACTEUR_DIFFUSIF = 1.66
ECHELLE_OPACITE_CO2 = 0.0327228010
MASSE_H2O_REFERENCE_KG_M2 = 10.0

COEFFICIENT_NUAGE_SW = 0.50
COEFFICIENT_NUAGE_LW = 0.10

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


# =============================================================================
# Outils scalaires simples
# =============================================================================


def borner(valeur, minimum, maximum):
    return max(minimum, min(maximum, valeur))


def fraction(valeur, defaut=0.0):
    if valeur is None:
        return defaut
    try:
        valeur = float(valeur)
    except (TypeError, ValueError):
        return defaut
    if not isfinite(valeur):
        return defaut
    return borner(valeur, 0.0, 1.0)


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


# =============================================================================
# Solaire
# =============================================================================


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


# =============================================================================
# Moyennes verticales et masses colonne
# =============================================================================


def _points_valides(pressions_hpa, valeurs):
    points = []
    for pression, valeur in zip(pressions_hpa, valeurs):
        pression = float(pression)
        valeur = float(valeur)
        if isfinite(pression) and isfinite(valeur):
            points.append((pression, valeur))
    if not points:
        raise ValueError("Profil vertical vide.")
    return sorted(points)


def interpoler_pression(pression_hpa, pressions_hpa, valeurs):
    points = _points_valides(pressions_hpa, valeurs)
    pressions = [point[0] for point in points]
    valeurs = [point[1] for point in points]

    if pression_hpa <= pressions[0]:
        return valeurs[0]
    if pression_hpa >= pressions[-1]:
        return valeurs[-1]

    for indice in range(len(pressions) - 1):
        p0 = pressions[indice]
        p1 = pressions[indice + 1]
        if p0 <= pression_hpa <= p1:
            poids = (pression_hpa - p0) / (p1 - p0)
            return valeurs[indice] + poids * (valeurs[indice + 1] - valeurs[indice])

    return valeurs[-1]


def moyenne_pression(pressions_hpa, valeurs, pression_bas_hpa, pression_haut_hpa):
    if pression_bas_hpa <= pression_haut_hpa:
        raise ValueError("pression_bas_hpa doit etre plus grande que pression_haut_hpa.")

    points = [pression_haut_hpa, pression_bas_hpa]
    for pression, _valeur in _points_valides(pressions_hpa, valeurs):
        if pression_haut_hpa < pression < pression_bas_hpa:
            points.append(pression)
    points = sorted(set(points))

    integrale = 0.0
    for p0, p1 in zip(points[:-1], points[1:]):
        v0 = interpoler_pression(p0, pressions_hpa, valeurs)
        v1 = interpoler_pression(p1, pressions_hpa, valeurs)
        integrale += 0.5 * (v0 + v1) * (p1 - p0)

    return integrale / (pression_bas_hpa - pression_haut_hpa)


def masse_air_depuis_delta_p(delta_p_pa):
    return delta_p_pa / GRAVITE


def masse_h2o_colonne(humidite_specifique_kgkg, masse_air_kg_m2):
    return max(0.0, humidite_specifique_kgkg) * masse_air_kg_m2


# =============================================================================
# Long-onde, opacites et court-onde simple
# =============================================================================


def luminance_spectrale_planck(longueur_onde_m, temperature_k):
    exposant = PLANCK * VITESSE_LUMIERE / (longueur_onde_m * BOLTZMANN * temperature_k)
    return (
        2.0
        * PLANCK
        * VITESSE_LUMIERE**2
        / longueur_onde_m**5
        / (exp(exposant) - 1.0)
    )


def flux_corps_noir_dans_bande(temperature_k, lambda_min_um, lambda_max_um, nombre_pas=2000):
    lambda_min_m = lambda_min_um * 1e-6
    lambda_max_m = lambda_max_um * 1e-6
    pas = (lambda_max_m - lambda_min_m) / nombre_pas
    total = 0.0

    for indice in range(nombre_pas):
        longueur_onde_m = lambda_min_m + (indice + 0.5) * pas
        total += pi * luminance_spectrale_planck(longueur_onde_m, temperature_k) * pas

    return total


def flux_lw_surface(temperature_surface_k, emissivite_surface):
    return emissivite_surface * SIGMA * temperature_surface_k**4


def tau_co2(couche, bande):
    return (
        bande["a_co2"]
        * (couche["co2_ppm"] / CO2_REFERENCE_PPM)
        * ((couche["pression_bas_pa"] - couche["pression_haut_pa"]) / 101_325.0)
    )


def tau_h2o(couche, bande):
    return bande["a_h2o"] * (couche["masse_h2o_kg_m2"] / MASSE_H2O_REFERENCE_KG_M2)


def tau_nuage(couche):
    return COEFFICIENT_NUAGE_LW * couche["fraction_nuageuse"]


def transmission_depuis_tau(tau_total):
    return exp(-FACTEUR_DIFFUSIF * max(tau_total, 0.0))


def opacites_couche_bande(couche, bande):
    opacite_co2 = tau_co2(couche, bande)
    opacite_h2o = tau_h2o(couche, bande)
    opacite_nuage = tau_nuage(couche)
    tau_total = opacite_co2 + opacite_h2o + opacite_nuage
    transmission = transmission_depuis_tau(tau_total)

    return {
        "couche": couche["nom"],
        "bande": bande["nom"],
        "tau_co2": opacite_co2,
        "tau_h2o": opacite_h2o,
        "tau_nuage": opacite_nuage,
        "tau_total": tau_total,
        "transmission": transmission,
        "emissivite": 1.0 - transmission,
    }


def albedo_nuage_effectif(cloud_total):
    return borner(COEFFICIENT_NUAGE_SW * fraction(cloud_total), 0.0, 0.95)


def flux_sw_absorbe_surface(sw_incident_surface, albedo_surface, albedo_cloud):
    return sw_incident_surface * (1.0 - albedo_surface) * (1.0 - albedo_cloud)
