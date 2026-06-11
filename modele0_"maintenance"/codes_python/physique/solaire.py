"""Geometrie solaire reprise du groupe Carcajous.

Ce module porte les fonctions solaires afin que `bibliotheque.py` n'importe plus
`fonctions.py`, ce qui supprime l'import circulaire du modele d'origine.
"""

from math import pi

import numpy as np

constante_solaire = 1361.0  # W m-2, irradiance au sommet de l'atmosphere


def get_time_variables(t_sec, lon_deg):
    """Retourne le jour de l'annee et l'heure solaire locale."""
    jour_sim = int(t_sec // 86400)
    day_of_year = jour_sim % 365
    heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
    return day_of_year, heure_solaire


def declination(day):
    """Declinaison solaire en radians pour un jour numerote de 1 a 365."""
    return np.radians(23.44) * np.sin(2 * pi * (284 + day) / 365)


def cos_incidence(lat_rad, day, hour):
    """Cosinus positif de l'angle d'incidence solaire."""
    delta = declination(day)
    hour_angle = np.radians(15 * (hour - 12))
    ci = (
        np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.cos(hour_angle)
    )
    return max(ci, 0.0)


def P_inc_solar(lat_rad, day, hour, albedo_sol, albedo_nuages):
    """Flux solaire net absorbe par la surface selon Carcajous modele 4."""
    phi_entrant = constante_solaire * cos_incidence(lat_rad, day, hour)
    return phi_entrant * (1 - albedo_nuages) * (1 - albedo_sol)

