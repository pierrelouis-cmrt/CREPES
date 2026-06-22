"""Constantes et flux publics du moteur CREPES.

Les noms de fonctions Carcajous sont conserves pour compatibilite avec les
scripts de modele 4.
"""

from physique import chaleur_latente, solaire
from physique.capacite_surface import EPAISSEUR_ACTIVE

# Constantes globales gardées sous les noms historiques du moteur Carcajous.
constante_solaire = solaire.constante_solaire
sigma = 5.670374419e-8
Tatm = 223.15
dt = 1800.0

# Les fonctions actives sont réexportées pour ne pas casser les anciens imports.
P_inc_solar = solaire.P_inc_solar
P_em_surf_evap = chaleur_latente.P_em_surf_evap


def P_em_surf_thermal(T: float):
    """Puissance thermique emise par la surface, loi de Stefan-Boltzmann."""
    return sigma * (T**4)


def P_em_atm_thermal(T_atm: float):
    """Puissance thermique atmospherique descendante simplifiee."""
    return sigma * (T_atm**4)


def P_em_surf_conv(lat: float, long: float, t: float):
    """Placeholder Carcajous conserve: flux convectif gere dans modele_courbe."""
    return 0


# Les placeholders ci-dessous restent présents pour les vieux scripts,
# mais le modèle ponctuel actif ne les branche pas encore.
def P_abs_atm_solar(lat: float, long: float, t: float, Pinc: float):
    """Placeholder Carcajous conserve: absorption solaire atm non activee."""
    return 0


def P_em_atm_thermal_up(lat: float, long: float, t: float):
    """Placeholder Carcajous conserve: emission atm vers le haut."""
    return 0


def P_em_atm_thermal_down(lat: float, long: float, t: float):
    """Placeholder Carcajous conserve: emission atm vers la surface."""
    return 0
