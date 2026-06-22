"""Modele 2.5 : colonne atmospherique CO2 a 10 couches de pression.

Le modele 2.5 reprend le noyau radiatif du modele 2, mais remplace les
temperatures lues graphiquement par le profil standard 1976, utilise une grille
verticale en pression et decoupe les principales bandes CO2 en sous-bandes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import exp, pi
from pathlib import Path
import sys

RACINE_PROJET = Path(__file__).resolve().parents[2]
try:
    sys.path.remove(str(RACINE_PROJET))
except ValueError:
    pass
sys.path.insert(0, str(RACINE_PROJET))

import numpy as np

from modele2_5.ressources.profil_vertical_atmosphere_co2 import (
    altitude_depuis_pression,
    calculer_profil,
    temperature_moyenne_altitude,
)


# =============================================================================
# Constantes physiques
# =============================================================================

CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
CONSTANTE_PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
CONSTANTE_BOLTZMANN = 1.380649e-23  # J K-1

# =============================================================================
# Parametres globaux du cas de reference
# =============================================================================

TEMPERATURE_SURFACE_K = 288.15
PRESSION_SURFACE_PA = 101_325.0
CO2_REFERENCE_PPM = 280.0
CO2_SURFACE_PPM = 420.0
GRADIENT_CO2_PPM_PAR_KM = 0.0

# Diffusivity approximation : mu_eff ~= 0.6, donc D = 1 / mu_eff ~= 1.66.
FACTEUR_DIFFUSIF = 1.66

# 10 couches en coordonnee pression. Ces niveaux meteorologiques resolvent bien
# la troposphere et conservent un sommet assez haut pour negliger la masse au
# dessus de 1 hPa dans ce modele CO2 simplifie.
PRESSION_BORDS_HPA = (
    1013.25,
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
)

# Echelle multiplicative calibree par ressources/calibrer_coefficients_optiques.py
# pour que OLR(280 ppm) - OLR(560 ppm) soit proche de 3.93 W m-2.
ECHELLE_OPACITE_CO2 = 0.0327228010


@dataclass(frozen=True)
class CoucheAtmospherique:
    """Couche verticale definie par deux pressions d'interface."""

    nom: str
    pression_bas_pa: float
    pression_haut_pa: float
    altitude_bas_km: float
    altitude_haut_km: float
    temperature_k: float
    co2_moyen_ppm: float

    @property
    def epaisseur_pression_pa(self) -> float:
        """Epaisseur de la couche en coordonnee pression."""

        return self.pression_bas_pa - self.pression_haut_pa

    @property
    def altitude_km(self) -> str:
        """Intervalle d'altitude lisible pour les sorties texte."""

        return f"{self.altitude_bas_km:.3f}-{self.altitude_haut_km:.3f}"

    @property
    def pression_hpa(self) -> str:
        """Intervalle de pression lisible pour les sorties texte."""

        return f"{self.pression_bas_pa / 100.0:.3f}-{self.pression_haut_pa / 100.0:.3f}"


@dataclass(frozen=True)
class BandeSpectrale:
    """Sous-bande infrarouge avec opacite effective calibree."""

    nom: str
    longueur_onde_min_um: float
    longueur_onde_max_um: float
    coefficient_opacite: float
    bande_majeure: str
    role: str


BANDES_CO2_BASE = (
    # Les bandes sont separees en coeur et ailes pour garder une opacite simple.
    # Bande de flexion v2 autour de 15 um : coeur tres opaque + ailes actives.
    BandeSpectrale("CO2_15um_aile_gauche_externe", 13.00, 14.00, 0.32, "15 um", "aile"),
    BandeSpectrale("CO2_15um_aile_gauche_interne", 14.00, 14.60, 3.50, "15 um", "aile"),
    BandeSpectrale("CO2_15um_coeur_sature", 14.60, 15.40, 40.0, "15 um", "coeur sature"),
    BandeSpectrale("CO2_15um_aile_droite_interne", 15.40, 16.20, 4.00, "15 um", "aile"),
    BandeSpectrale("CO2_15um_aile_droite_externe", 16.20, 18.00, 0.48, "15 um", "aile"),
    # Bande v3 autour de 4.3 um : incluse, mais faible pour l'IR terrestre.
    BandeSpectrale("CO2_4_3um_aile_gauche", 4.00, 4.20, 0.20, "4.3 um", "aile"),
    BandeSpectrale("CO2_4_3um_coeur_sature", 4.20, 4.40, 15.0, "4.3 um", "coeur sature"),
    BandeSpectrale("CO2_4_3um_aile_droite", 4.40, 4.60, 0.20, "4.3 um", "aile"),
)


def construire_bandes_co2(echelle: float = ECHELLE_OPACITE_CO2) -> tuple[BandeSpectrale, ...]:
    """Applique l'echelle de calibration aux coefficients de base."""

    if echelle < 0.0:
        raise ValueError("L'echelle d'opacite doit etre positive ou nulle.")
    return tuple(
        replace(bande, coefficient_opacite=bande.coefficient_opacite * echelle)
        for bande in BANDES_CO2_BASE
    )


BANDES_CO2 = construire_bandes_co2()


def verifier_pressions_bords(pressions_bords_hpa: tuple[float, ...]) -> None:
    """Verifie que la grille verticale en pression est strictement descendante."""

    if len(pressions_bords_hpa) < 2:
        raise ValueError("Il faut au moins deux pressions d'interface.")
    if any(pression <= 0.0 for pression in pressions_bords_hpa):
        raise ValueError("Les pressions d'interface doivent etre positives.")
    if any(
        pression_bas <= pression_haut
        for pression_bas, pression_haut in zip(
            pressions_bords_hpa[:-1],
            pressions_bords_hpa[1:],
        )
    ):
        raise ValueError("Les pressions doivent decroitre de la surface au sommet.")


@lru_cache(maxsize=4096)
def luminance_spectrale_planck(longueur_onde_m: float, temperature_k: float) -> float:
    """Calcule la luminance spectrale de Planck B_lambda en W m-3 sr-1."""

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
    )


@lru_cache(maxsize=4096)
def flux_corps_noir_dans_bande(
    temperature_k: float,
    longueur_onde_min_um: float,
    longueur_onde_max_um: float,
    nombre_pas: int = 2_000,
) -> float:
    """Integre le flux hemispherique de corps noir dans une bande spectrale."""

    longueur_onde_min_m = longueur_onde_min_um * 1e-6
    longueur_onde_max_m = longueur_onde_max_um * 1e-6
    pas_m = (longueur_onde_max_m - longueur_onde_min_m) / nombre_pas

    total = 0.0
    for indice in range(nombre_pas):
        longueur_onde_m = longueur_onde_min_m + (indice + 0.5) * pas_m
        total += (
            pi
            * luminance_spectrale_planck(longueur_onde_m, temperature_k)
            * pas_m
        )

    return total


def co2_moyen_ppm_par_couche(
    altitude_bas_km: float,
    altitude_haut_km: float,
    co2_surface_ppm: float = CO2_SURFACE_PPM,
    gradient_co2_ppm_par_km: float = GRADIENT_CO2_PPM_PAR_KM,
) -> float:
    """Calcule le CO2 moyen d'une couche, pondere par la masse d'air."""

    altitudes_m = np.linspace(altitude_bas_km * 1000.0, altitude_haut_km * 1000.0, 401)
    profil = calculer_profil(
        altitudes_m,
        co2_surface_ppm,
        gradient_co2_ppm_par_km,
        PRESSION_SURFACE_PA,
        TEMPERATURE_SURFACE_K,
    )

    pressions_pa = profil["pression_pa"]
    co2_ppm = profil["co2_ppm"]
    # La masse d'air d'une tranche suit surtout la difference de pression.
    poids_delta_p = pressions_pa[:-1] - pressions_pa[1:]
    co2_milieu_ppm = 0.5 * (co2_ppm[:-1] + co2_ppm[1:])
    return float(np.sum(co2_milieu_ppm * poids_delta_p) / np.sum(poids_delta_p))


def creer_couches_atmospheriques(
    co2_surface_ppm: float = CO2_SURFACE_PPM,
    gradient_co2_ppm_par_km: float = GRADIENT_CO2_PPM_PAR_KM,
    pressions_bords_hpa: tuple[float, ...] = PRESSION_BORDS_HPA,
) -> tuple[CoucheAtmospherique, ...]:
    """Construit les couches du modele a partir de pressions d'interface."""

    verifier_pressions_bords(pressions_bords_hpa)
    pressions_bords_pa = np.asarray(pressions_bords_hpa, dtype=float) * 100.0
    couches: list[CoucheAtmospherique] = []

    for indice, (pression_bas_pa, pression_haut_pa) in enumerate(
        zip(pressions_bords_pa[:-1], pressions_bords_pa[1:]),
        start=1,
    ):
        # Chaque paire de pressions devient une couche du bas vers le sommet.
        altitude_bas_m = altitude_depuis_pression(float(pression_bas_pa))
        altitude_haut_m = altitude_depuis_pression(float(pression_haut_pa))
        temperature_moyenne_k = temperature_moyenne_altitude(
            altitude_bas_m,
            altitude_haut_m,
        )
        altitude_bas_km = altitude_bas_m / 1000.0
        altitude_haut_km = altitude_haut_m / 1000.0
        couches.append(
            CoucheAtmospherique(
                nom=f"couche_{indice:02d}",
                pression_bas_pa=float(pression_bas_pa),
                pression_haut_pa=float(pression_haut_pa),
                altitude_bas_km=altitude_bas_km,
                altitude_haut_km=altitude_haut_km,
                temperature_k=temperature_moyenne_k,
                co2_moyen_ppm=co2_moyen_ppm_par_couche(
                    altitude_bas_km,
                    altitude_haut_km,
                    co2_surface_ppm,
                    gradient_co2_ppm_par_km,
                ),
            )
        )

    return tuple(couches)


def calculer_profondeur_optique(
    couche: CoucheAtmospherique,
    bande: BandeSpectrale,
    co2_reference_ppm: float = CO2_REFERENCE_PPM,
) -> float:
    """Calcule la profondeur optique effective d'une couche."""

    return (
        bande.coefficient_opacite
        * (couche.co2_moyen_ppm / co2_reference_ppm)
        * (couche.epaisseur_pression_pa / PRESSION_SURFACE_PA)
    )


def transmission_depuis_tau(
    tau: float,
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> float:
    """Convertit une profondeur optique verticale en transmission diffuse."""

    return exp(-facteur_diffusif * tau)


def emissivite_depuis_transmission(transmission: float) -> float:
    """Convertit une transmission en emissivite/absorptivite effective."""

    return 1.0 - transmission


def propager_flux_montant(
    flux_surface_bande: float,
    bande: BandeSpectrale,
    couches: tuple[CoucheAtmospherique, ...],
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> float:
    """Propage le flux infrarouge montant jusqu'au sommet de l'atmosphere."""

    flux = flux_surface_bande
    for couche in couches:
        # A chaque couche, une part traverse et le reste est remplace par son emission.
        tau = calculer_profondeur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau, facteur_diffusif)
        emissivite = emissivite_depuis_transmission(transmission)
        emission_couche = flux_corps_noir_dans_bande(
            couche.temperature_k,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux = transmission * flux + emissivite * emission_couche

    return flux


def propager_flux_descendant(
    bande: BandeSpectrale,
    couches: tuple[CoucheAtmospherique, ...],
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> float:
    """Propage le flux infrarouge descendant vers la surface."""

    flux = 0.0
    for couche in reversed(couches):
        # Meme bilan que vers le haut, mais en repartant du sommet de l'atmosphere.
        tau = calculer_profondeur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau, facteur_diffusif)
        emissivite = emissivite_depuis_transmission(transmission)
        emission_couche = flux_corps_noir_dans_bande(
            couche.temperature_k,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux = transmission * flux + emissivite * emission_couche

    return flux


def calculer_flux_colonne(
    couches: tuple[CoucheAtmospherique, ...],
    bandes: tuple[BandeSpectrale, ...] = BANDES_CO2,
    temperature_surface_k: float = TEMPERATURE_SURFACE_K,
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> tuple[float, float]:
    """Calcule OLR et flux infrarouge descendant a la surface."""

    flux_surface_total = CONSTANTE_STEFAN_BOLTZMANN * temperature_surface_k**4
    flux_surface_bandes_absorbantes = 0.0
    flux_sommet_bandes_absorbantes = 0.0
    flux_descendant_surface = 0.0

    for bande in bandes:
        flux_surface_bande = flux_corps_noir_dans_bande(
            temperature_surface_k,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux_surface_bandes_absorbantes += flux_surface_bande
        flux_sommet_bandes_absorbantes += propager_flux_montant(
            flux_surface_bande,
            bande,
            couches,
            facteur_diffusif,
        )
        flux_descendant_surface += propager_flux_descendant(
            bande,
            couches,
            facteur_diffusif,
        )

    flux_surface_transparent = flux_surface_total - flux_surface_bandes_absorbantes
    flux_sortant_sommet = flux_surface_transparent + flux_sommet_bandes_absorbantes
    return flux_sortant_sommet, flux_descendant_surface


def calculer_flux_par_bande(
    couches: tuple[CoucheAtmospherique, ...],
    bandes: tuple[BandeSpectrale, ...] = BANDES_CO2,
    temperature_surface_k: float = TEMPERATURE_SURFACE_K,
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> list[dict[str, float | str]]:
    """Retourne les contributions de chaque sous-bande pour diagnostic."""

    lignes: list[dict[str, float | str]] = []
    for bande in bandes:
        flux_surface_bande = flux_corps_noir_dans_bande(
            temperature_surface_k,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux_sommet = propager_flux_montant(
            flux_surface_bande,
            bande,
            couches,
            facteur_diffusif,
        )
        flux_descendant = propager_flux_descendant(bande, couches, facteur_diffusif)
        lignes.append(
            {
                "bande": bande.nom,
                "famille": bande.bande_majeure,
                "role": bande.role,
                "lambda_min_um": bande.longueur_onde_min_um,
                "lambda_max_um": bande.longueur_onde_max_um,
                "coefficient_opacite": bande.coefficient_opacite,
                "flux_surface_W_m2": flux_surface_bande,
                "flux_sommet_W_m2": flux_sommet,
                "flux_descendant_surface_W_m2": flux_descendant,
            }
        )
    return lignes


def calculer_forcage_doublement_co2(
    co2_initial_ppm: float = 280.0,
    co2_double_ppm: float = 560.0,
    bandes: tuple[BandeSpectrale, ...] = BANDES_CO2,
    facteur_diffusif: float = FACTEUR_DIFFUSIF,
) -> float:
    """Retourne OLR(C0) - OLR(2*C0), en W m-2, temperatures fixees."""

    couches_initiales = creer_couches_atmospheriques(co2_surface_ppm=co2_initial_ppm)
    couches_double = creer_couches_atmospheriques(co2_surface_ppm=co2_double_ppm)
    olr_initial, _ = calculer_flux_colonne(
        couches_initiales,
        bandes,
        facteur_diffusif=facteur_diffusif,
    )
    olr_double, _ = calculer_flux_colonne(
        couches_double,
        bandes,
        facteur_diffusif=facteur_diffusif,
    )
    return olr_initial - olr_double


def afficher_resume_couches(couches: tuple[CoucheAtmospherique, ...]) -> None:
    """Affiche les grandeurs principales de chaque couche."""

    print("couches_atmospheriques")
    print(
        "nom, pression_hPa, altitude_km, temperature_K, "
        "pression_bas_hPa, pression_haut_hPa, co2_moyen_ppm"
    )
    for couche in couches:
        print(
            f"{couche.nom}, "
            f"{couche.pression_hpa}, "
            f"{couche.altitude_km}, "
            f"{couche.temperature_k:.3f}, "
            f"{couche.pression_bas_pa / 100.0:.3f}, "
            f"{couche.pression_haut_pa / 100.0:.3f}, "
            f"{couche.co2_moyen_ppm:.3f}"
        )


def afficher_resume_opacites(couches: tuple[CoucheAtmospherique, ...]) -> None:
    """Affiche tau, transmission et emissivite pour chaque couche et bande."""

    print()
    print("opacites_par_couche")
    print("couche, bande, tau, transmission, emissivite")
    for couche in couches:
        for bande in BANDES_CO2:
            tau = calculer_profondeur_optique(couche, bande)
            transmission = transmission_depuis_tau(tau)
            emissivite = emissivite_depuis_transmission(transmission)
            print(
                f"{couche.nom}, {bande.nom}, "
                f"{tau:.8f}, {transmission:.8f}, {emissivite:.8f}"
            )


def afficher_resume_bandes(couches: tuple[CoucheAtmospherique, ...]) -> None:
    """Affiche les flux par sous-bande."""

    print()
    print("flux_par_sous_bande")
    print(
        "bande, famille, role, lambda_um, coefficient_opacite, "
        "flux_surface_W_m2, flux_sommet_W_m2, flux_descendant_surface_W_m2"
    )
    for ligne in calculer_flux_par_bande(couches):
        print(
            f"{ligne['bande']}, "
            f"{ligne['famille']}, "
            f"{ligne['role']}, "
            f"{ligne['lambda_min_um']:.2f}-{ligne['lambda_max_um']:.2f}, "
            f"{ligne['coefficient_opacite']:.8f}, "
            f"{ligne['flux_surface_W_m2']:.6f}, "
            f"{ligne['flux_sommet_W_m2']:.6f}, "
            f"{ligne['flux_descendant_surface_W_m2']:.6f}"
        )


def main() -> None:
    """Point d'entree du script."""

    couches = creer_couches_atmospheriques()
    flux_sortant_sommet, flux_descendant_surface = calculer_flux_colonne(couches)
    forcage_doublement = calculer_forcage_doublement_co2()

    afficher_resume_couches(couches)
    afficher_resume_opacites(couches)
    afficher_resume_bandes(couches)
    print()
    print(f"facteur_diffusif = {FACTEUR_DIFFUSIF:.3f}")
    print(f"echelle_opacite_co2 = {ECHELLE_OPACITE_CO2:.8f}")
    print(f"forcage_280_560_ppm_W_m2 = {forcage_doublement:.6f}")
    print(f"flux_infrarouge_sortant_sommet_W_m2 = {flux_sortant_sommet:.6f}")
    print(f"flux_infrarouge_descendant_surface_W_m2 = {flux_descendant_surface:.6f}")


if __name__ == "__main__":
    main()
