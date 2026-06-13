"""Modèle 2 : colonne atmosphérique CO2 à 6 couches.

Ce script est le noyau radiatif simplifié du modèle 2. Il ne calcule pas encore
l'évolution temporelle des températures : les températures de surface et de
couches sont imposées, puis le script calcule uniquement les flux infrarouges.

Principe du calcul :

1. découper l'atmosphère en 6 couches verticales ;
2. associer à chaque couche une température, une pression basse, une pression
   haute et une concentration moyenne de CO2 ;
3. convertir la quantité de CO2 de chaque couche en profondeur optique ;
4. propager les flux infrarouges montant et descendant couche par couche.

Formule d'opacité utilisée dans chaque bande spectrale :

    tau = a_bande * (CO2_moyen / CO2_reference) * (delta_p / pression_surface)

où ``a_bande`` est un coefficient effectif à calibrer, et non une constante
fondamentale. Le profil de pression et la moyenne de CO2 par couche viennent du
script ``profil_vertical_atmosphere_co2.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi

import numpy as np

from profil_vertical_atmosphere_co2 import atmosphere_standard, calculer_profil


# =============================================================================
# Constantes physiques
# =============================================================================

CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
CONSTANTE_PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
CONSTANTE_BOLTZMANN = 1.380649e-23  # J K-1


# =============================================================================
# Paramètres globaux du cas de référence
# =============================================================================

TEMPERATURE_SURFACE_K = 288.15  # température de surface imposée
PRESSION_SURFACE_PA = 101_325.0  # pression de surface imposée
CO2_REFERENCE_PPM = 280.0  # référence préindustrielle utilisée pour normaliser tau
CO2_SURFACE_PPM = 420.0  # concentration de CO2 au niveau de la surface
GRADIENT_CO2_PPM_PAR_KM = 0.0  # 0 signifie que le CO2 est bien mélangé

# D = 1 garde un trajet vertical simple. D = 1.66 pourra être testé ensuite
# pour représenter grossièrement des trajets obliques moyens.
FACTEUR_DIFFUSIF = 1.0


@dataclass(frozen=True)
class CoucheAtmospherique:
    """Couche verticale du modèle.

    Les pressions sont les pressions aux interfaces basse et haute de la couche.
    Leur différence représente la masse d'air de la couche par unité de surface,
    à un facteur ``1/g`` près.
    """

    nom: str
    altitude_bas_km: float
    altitude_haut_km: float
    temperature_k: float
    pression_bas_pa: float
    pression_haut_pa: float
    co2_moyen_ppm: float

    @property
    def epaisseur_pression_pa(self) -> float:
        """Épaisseur de la couche en coordonnée pression."""

        return self.pression_bas_pa - self.pression_haut_pa

    @property
    def altitude_km(self) -> str:
        """Intervalle d'altitude lisible pour les sorties texte."""

        return f"{self.altitude_bas_km:g}-{self.altitude_haut_km:g}"


@dataclass(frozen=True)
class BandeSpectrale:
    """Bande infrarouge avec un coefficient d'opacité effectif."""

    nom: str
    longueur_onde_min_um: float
    longueur_onde_max_um: float
    coefficient_opacite: float


# Températures lues sur l'image fournie par l'utilisateur.
# Le découpage reste volontairement simple : le modèle 2 est encore un prototype
# de noyau radiatif, pas un modèle atmosphérique complet.
PARAMETRES_COUCHES = (
    ("couche_1_troposphere_basse", 0.0, 5.0, 271.0),
    ("couche_2_troposphere_moyenne", 5.0, 10.0, 236.0),
    ("couche_3_tropopause", 10.0, 30.0, 223.0),
    ("couche_4_stratosphere", 30.0, 50.0, 257.0),
    ("couche_5_mesosphere_basse", 50.0, 65.0, 252.0),
    ("couche_6_mesosphere_haute", 65.0, 80.0, 212.0),
)

# Coefficients de départ repris comme opacités effectives de bande.
# Ils devront être recalibrés avec le test de forçage 280 -> 560 ppm.
BANDES_CO2 = (
    BandeSpectrale("CO2_15um", 14.25, 15.75, 1.0),
    BandeSpectrale("CO2_4_3um", 4.20, 4.35, 3.25),
)


def luminance_spectrale_planck(longueur_onde_m: float, temperature_k: float) -> float:
    """Calcule la luminance spectrale de Planck ``B_lambda``.

    Unités de sortie : W m-3 sr-1.
    """

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


def flux_corps_noir_dans_bande(
    temperature_k: float,
    longueur_onde_min_um: float,
    longueur_onde_max_um: float,
    nombre_pas: int = 2_000,
) -> float:
    """Intègre le flux hémisphérique de corps noir dans une bande spectrale.

    Le facteur ``pi`` transforme la luminance spectrale en flux hémisphérique
    pour une surface lambertienne. L'intégration numérique utilise la méthode
    des milieux, suffisante ici pour un prototype lisible.
    """

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
    """Calcule le CO2 moyen d'une couche, pondéré par la masse d'air.

    En équilibre hydrostatique, la masse d'air par unité de surface est
    proportionnelle à ``delta_p``. On moyenne donc le CO2 avec les différences
    de pression entre points successifs du profil vertical.
    """

    altitudes_m = np.linspace(altitude_bas_km * 1000.0, altitude_haut_km * 1000.0, 201)
    profil = calculer_profil(
        altitudes_m,
        co2_surface_ppm,
        gradient_co2_ppm_par_km,
        PRESSION_SURFACE_PA,
        TEMPERATURE_SURFACE_K,
    )

    pressions_pa = profil["pression_pa"]
    co2_ppm = profil["co2_ppm"]
    poids_delta_p = pressions_pa[:-1] - pressions_pa[1:]
    co2_milieu_ppm = 0.5 * (co2_ppm[:-1] + co2_ppm[1:])

    return float(np.sum(co2_milieu_ppm * poids_delta_p) / np.sum(poids_delta_p))


def creer_couches_atmospheriques() -> tuple[CoucheAtmospherique, ...]:
    """Construit les 6 couches à partir du découpage et du profil vertical."""

    altitudes_bords_m = np.array(
        [altitude for _, altitude, _, _ in PARAMETRES_COUCHES]
        + [PARAMETRES_COUCHES[-1][2]]
    ) * 1000.0
    _, pressions_bords_pa = atmosphere_standard(
        altitudes_bords_m,
        PRESSION_SURFACE_PA,
        TEMPERATURE_SURFACE_K,
    )

    couches = []
    for indice, (nom, altitude_bas_km, altitude_haut_km, temperature_k) in enumerate(
        PARAMETRES_COUCHES
    ):
        couches.append(
            CoucheAtmospherique(
                nom=nom,
                altitude_bas_km=altitude_bas_km,
                altitude_haut_km=altitude_haut_km,
                temperature_k=temperature_k,
                pression_bas_pa=float(pressions_bords_pa[indice]),
                pression_haut_pa=float(pressions_bords_pa[indice + 1]),
                co2_moyen_ppm=co2_moyen_ppm_par_couche(
                    altitude_bas_km,
                    altitude_haut_km,
                ),
            )
        )

    return tuple(couches)


def calculer_profondeur_optique(
    couche: CoucheAtmospherique,
    bande: BandeSpectrale,
) -> float:
    """Calcule la profondeur optique effective d'une couche.

    La profondeur optique augmente avec :

    - le coefficient de bande ``a_b`` ;
    - la concentration moyenne de CO2 de la couche ;
    - l'épaisseur de la couche en pression, qui représente sa masse d'air.
    """

    return (
        bande.coefficient_opacite
        * (couche.co2_moyen_ppm / CO2_REFERENCE_PPM)
        * (couche.epaisseur_pression_pa / PRESSION_SURFACE_PA)
    )


def transmission_depuis_tau(tau: float) -> float:
    """Convertit une profondeur optique en transmission Beer-Lambert."""

    return exp(-FACTEUR_DIFFUSIF * tau)


def emissivite_depuis_transmission(transmission: float) -> float:
    """Convertit une transmission en émissivité.

    Sans diffusion ni réflexion, toute l'énergie non transmise est absorbée.
    Par la loi de Kirchhoff, l'absorptivité vaut l'émissivité.
    """

    return 1.0 - transmission


def propager_flux_montant(
    flux_surface_bande: float,
    bande: BandeSpectrale,
    couches: tuple[CoucheAtmospherique, ...],
) -> float:
    """Propage le flux infrarouge montant jusqu'au sommet de l'atmosphère.

    À chaque couche, une fraction du flux incident traverse la couche et une
    fraction est remplacée par l'émission thermique propre de cette couche.
    """

    flux = flux_surface_bande
    for couche in couches:
        tau = calculer_profondeur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau)
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
) -> float:
    """Propage le flux infrarouge descendant vers la surface.

    Le flux descendant au sommet de l'atmosphère est nul : on suppose qu'aucun
    rayonnement infrarouge externe n'entre par le haut de la colonne.
    """

    flux = 0.0
    for couche in reversed(couches):
        tau = calculer_profondeur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau)
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
) -> tuple[float, float]:
    """Calcule les deux flux infrarouges principaux de la colonne.

    Retourne :

    - ``flux_sortant_sommet`` : OLR, flux infrarouge sortant au sommet ;
    - ``flux_descendant_surface`` : flux infrarouge atmosphérique reçu par la
      surface.

    Les bandes explicitement absorbantes sont traitées couche par couche. Le
    reste du spectre est considéré transparent et sort directement vers l'espace.
    """

    flux_surface_total = CONSTANTE_STEFAN_BOLTZMANN * TEMPERATURE_SURFACE_K**4
    flux_surface_bandes_absorbantes = 0.0
    flux_sommet_bandes_absorbantes = 0.0
    flux_descendant_surface = 0.0

    for bande in bandes:
        flux_surface_bande = flux_corps_noir_dans_bande(
            TEMPERATURE_SURFACE_K,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux_surface_bandes_absorbantes += flux_surface_bande
        flux_sommet_bandes_absorbantes += propager_flux_montant(
            flux_surface_bande,
            bande,
            couches,
        )
        flux_descendant_surface += propager_flux_descendant(bande, couches)

    flux_surface_transparent = flux_surface_total - flux_surface_bandes_absorbantes
    flux_sortant_sommet = flux_surface_transparent + flux_sommet_bandes_absorbantes

    return flux_sortant_sommet, flux_descendant_surface


def afficher_resume_couches(couches: tuple[CoucheAtmospherique, ...]) -> None:
    """Affiche les grandeurs principales de chaque couche."""

    print("couches_atmospheriques")
    print(
        "nom, altitude_km, temperature_K, pression_bas_hPa, "
        "pression_haut_hPa, co2_moyen_ppm"
    )
    for couche in couches:
        print(
            f"{couche.nom}, "
            f"{couche.altitude_km}, "
            f"{couche.temperature_k:.2f}, "
            f"{couche.pression_bas_pa / 100.0:.3f}, "
            f"{couche.pression_haut_pa / 100.0:.3f}, "
            f"{couche.co2_moyen_ppm:.3f}"
        )


def afficher_resume_opacites(couches: tuple[CoucheAtmospherique, ...]) -> None:
    """Affiche tau, transmission et émissivité pour chaque couche et bande."""

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
                f"{tau:.6f}, {transmission:.6f}, {emissivite:.6f}"
            )


def main() -> None:
    """Point d'entrée du script."""

    couches = creer_couches_atmospheriques()
    flux_sortant_sommet, flux_descendant_surface = calculer_flux_colonne(couches)

    afficher_resume_couches(couches)
    afficher_resume_opacites(couches)
    print()
    print(f"flux_infrarouge_sortant_sommet_W_m2 = {flux_sortant_sommet:.6f}")
    print(f"flux_infrarouge_descendant_surface_W_m2 = {flux_descendant_surface:.6f}")


if __name__ == "__main__":
    main()
