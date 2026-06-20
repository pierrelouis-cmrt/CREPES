"""Modele 3 : colonne radiative simplifiee a 6 couches avec CH4 uniquement.

Cette version reprend l'idee du modele 2 :
- emission infrarouge de la surface ;
- absorption et reemission par bandes spectrales ;
- propagation d'un flux montant et d'un flux descendant.

La nouveaute est l'ajout du methane (CH4) avec sa propre bande d'absorption.
Chaque gaz possede sa concentration et son opacite est calculee selon :

    tau_gaz = a_bande * (concentration / reference) * (delta_p / p_surface)
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, pi

import numpy as np



# =========================
# Constantes physiques
# =========================

CONSTANTE_STEFAN_BOLTZMANN = 5.670374419e-8  # W m-2 K-4
CONSTANTE_PLANCK = 6.62607015e-34  # J s
VITESSE_LUMIERE = 299_792_458.0  # m s-1
CONSTANTE_BOLTZMANN = 1.380649e-23  # J K-1


# =========================
# Parametres globaux
# =========================

TEMPERATURE_SURFACE_K = 288.15
PRESSION_SURFACE_PA = 101_325.0

# Parametres CH4
CH4_REFERENCE_PPM = 0.72  # 720 ppb pre-industriel
CH4_SURFACE_PPM = 1.90    # 1900 ppb actuel
GRADIENT_CH4_PPM_PAR_KM = 0.0

FACTEUR_DIFFUSIF = 1.0


@dataclass(frozen=True)
class CoucheAtmospherique:
    """Une couche verticale avec temperature imposee et concentrations."""

    nom: str
    altitude_bas_km: float
    altitude_haut_km: float
    temperature_k: float
    pression_bas_pa: float
    pression_haut_pa: float
    ch4_moyen_ppm: float

    @property
    def epaisseur_pression_pa(self) -> float:
        return self.pression_bas_pa - self.pression_haut_pa


@dataclass(frozen=True)
class BandeSpectrale:
    """Bande infrarouge avec coefficient d'opacite effectif et gaz associe."""

    nom: str
    gaz: str  # "CH4"
    longueur_onde_min_um: float
    longueur_onde_max_um: float
    coefficient_opacite: float


COUCHES_DEPART = (
    ("couche_1_troposphere_basse", 0.0, 5.0, 271.0),
    ("couche_2_troposphere_moyenne", 5.0, 10.0, 236.0),
    ("couche_3_tropopause", 10.0, 30.0, 223.0),
    ("couche_4_stratosphere", 30.0, 50.0, 257.0),
    ("couche_5_mesosphere_basse", 50.0, 65.0, 252.0),
    ("couche_6_mesosphere_haute", 65.0, 80.0, 212.0),
)

BANDES_ABSORPTION = (
    BandeSpectrale("CH4_7_6um", "CH4", 7.30, 8.00, 1.5),  # Opacite estimative du CH4
)


def luminance_spectrale_planck(longueur_onde_m: float, temperature_k: float) -> float:
    """Luminance spectrale de Planck B_lambda en W m-3 sr-1."""
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


def flux_corps_noir_bande(
    temperature_k: float,
    longueur_onde_min_um: float,
    longueur_onde_max_um: float,
    nombre_pas: int = 2_000,
) -> float:
    """Flux hemispherique de corps noir integre dans une bande spectrale."""
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


def calculer_atmosphere_standard(
    altitudes_m: np.ndarray,
    pression_surface_pa: float,
    temperature_surface_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule la temperature et la pression selon l'atmosphere standard internationale (ISA)."""
    g0 = 9.80665
    R = 287.0528

    h_b = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0, 100000.0])
    beta = np.array([-0.0065, 0.0, 0.001, 0.0028, 0.0, -0.0028, -0.002, 0.0])
    
    t_b = np.zeros(8)
    t_b[0] = temperature_surface_k
    for i in range(1, 8):
        t_b[i] = t_b[i-1] + beta[i-1] * (h_b[i] - h_b[i-1])

    p_b = np.zeros(8)
    p_b[0] = pression_surface_pa
    for i in range(1, 8):
        if beta[i-1] != 0.0:
            p_b[i] = p_b[i-1] * (t_b[i] / t_b[i-1]) ** (-g0 / (beta[i-1] * R))
        else:
            p_b[i] = p_b[i-1] * np.exp(-g0 * (h_b[i] - h_b[i-1]) / (R * t_b[i-1]))

    temperatures = np.zeros_like(altitudes_m, dtype=float)
    pressions = np.zeros_like(altitudes_m, dtype=float)

    for j, z in enumerate(altitudes_m):
        for i in range(len(h_b) - 1):
            if h_b[i] <= z <= h_b[i+1] or (i == 6 and z > h_b[7]):
                if beta[i] != 0.0:
                    temperatures[j] = t_b[i] + beta[i] * (z - h_b[i])
                    pressions[j] = p_b[i] * (temperatures[j] / t_b[i]) ** (-g0 / (beta[i] * R))
                else:
                    temperatures[j] = t_b[i]
                    pressions[j] = p_b[i] * np.exp(-g0 * (z - h_b[i]) / (R * t_b[i]))
                break

    return temperatures, pressions


def concentration_moyenne_ppm_par_couche(
    altitude_bas_km: float,
    altitude_haut_km: float,
    surface_ppm: float,
    gradient_ppm_par_km: float,
) -> float:
    """Calcule une concentration moyenne ponderee par la masse d'air de la couche."""
    altitudes_m = np.linspace(altitude_bas_km * 1000.0, altitude_haut_km * 1000.0, 201)
    
    _, pressions_pa = calculer_atmosphere_standard(
        altitudes_m,
        PRESSION_SURFACE_PA,
        TEMPERATURE_SURFACE_K,
    )

    concentration_ppm = surface_ppm + gradient_ppm_par_km * (altitudes_m / 1000.0)

    poids_delta_p = pressions_pa[:-1] - pressions_pa[1:]
    concentration_milieu_ppm = 0.5 * (concentration_ppm[:-1] + concentration_ppm[1:])

    masse_air = np.sum(poids_delta_p)
    if masse_air == 0:
        return float(np.mean(concentration_ppm))

    return float(np.sum(concentration_milieu_ppm * poids_delta_p) / masse_air)


def creer_couches() -> tuple[CoucheAtmospherique, ...]:
    """Construit les couches a partir des donnees de depart et des concentrations."""
    altitudes_bords_m = np.array(
        [altitude for _, altitude, _, _ in COUCHES_DEPART]
        + [COUCHES_DEPART[-1][2]]
    ) * 1000.0
    _, pressions_bords_pa = calculer_atmosphere_standard(
        altitudes_bords_m,
        PRESSION_SURFACE_PA,
        TEMPERATURE_SURFACE_K,
    )

    couches = []
    for indice, (nom, altitude_bas_km, altitude_haut_km, temperature_k) in enumerate(
        COUCHES_DEPART
    ):
        couches.append(
            CoucheAtmospherique(
                nom=nom,
                altitude_bas_km=altitude_bas_km,
                altitude_haut_km=altitude_haut_km,
                temperature_k=temperature_k,
                pression_bas_pa=float(pressions_bords_pa[indice]),
                pression_haut_pa=float(pressions_bords_pa[indice + 1]),
                ch4_moyen_ppm=concentration_moyenne_ppm_par_couche(
                    altitude_bas_km, altitude_haut_km, CH4_SURFACE_PPM, GRADIENT_CH4_PPM_PAR_KM
                ),
            )
        )

    return tuple(couches)


def epaisseur_optique(couche: CoucheAtmospherique, bande: BandeSpectrale) -> float:
    """Profondeur optique effective de la couche en fonction du gaz."""
    if bande.gaz == "CH4":
        concentration_ppm = couche.ch4_moyen_ppm
        reference_ppm = CH4_REFERENCE_PPM
    else:
        raise ValueError(f"Gaz inconnu : {bande.gaz}")

    return (
        bande.coefficient_opacite
        * (concentration_ppm / reference_ppm)
        * (couche.epaisseur_pression_pa / PRESSION_SURFACE_PA)
    )


def transmission_depuis_tau(tau: float) -> float:
    return exp(-FACTEUR_DIFFUSIF * tau)


def emissivite_depuis_transmission(transmission: float) -> float:
    return 1.0 - transmission


def propager_flux_montant(
    flux_surface_bande: float,
    bande: BandeSpectrale,
    couches: tuple[CoucheAtmospherique, ...],
) -> float:
    flux = flux_surface_bande
    for couche in couches:
        tau = epaisseur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau)
        emissivite = emissivite_depuis_transmission(transmission)
        emission_couche = flux_corps_noir_bande(
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
    flux = 0.0
    for couche in reversed(couches):
        tau = epaisseur_optique(couche, bande)
        transmission = transmission_depuis_tau(tau)
        emissivite = emissivite_depuis_transmission(transmission)
        emission_couche = flux_corps_noir_bande(
            couche.temperature_k,
            bande.longueur_onde_min_um,
            bande.longueur_onde_max_um,
        )
        flux = transmission * flux + emissivite * emission_couche
    return flux


def calculer_flux(
    couches: tuple[CoucheAtmospherique, ...],
    bandes: tuple[BandeSpectrale, ...] = BANDES_ABSORPTION,
) -> tuple[float, float]:
    flux_surface_total = CONSTANTE_STEFAN_BOLTZMANN * TEMPERATURE_SURFACE_K**4
    flux_surface_bandes_absorbantes = 0.0
    flux_sommet_bandes_absorbantes = 0.0
    flux_descendant_surface = 0.0

    for bande in bandes:
        flux_surface_bande = flux_corps_noir_bande(
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
    print("couches_atmospheriques")
    print(
        "nom, altitude_km, temperature_K, pression_bas_hPa, "
        "pression_haut_hPa, ch4_moyen_ppm"
    )
    for couche in couches:
        print(
            f"{couche.nom}, "
            f"{couche.altitude_bas_km:g}-{couche.altitude_haut_km:g}, "
            f"{couche.temperature_k:.2f}, "
            f"{couche.pression_bas_pa / 100.0:.3f}, "
            f"{couche.pression_haut_pa / 100.0:.3f}, "
            f"{couche.ch4_moyen_ppm:.3f}"
        )


def afficher_resume_opacites(couches: tuple[CoucheAtmospherique, ...]) -> None:
    print("\nopacites_par_couche")
    print("couche, bande, gaz, tau, transmission, emissivite")
    for couche in couches:
        for bande in BANDES_ABSORPTION:
            tau = epaisseur_optique(couche, bande)
            transmission = transmission_depuis_tau(tau)
            emissivite = emissivite_depuis_transmission(transmission)
            print(
                f"{couche.nom}, {bande.nom}, {bande.gaz}, "
                f"{tau:.6f}, {transmission:.6f}, {emissivite:.6f}"
            )


def main() -> None:
    couches = creer_couches()
    flux_sortant_sommet, flux_descendant_surface = calculer_flux(couches)

    afficher_resume_couches(couches)
    afficher_resume_opacites(couches)
    print()
    print(f"flux_infrarouge_sortant_sommet_W_m2 = {flux_sortant_sommet:.6f}")
    print(f"flux_infrarouge_descendant_surface_W_m2 = {flux_descendant_surface:.6f}")


if __name__ == "__main__":
    main()