"""Calibrage spectral des coefficients CO2 du modele 3.

Le script transforme un calcul de reference HITRAN, effectue via RADIS, en
coefficients effectifs `a_co2` compatibles avec la formule du modele 3 :

    tau_CO2 = a_co2 * (CO2_ppm / 280) * (delta_p / 101325)

Il fait ensuite un recalage global optionnel sur le forcage OLR 280 -> 560 ppm
et quantifie la contribution des bandes coeur/aile. Les fonctions numeriques
de base restent importables sans RADIS ; RADIS n'est requis qu'au moment du
calcul spectral.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import physique
    from .donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from .modele3 import calculer_colonne_radiative
except ImportError:  # Permet aussi : python modele3/calibrer_coefficients_co2.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3 import physique
    from modele3.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.modele3 import calculer_colonne_radiative


R_UNIVERSEL = 8.31446261815324  # J mol-1 K-1
MASSE_MOLAIRE_AIR = 0.0289647  # kg mol-1
PRESSION_REFERENCE_PA = 101_325.0
CO2_REFERENCE_PPM = physique.CO2_REFERENCE_PPM
CO2_DOUBLE_PPM = 2.0 * CO2_REFERENCE_PPM
FORCAGE_MYHRE_2XCO2 = 5.35 * math.log(2.0)
TRANSMISSION_MIN = 1e-12

SOURCES_METHODE = [
    {
        "nom": "HITRANonline / HAPI",
        "usage": "base de donnees spectroscopiques ligne par ligne et outils de transmission",
        "url": "https://hitran.org/hapi/",
    },
    {
        "nom": "HITRAN Application Programming Interface paper",
        "usage": "description de HAPI et des calculs d'absorption/transmission",
        "doi": "10.1016/j.jqsrt.2016.03.005",
        "url": "https://doi.org/10.1016/j.jqsrt.2016.03.005",
    },
    {
        "nom": "HITRAN2024 molecular spectroscopic database",
        "usage": "reference actuelle des parametres spectroscopiques HITRAN",
        "url": "https://hitran.org/hapi/",
    },
    {
        "nom": "RADIS calc_spectrum",
        "usage": "calcul line-by-line Python utilisant HITRAN comme databank",
        "url": "https://radis.readthedocs.io/en/latest/source/radis.lbl.calc.html",
    },
    {
        "nom": "Myhre et al. 1998",
        "usage": "cible de forcage a temperature fixee : DeltaF = 5.35 ln(C/C0)",
        "doi": "10.1029/98GL01908",
        "url": "https://doi.org/10.1029/98GL01908",
    },
    {
        "nom": "IPCC AR6 WGI Chapter 7 Supplementary Material",
        "usage": "valeur ERF 2xCO2 = 3.93 W m-2 et contexte AR6",
        "url": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf",
    },
    {
        "nom": "ERA5 pressure levels",
        "usage": "source des profils locaux T(p), q(p) du paquet modele 3",
        "doi": "10.24381/cds.bd0915c6",
        "url": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels",
    },
]


@dataclass(frozen=True)
class ColonneCalibration:
    identifiant: str
    donnees: dict[str, Any]
    poids: float
    temperature_surface_k: float


@dataclass(frozen=True)
class CoucheReference:
    identifiant_colonne: str
    identifiant_couche: str
    latitude_deg: float
    longitude_deg: float
    mois: int
    pression_bas_pa: float
    pression_haut_pa: float
    pression_milieu_bar: float
    delta_p_pa: float
    temperature_k: float
    chemin_optique_cm: float
    poids_colonne: float


@dataclass(frozen=True)
class MesureReference:
    bande: str
    role: str
    lambda_min_um: float
    lambda_max_um: float
    couche: str
    colonne: str
    co2_ppm: float
    x_modele: float
    transmission_ref: float
    tau_eq_ref: float
    poids_fit: float


def lire_liste_float(texte: str) -> list[float]:
    valeurs = []
    for morceau in str(texte).split(","):
        morceau = morceau.strip()
        if morceau:
            valeurs.append(float(morceau))
    if not valeurs:
        raise ValueError("La liste ne peut pas etre vide.")
    return valeurs


def lire_liste_int(texte: str) -> list[int]:
    valeurs = [int(valeur) for valeur in lire_liste_float(texte)]
    if any(valeur < 1 or valeur > 12 for valeur in valeurs):
        raise ValueError("Les mois doivent etre compris entre 1 et 12.")
    return valeurs


def plage_nombre_onde_cm(bande: dict[str, Any]) -> tuple[float, float]:
    """Convertit une bande en micrometres vers un intervalle en cm-1."""

    nu_min = 10_000.0 / float(bande["lambda_max_um"])
    nu_max = 10_000.0 / float(bande["lambda_min_um"])
    return min(nu_min, nu_max), max(nu_min, nu_max)


def poids_planck_nombre_onde_cm(nombre_onde_cm: np.ndarray, temperature_k: float) -> np.ndarray:
    """Poids de Planck par nombre d'onde, a une constante multiplicative pres."""

    nombre_onde_m = np.asarray(nombre_onde_cm, dtype=float) * 100.0
    x = (
        physique.PLANCK
        * physique.VITESSE_LUMIERE
        * nombre_onde_m
        / (physique.BOLTZMANN * temperature_k)
    )
    poids = np.zeros_like(nombre_onde_m, dtype=float)
    masque = x < 700.0
    poids[masque] = nombre_onde_m[masque] ** 3 / np.expm1(x[masque])
    return poids


def moyenne_planck_transmission(
    nombre_onde_cm: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
) -> float:
    """Moyenne la transmission spectrale avec le flux thermique de la couche."""

    nombre_onde_cm = np.asarray(nombre_onde_cm, dtype=float)
    transmission = np.clip(np.asarray(transmission, dtype=float), 0.0, 1.0)
    ordre = np.argsort(nombre_onde_cm)
    nombre_onde_cm = nombre_onde_cm[ordre]
    transmission = transmission[ordre]
    poids = poids_planck_nombre_onde_cm(nombre_onde_cm, temperature_k)
    integrer = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    denominateur = float(integrer(poids, nombre_onde_cm))
    if denominateur <= 0.0:
        return float(np.mean(transmission))
    numerateur = float(integrer(poids * transmission, nombre_onde_cm))
    return max(TRANSMISSION_MIN, min(1.0, numerateur / denominateur))


def tau_equivalent_depuis_transmission(transmission_moyenne: float) -> float:
    """Convertit Tbar_ref en tau compatible avec exp(-D * tau)."""

    transmission_moyenne = max(TRANSMISSION_MIN, min(1.0, float(transmission_moyenne)))
    return -math.log(transmission_moyenne) / physique.FACTEUR_DIFFUSIF


def ajuster_coefficient_moindres_carres(
    x_modele: list[float] | np.ndarray,
    tau_reference: list[float] | np.ndarray,
    poids: list[float] | np.ndarray | None = None,
) -> float:
    """Ajuste tau_reference ~= a * x_modele par moindres carres ponderes."""

    x = np.asarray(x_modele, dtype=float)
    tau = np.asarray(tau_reference, dtype=float)
    if poids is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(poids, dtype=float)
    masque = np.isfinite(x) & np.isfinite(tau) & np.isfinite(w) & (x > 0.0) & (w > 0.0)
    if not np.any(masque):
        raise ValueError("Aucune mesure valide pour ajuster le coefficient.")
    x = x[masque]
    tau = tau[masque]
    w = w[masque]
    denominateur = float(np.sum(w * x * x))
    if denominateur <= 0.0:
        raise ValueError("Denominateur nul dans l'ajustement.")
    return float(np.sum(w * x * tau) / denominateur)


def transmission_modele_depuis_a(a_co2: float, x_modele: np.ndarray) -> np.ndarray:
    tau = max(0.0, float(a_co2)) * np.asarray(x_modele, dtype=float)
    return np.exp(-physique.FACTEUR_DIFFUSIF * tau)


def erreur_quadratique_ponderee(valeurs: np.ndarray, poids: np.ndarray) -> float:
    valeurs = np.asarray(valeurs, dtype=float)
    poids = np.asarray(poids, dtype=float)
    masque = np.isfinite(valeurs) & np.isfinite(poids) & (poids > 0.0)
    if not np.any(masque):
        return float("nan")
    return float(math.sqrt(np.sum(poids[masque] * valeurs[masque] ** 2) / np.sum(poids[masque])))


def bandes_co2_modele3() -> list[dict[str, Any]]:
    return [
        copy.deepcopy(bande)
        for bande in physique.BANDES_INFRAROUGES
        if float(bande.get("a_co2", 0.0)) > 0.0
    ]


def poids_aire_latitude(latitude_deg: float) -> float:
    return max(0.0, math.cos(math.radians(latitude_deg)))


def temperature_surface_colonne(colonne: dict[str, Any], source: str, defaut: float) -> float:
    surface = colonne["surface"]
    if source == "skin" and surface.get("skin_temperature_k") is not None:
        return float(surface["skin_temperature_k"])
    if source == "t2m" and surface.get("temperature_2m_k") is not None:
        return float(surface["temperature_2m_k"])
    if source == "auto":
        if surface.get("skin_temperature_k") is not None:
            return float(surface["skin_temperature_k"])
        if surface.get("temperature_2m_k") is not None:
            return float(surface["temperature_2m_k"])
    return float(defaut)


def charger_colonnes_calibration(args: argparse.Namespace) -> list[ColonneCalibration]:
    paquet = charger_paquet_grille(args.paquet)
    latitudes = lire_liste_float(args.latitudes)
    longitudes = lire_liste_float(args.longitudes)
    mois_liste = lire_liste_int(args.mois)

    colonnes = []
    for mois in mois_liste:
        for latitude in latitudes:
            for longitude in longitudes:
                donnees = extraire_colonne(paquet, latitude, longitude, mois=mois)
                latitude_reelle = donnees["surface"]["latitude_deg"]
                longitude_reelle = donnees["surface"]["longitude_deg"]
                identifiant = f"lat{latitude_reelle:+06.1f}_lon{longitude_reelle:+07.1f}_m{mois:02d}"
                colonnes.append(
                    ColonneCalibration(
                        identifiant=identifiant,
                        donnees=donnees,
                        poids=poids_aire_latitude(latitude_reelle),
                        temperature_surface_k=temperature_surface_colonne(
                            donnees,
                            args.temperature_surface_source,
                            args.temperature_surface,
                        ),
                    )
                )
    somme_poids = sum(colonne.poids for colonne in colonnes)
    if somme_poids <= 0.0:
        raise ValueError("Les poids de colonnes sont nuls.")
    return colonnes


def couche_reference_depuis_modele(
    colonne: ColonneCalibration,
    couche: dict[str, Any],
) -> CoucheReference:
    pression_bas_pa = float(couche["pression_bas_pa"])
    pression_haut_pa = float(couche["pression_haut_pa"])
    delta_p_pa = pression_bas_pa - pression_haut_pa
    temperature_k = float(couche["temperature_k"])
    pression_milieu_pa = 0.5 * (pression_bas_pa + pression_haut_pa)

    masse_air_kg_m2 = delta_p_pa / physique.GRAVITE
    densite_air_kg_m3 = pression_milieu_pa * MASSE_MOLAIRE_AIR / (R_UNIVERSEL * temperature_k)
    epaisseur_m = masse_air_kg_m2 / densite_air_kg_m3

    surface = colonne.donnees["surface"]
    return CoucheReference(
        identifiant_colonne=colonne.identifiant,
        identifiant_couche=str(couche["nom"]),
        latitude_deg=float(surface["latitude_deg"]),
        longitude_deg=float(surface["longitude_deg"]),
        mois=int(surface["mois"]),
        pression_bas_pa=pression_bas_pa,
        pression_haut_pa=pression_haut_pa,
        pression_milieu_bar=pression_milieu_pa / 100_000.0,
        delta_p_pa=delta_p_pa,
        temperature_k=temperature_k,
        chemin_optique_cm=epaisseur_m * 100.0,
        poids_colonne=colonne.poids,
    )


def extraire_transmission_spectre(spectre: Any) -> tuple[np.ndarray, np.ndarray]:
    for grandeur in ("transmittance_noslit", "transmittance"):
        try:
            nombre_onde_cm, transmission = spectre.get(grandeur)
            return np.asarray(nombre_onde_cm, dtype=float), np.asarray(transmission, dtype=float)
        except Exception:
            continue
    raise RuntimeError(
        "RADIS n'a pas renvoye de transmittance exploitable "
        "('transmittance_noslit' ou 'transmittance')."
    )


def databank_radis(valeur: str) -> str | tuple[str, str]:
    if valeur == "hitran-range":
        return ("hitran", "range")
    if valeur == "hitran-full":
        return ("hitran", "full")
    return valeur


def calculer_spectre_radis_hitran(
    bande: dict[str, Any],
    couche: CoucheReference,
    co2_ppm: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule T_ref(nu) pour une couche homogene avec RADIS/HITRAN."""

    try:
        from radis import calc_spectrum
    except ImportError as exc:
        raise RuntimeError(
            "RADIS n'est pas installe. Installer les dependances optionnelles avec "
            "`python -m pip install -r modele3/requirements-calibrage.txt`."
        ) from exc

    wmin, wmax = plage_nombre_onde_cm(bande)
    kwargs = {
        "isotope": args.isotopes,
        "pressure": couche.pression_milieu_bar,
        "Tgas": couche.temperature_k,
        "mole_fraction": co2_ppm * 1e-6,
        "path_length": couche.chemin_optique_cm,
        "databank": databank_radis(args.databank),
        "medium": "air",
        "wstep": args.wstep,
        "truncation": args.truncation,
        "verbose": args.radis_verbose,
    }

    try:
        spectre = calc_spectrum(wmin, wmax, molecule="CO2", **kwargs)
    except TypeError as exc:
        if "molecule" not in str(exc):
            raise
        spectre = calc_spectrum(wmin, wmax, species="CO2", **kwargs)
    return extraire_transmission_spectre(spectre)


def construire_mesures_reference(
    colonnes: list[ColonneCalibration],
    bandes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[MesureReference]:
    co2_values = lire_liste_float(args.co2_values)
    mesures: list[MesureReference] = []

    for colonne in colonnes:
        for couche_modele in colonne.donnees["couches"]:
            couche = couche_reference_depuis_modele(colonne, couche_modele)
            for bande in bandes:
                flux_planck = physique.flux_corps_noir_dans_bande(
                    couche.temperature_k,
                    bande["lambda_min_um"],
                    bande["lambda_max_um"],
                )
                poids_fit_base = couche.poids_colonne
                if not args.sans_ponderation_planck:
                    poids_fit_base *= max(flux_planck, 1e-30)

                for co2_ppm in co2_values:
                    nombre_onde_cm, transmission = calculer_spectre_radis_hitran(
                        bande,
                        couche,
                        co2_ppm,
                        args,
                    )
                    transmission_moyenne = moyenne_planck_transmission(
                        nombre_onde_cm,
                        transmission,
                        couche.temperature_k,
                    )
                    mesures.append(
                        MesureReference(
                            bande=bande["nom"],
                            role=bande["role"],
                            lambda_min_um=float(bande["lambda_min_um"]),
                            lambda_max_um=float(bande["lambda_max_um"]),
                            couche=couche.identifiant_couche,
                            colonne=couche.identifiant_colonne,
                            co2_ppm=float(co2_ppm),
                            x_modele=(
                                (float(co2_ppm) / CO2_REFERENCE_PPM)
                                * (couche.delta_p_pa / PRESSION_REFERENCE_PA)
                            ),
                            transmission_ref=transmission_moyenne,
                            tau_eq_ref=tau_equivalent_depuis_transmission(transmission_moyenne),
                            poids_fit=poids_fit_base,
                        )
                    )
    return mesures


def ajuster_coefficients_par_bande(
    mesures: list[MesureReference],
    bandes: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    resultats: dict[str, dict[str, Any]] = {}
    mesures_par_bande: dict[str, list[MesureReference]] = {}
    for mesure in mesures:
        mesures_par_bande.setdefault(mesure.bande, []).append(mesure)

    bandes_par_nom = {bande["nom"]: bande for bande in bandes}
    for nom_bande, mesures_bande in mesures_par_bande.items():
        x = np.asarray([mesure.x_modele for mesure in mesures_bande], dtype=float)
        tau = np.asarray([mesure.tau_eq_ref for mesure in mesures_bande], dtype=float)
        transmission = np.asarray([mesure.transmission_ref for mesure in mesures_bande], dtype=float)
        poids = np.asarray([mesure.poids_fit for mesure in mesures_bande], dtype=float)
        a_co2 = ajuster_coefficient_moindres_carres(x, tau, poids)
        tau_modele = a_co2 * x
        transmission_modele = transmission_modele_depuis_a(a_co2, x)
        bande = bandes_par_nom[nom_bande]
        resultats[nom_bande] = {
            "nom": nom_bande,
            "role": bande["role"],
            "lambda_min_um": float(bande["lambda_min_um"]),
            "lambda_max_um": float(bande["lambda_max_um"]),
            "a_co2_avant": float(bande["a_co2"]),
            "a_co2_hitran": a_co2,
            "rmse_tau": erreur_quadratique_ponderee(tau_modele - tau, poids),
            "rmse_transmission": erreur_quadratique_ponderee(transmission_modele - transmission, poids),
            "transmission_ref_moyenne": float(np.average(transmission, weights=poids)),
            "tau_eq_ref_moyen": float(np.average(tau, weights=poids)),
            "nombre_mesures": len(mesures_bande),
        }
    return resultats


def bandes_avec_coefficients(
    coefficients: dict[str, dict[str, Any]],
    facteur: float,
    zero_h2o: bool = False,
) -> list[dict[str, Any]]:
    bandes = []
    for bande in physique.BANDES_INFRAROUGES:
        copie = copy.deepcopy(bande)
        if copie["nom"] in coefficients:
            copie["a_co2"] = float(coefficients[copie["nom"]]["a_co2_hitran"]) * facteur
        if zero_h2o:
            copie["a_h2o"] = 0.0
        bandes.append(copie)
    return bandes


def calculer_forcage_echantillon(
    colonnes: list[ColonneCalibration],
    bandes: list[dict[str, Any]],
    co2_initial: float,
    co2_final: float,
) -> float:
    numerateur = 0.0
    denominateur = 0.0
    for colonne in colonnes:
        resultat_initial = calculer_colonne_radiative(
            colonne.donnees,
            temperature_surface_k=colonne.temperature_surface_k,
            co2_ppm=co2_initial,
            moyenne_journaliere_sw=True,
            bandes=bandes,
        )
        resultat_final = calculer_colonne_radiative(
            colonne.donnees,
            temperature_surface_k=colonne.temperature_surface_k,
            co2_ppm=co2_final,
            moyenne_journaliere_sw=True,
            bandes=bandes,
        )
        numerateur += colonne.poids * (resultat_initial["OLR"] - resultat_final["OLR"])
        denominateur += colonne.poids
    if denominateur <= 0.0:
        raise ValueError("Poids de colonnes nul pour le forcage.")
    return numerateur / denominateur


def trouver_facteur_forcage(
    colonnes: list[ColonneCalibration],
    coefficients: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    cible = float(args.cible_forcage)
    zero_h2o = bool(args.zero_h2o_pour_forcage)

    def forcage(facteur: float) -> float:
        bandes = bandes_avec_coefficients(coefficients, facteur, zero_h2o=zero_h2o)
        return calculer_forcage_echantillon(
            colonnes,
            bandes,
            CO2_REFERENCE_PPM,
            CO2_DOUBLE_PPM,
        )

    forcage_brut = forcage(1.0)
    if cible <= 0.0:
        return {
            "cible_W_m2": cible,
            "facteur_global": 1.0,
            "forcage_brut_W_m2": forcage_brut,
            "forcage_final_W_m2": forcage_brut,
            "avertissement": "cible <= 0, recalage ignore",
        }

    bas = 0.0
    haut = 1.0
    forcage_haut = forcage_brut
    while forcage_haut < cible and haut < args.facteur_max:
        haut *= 2.0
        forcage_haut = forcage(haut)

    if forcage_haut < cible:
        return {
            "cible_W_m2": cible,
            "facteur_global": haut,
            "forcage_brut_W_m2": forcage_brut,
            "forcage_final_W_m2": forcage_haut,
            "avertissement": (
                "La cible n'est pas atteinte avant facteur_max; "
                "verifier les bandes, l'echantillon ou la cible."
            ),
        }

    for _ in range(args.iterations_forcage):
        milieu = 0.5 * (bas + haut)
        forcage_milieu = forcage(milieu)
        if forcage_milieu < cible:
            bas = milieu
        else:
            haut = milieu

    facteur = 0.5 * (bas + haut)
    return {
        "cible_W_m2": cible,
        "facteur_global": facteur,
        "forcage_brut_W_m2": forcage_brut,
        "forcage_final_W_m2": forcage(facteur),
        "avertissement": None,
    }


def normaliser_role(role: str) -> str:
    role_min = role.lower()
    if "coeur" in role_min:
        return "coeur"
    if "aile" in role_min:
        return "aile"
    return role_min


def diagnostics_bandes_par_nom(resultat: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {diagnostic["bande"]: diagnostic for diagnostic in resultat["diagnostics_bandes"]}


def quantifier_impacts_bandes(
    colonnes: list[ColonneCalibration],
    bandes: list[dict[str, Any]],
    co2_initial: float = CO2_REFERENCE_PPM,
    co2_final: float = CO2_DOUBLE_PPM,
) -> dict[str, Any]:
    impacts: dict[str, dict[str, Any]] = {}
    somme_poids = sum(colonne.poids for colonne in colonnes)
    if somme_poids <= 0.0:
        raise ValueError("Poids de colonnes nul pour les impacts.")

    for bande in bandes:
        if float(bande.get("a_co2", 0.0)) > 0.0:
            impacts[bande["nom"]] = {
                "bande": bande["nom"],
                "role": bande["role"],
                "role_groupe": normaliser_role(bande["role"]),
                "lambda_min_um": float(bande["lambda_min_um"]),
                "lambda_max_um": float(bande["lambda_max_um"]),
                "a_co2_final": float(bande["a_co2"]),
                "olr_280_W_m2": 0.0,
                "olr_560_W_m2": 0.0,
                "tau_CO2_total_280": 0.0,
                "tau_CO2_total_560": 0.0,
                "impact_W_m2": 0.0,
            }

    for colonne in colonnes:
        resultat_initial = calculer_colonne_radiative(
            colonne.donnees,
            temperature_surface_k=colonne.temperature_surface_k,
            co2_ppm=co2_initial,
            moyenne_journaliere_sw=True,
            bandes=bandes,
        )
        resultat_final = calculer_colonne_radiative(
            colonne.donnees,
            temperature_surface_k=colonne.temperature_surface_k,
            co2_ppm=co2_final,
            moyenne_journaliere_sw=True,
            bandes=bandes,
        )
        diag_initial = diagnostics_bandes_par_nom(resultat_initial)
        diag_final = diagnostics_bandes_par_nom(resultat_final)
        for nom_bande, impact in impacts.items():
            poids = colonne.poids / somme_poids
            olr_initial = diag_initial[nom_bande]["flux_sommet_W_m2"]
            olr_final = diag_final[nom_bande]["flux_sommet_W_m2"]
            impact["olr_280_W_m2"] += poids * olr_initial
            impact["olr_560_W_m2"] += poids * olr_final
            impact["tau_CO2_total_280"] += poids * diag_initial[nom_bande]["tau_CO2_total"]
            impact["tau_CO2_total_560"] += poids * diag_final[nom_bande]["tau_CO2_total"]

    total_impact = 0.0
    for impact in impacts.values():
        impact["impact_W_m2"] = impact["olr_280_W_m2"] - impact["olr_560_W_m2"]
        total_impact += impact["impact_W_m2"]

    for impact in impacts.values():
        if abs(total_impact) > 1e-12:
            impact["part_impact_pct"] = 100.0 * impact["impact_W_m2"] / total_impact
        else:
            impact["part_impact_pct"] = 0.0

    groupes: dict[str, dict[str, Any]] = {}
    for impact in impacts.values():
        groupe = groupes.setdefault(
            impact["role_groupe"],
            {"role_groupe": impact["role_groupe"], "impact_W_m2": 0.0, "part_impact_pct": 0.0},
        )
        groupe["impact_W_m2"] += impact["impact_W_m2"]
    for groupe in groupes.values():
        if abs(total_impact) > 1e-12:
            groupe["part_impact_pct"] = 100.0 * groupe["impact_W_m2"] / total_impact

    return {
        "co2_initial_ppm": co2_initial,
        "co2_final_ppm": co2_final,
        "impact_total_W_m2": total_impact,
        "par_bande": sorted(impacts.values(), key=lambda item: item["lambda_min_um"]),
        "par_role": sorted(groupes.values(), key=lambda item: item["role_groupe"]),
    }


def arrondir_json(objet: Any, chiffres: int = 10) -> Any:
    if isinstance(objet, float):
        if not math.isfinite(objet):
            return None
        return round(objet, chiffres)
    if isinstance(objet, dict):
        return {cle: arrondir_json(valeur, chiffres) for cle, valeur in objet.items()}
    if isinstance(objet, list):
        return [arrondir_json(valeur, chiffres) for valeur in objet]
    return objet


def ecrire_snippet_python(
    chemin: Path,
    coefficients: dict[str, dict[str, Any]],
    facteur_global: float,
) -> None:
    lignes = [
        '"""Coefficients CO2 calibres par modele3.calibrer_coefficients_co2.',
        "",
        "Ce fichier est un artefact de sortie. Il ne modifie pas automatiquement",
        "modele3/physique.py : comparer les diagnostics avant de remplacer les",
        "valeurs de production.",
        '"""',
        "",
        f"FACTEUR_FORCAGE_GLOBAL = {facteur_global:.12g}",
        "A_CO2_CALIBRES = {",
    ]
    for nom, valeurs in sorted(coefficients.items()):
        a_final = valeurs["a_co2_hitran"] * facteur_global
        lignes.append(f'    "{nom}": {a_final:.12g},')
    lignes.extend(["}", ""])
    chemin.write_text("\n".join(lignes), encoding="utf-8")


def construire_resultat(
    args: argparse.Namespace,
    colonnes: list[ColonneCalibration],
    mesures: list[MesureReference],
    coefficients: dict[str, dict[str, Any]],
    recalage: dict[str, Any],
    impacts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "methode": {
            "resume": (
                "HITRAN/RADIS -> transmission spectrale -> moyenne Planck par bande "
                "-> tau equivalent -> moindres carres -> recalage forcage -> impacts coeur/aile"
            ),
            "formule_modele3": (
                "tau_CO2 = a_CO2 * (CO2_ppm / 280) * (delta_p / 101325)"
            ),
            "transmission_modele3": "T = exp(-1.66 * (tau_CO2 + tau_H2O))",
            "sources": SOURCES_METHODE,
        },
        "configuration": {
            "latitudes": args.latitudes,
            "longitudes": args.longitudes,
            "mois": args.mois,
            "co2_values": args.co2_values,
            "temperature_surface_source": args.temperature_surface_source,
            "temperature_surface_defaut_K": args.temperature_surface,
            "wstep_cm_1": args.wstep,
            "truncation_cm_1": args.truncation,
            "databank": args.databank,
            "isotopes": args.isotopes,
            "cible_forcage_W_m2": args.cible_forcage,
            "zero_h2o_pour_forcage": args.zero_h2o_pour_forcage,
            "nombre_colonnes": len(colonnes),
            "nombre_mesures_reference": len(mesures),
        },
        "colonnes": [
            {
                "identifiant": colonne.identifiant,
                "poids": colonne.poids,
                "temperature_surface_k": colonne.temperature_surface_k,
                "latitude_deg": colonne.donnees["surface"]["latitude_deg"],
                "longitude_deg": colonne.donnees["surface"]["longitude_deg"],
                "mois": colonne.donnees["surface"]["mois"],
            }
            for colonne in colonnes
        ],
        "coefficients": [
            {
                **valeurs,
                "facteur_forcage_global": recalage["facteur_global"],
                "a_co2_final": valeurs["a_co2_hitran"] * recalage["facteur_global"],
            }
            for valeurs in sorted(coefficients.values(), key=lambda item: item["lambda_min_um"])
        ],
        "forcage": recalage,
        "impacts": impacts,
        "mesures_reference": [asdict(mesure) for mesure in mesures],
    }


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Calibrer les coefficients CO2 effectifs du modele 3 avec HITRAN/RADIS."
    )
    parseur.add_argument(
        "--paquet",
        type=Path,
        default=DOSSIER_PAQUET_DEFAUT,
        help="Dossier du paquet compact modele3.",
    )
    parseur.add_argument(
        "--latitudes",
        default="-45,0,45",
        help="Latitudes d'echantillonnage, format CSV. Exemple: --latitudes=-60,-30,0,30,60",
    )
    parseur.add_argument(
        "--longitudes",
        default="0",
        help="Longitudes d'echantillonnage, format CSV.",
    )
    parseur.add_argument(
        "--mois",
        default="1,7",
        help="Mois d'echantillonnage, format CSV.",
    )
    parseur.add_argument(
        "--co2-values",
        default="280,420,560,1120",
        help="Concentrations CO2 utilisees pour le fit spectral.",
    )
    parseur.add_argument(
        "--temperature-surface",
        type=float,
        default=physique.TEMPERATURE_SURFACE_DEFAUT_K,
        help="Temperature de surface de repli pour le calcul du forcage.",
    )
    parseur.add_argument(
        "--temperature-surface-source",
        choices=("auto", "skin", "t2m", "constante"),
        default="auto",
        help="Source de T_surface pour le forcage OLR.",
    )
    parseur.add_argument(
        "--databank",
        default="hitran-range",
        help="Databank RADIS: hitran-range, hitran-full, hitran ou chemin local.",
    )
    parseur.add_argument(
        "--isotopes",
        default="1,2,3",
        help="Isotopologues CO2 RADIS/HITRAN.",
    )
    parseur.add_argument(
        "--wstep",
        type=float,
        default=0.02,
        help="Pas spectral RADIS en cm-1.",
    )
    parseur.add_argument(
        "--truncation",
        type=float,
        default=50.0,
        help="Troncature des profils de raies RADIS en cm-1.",
    )
    parseur.add_argument(
        "--cible-forcage",
        type=float,
        default=FORCAGE_MYHRE_2XCO2,
        help=(
            "Cible OLR(280)-OLR(560) en W m-2. Defaut: Myhre 5.35 ln(2). "
            "Utiliser 3.93 pour la valeur ERF AR6."
        ),
    )
    parseur.add_argument(
        "--zero-h2o-pour-forcage",
        action="store_true",
        help="Met a zero les bandes H2O pendant le recalage forcage.",
    )
    parseur.add_argument(
        "--sans-ponderation-planck",
        action="store_true",
        help="Ajuste les tau sans ponderation par flux de Planck.",
    )
    parseur.add_argument(
        "--facteur-max",
        type=float,
        default=128.0,
        help="Borne haute du facteur global de recalage.",
    )
    parseur.add_argument(
        "--iterations-forcage",
        type=int,
        default=40,
        help="Iterations de bissection du recalage forcage.",
    )
    parseur.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ressources" / "calibrage_opacite_co2",
        help="Dossier de sortie des artefacts de calibrage.",
    )
    parseur.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche le volume de calcul sans appeler RADIS/HITRAN.",
    )
    parseur.add_argument(
        "--radis-verbose",
        action="store_true",
        help="Laisse RADIS afficher ses messages.",
    )
    return parseur


def main() -> None:
    args = construire_parseur().parse_args()
    colonnes = charger_colonnes_calibration(args)
    bandes = bandes_co2_modele3()
    co2_values = lire_liste_float(args.co2_values)
    nb_couches = sum(len(colonne.donnees["couches"]) for colonne in colonnes)
    nb_spectres = nb_couches * len(bandes) * len(co2_values)

    if args.dry_run:
        print("calibrage_co2_dry_run")
        print(f"colonnes = {len(colonnes)}")
        print(f"couches = {nb_couches}")
        print(f"bandes_co2 = {len(bandes)}")
        print(f"concentrations = {len(co2_values)}")
        print(f"spectres_radis_hitran = {nb_spectres}")
        return

    mesures = construire_mesures_reference(colonnes, bandes, args)
    coefficients = ajuster_coefficients_par_bande(mesures, bandes)
    recalage = trouver_facteur_forcage(colonnes, coefficients, args)
    bandes_finales = bandes_avec_coefficients(
        coefficients,
        recalage["facteur_global"],
        zero_h2o=args.zero_h2o_pour_forcage,
    )
    impacts = quantifier_impacts_bandes(colonnes, bandes_finales)
    resultat = construire_resultat(args, colonnes, mesures, coefficients, recalage, impacts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "calibrage_coefficients_co2.json"
    snippet_path = args.output_dir / "coefficients_co2_calibres.py"
    json_path.write_text(
        json.dumps(arrondir_json(resultat), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ecrire_snippet_python(snippet_path, coefficients, recalage["facteur_global"])

    print("calibrage_co2_ok")
    print(f"json = {json_path}")
    print(f"snippet = {snippet_path}")
    print(f"forcage_cible_W_m2 = {recalage['cible_W_m2']:.6f}")
    print(f"forcage_final_W_m2 = {recalage['forcage_final_W_m2']:.6f}")
    print(f"facteur_global = {recalage['facteur_global']:.8f}")
    if recalage.get("avertissement"):
        print(f"avertissement = {recalage['avertissement']}")


if __name__ == "__main__":
    main()
