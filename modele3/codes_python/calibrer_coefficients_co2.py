"""Calibrage minimal des coefficients CO2 du modele 3.

Le but est seulement de compresser un calcul HITRAN/RADIS en coefficients
effectifs compatibles avec le modele 3 :

    tau_CO2 = a_co2 * (CO2_ppm / 280) * (delta_p / 101325)

Pour chaque bande, on calcule une transmission RADIS, on la moyenne avec le
poids de Planck de la couche, on la convertit en tau equivalent, puis on retient
la mediane de tau/X. Un facteur global recale ensuite le forcage 280 -> 560 ppm.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import physique
    from .coefficients_opacite import CHEMIN_COEFFICIENTS_CO2
    from .donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from .modele3 import calculer_colonne_radiative
except ImportError:  # Permet aussi : python modele3/codes_python/calibrer_coefficients_co2.py
    dossier_script = str(Path(__file__).resolve().parent)
    sys.path = [chemin for chemin in sys.path if chemin != dossier_script]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique
    from modele3.codes_python.coefficients_opacite import CHEMIN_COEFFICIENTS_CO2
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.codes_python.modele3 import calculer_colonne_radiative


R_UNIVERSEL = 8.31446261815324
MASSE_MOLAIRE_AIR = 0.0289647
PRESSION_REFERENCE_PA = physique.PRESSION_REFERENCE_PA
CO2_REFERENCE_PPM = physique.CO2_REFERENCE_PPM
CO2_DOUBLE_PPM = 2.0 * CO2_REFERENCE_PPM
FORCAGE_MYHRE_2XCO2 = 5.35 * math.log(2.0)
TRANSMISSION_MIN = 1e-12
FACTEUR_MAX = 128.0
ITERATIONS_FORCAGE = 40

RADIS_DATABANK = ("hitran", "range")
RADIS_ISOTOPES = "1,2,3"
RADIS_TRUNCATION_CM_1 = 50.0


@dataclass(frozen=True)
class ColonneCalibration:
    identifiant: str
    donnees: dict[str, Any]
    poids: float
    temperature_surface_k: float


@dataclass(frozen=True)
class CoucheReference:
    delta_p_pa: float
    pression_milieu_bar: float
    temperature_k: float
    chemin_optique_cm: float


def lire_liste_float(texte: str) -> list[float]:
    valeurs = [float(morceau.strip()) for morceau in str(texte).split(",") if morceau.strip()]
    if not valeurs:
        raise ValueError("La liste ne peut pas etre vide.")
    return valeurs


def lire_liste_int(texte: str) -> list[int]:
    valeurs = [int(valeur) for valeur in lire_liste_float(texte)]
    if any(valeur < 1 or valeur > 12 for valeur in valeurs):
        raise ValueError("Les mois doivent etre compris entre 1 et 12.")
    return valeurs


def lire_wstep(valeur: str) -> float | str:
    texte = str(valeur).strip().lower()
    if texte == "auto":
        return "auto"
    pas = float(texte)
    if pas <= 0.0:
        raise ValueError("wstep doit etre strictement positif, ou valoir 'auto'.")
    return pas


def plage_nombre_onde_cm(bande: dict[str, Any]) -> tuple[float, float]:
    nu_min = 10_000.0 / float(bande["lambda_max_um"])
    nu_max = 10_000.0 / float(bande["lambda_min_um"])
    return min(nu_min, nu_max), max(nu_min, nu_max)


def poids_planck_nombre_onde_cm(nombre_onde_cm: np.ndarray, temperature_k: float) -> np.ndarray:
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
    nombre_onde_cm = np.asarray(nombre_onde_cm, dtype=float)
    transmission = np.clip(np.asarray(transmission, dtype=float), 0.0, 1.0)
    if nombre_onde_cm.size == 0 or transmission.size == 0:
        raise ValueError("Spectre RADIS vide.")

    ordre = np.argsort(nombre_onde_cm)
    nombre_onde_cm = nombre_onde_cm[ordre]
    transmission = transmission[ordre]
    poids = poids_planck_nombre_onde_cm(nombre_onde_cm, temperature_k)

    integrer = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    denominateur = float(integrer(poids, nombre_onde_cm))
    if denominateur <= 0.0:
        return max(TRANSMISSION_MIN, min(1.0, float(np.mean(transmission))))

    moyenne = float(integrer(poids * transmission, nombre_onde_cm)) / denominateur
    return max(TRANSMISSION_MIN, min(1.0, moyenne))


def tau_equivalent_depuis_transmission(transmission_moyenne: float) -> float:
    transmission_moyenne = max(TRANSMISSION_MIN, min(1.0, float(transmission_moyenne)))
    return -math.log(transmission_moyenne) / physique.FACTEUR_DIFFUSIF


def coefficient_effectif_median(
    x_modele: list[float] | np.ndarray,
    tau_reference: list[float] | np.ndarray,
) -> float:
    x = np.asarray(x_modele, dtype=float)
    tau = np.asarray(tau_reference, dtype=float)
    masque = np.isfinite(x) & np.isfinite(tau) & (x > 0.0) & (tau >= 0.0)
    if not np.any(masque):
        raise ValueError("Aucune mesure valide pour ajuster le coefficient.")
    return float(np.median(tau[masque] / x[masque]))


def temperature_surface_colonne(colonne: dict[str, Any]) -> float:
    surface = colonne["surface"]
    for cle in ("skin_temperature_k", "temperature_2m_k"):
        valeur = surface.get(cle)
        if valeur is not None and math.isfinite(float(valeur)):
            return float(valeur)
    return physique.TEMPERATURE_SURFACE_DEFAUT_K


def charger_colonnes_calibration(args: argparse.Namespace) -> list[ColonneCalibration]:
    paquet = charger_paquet_grille(args.paquet)
    colonnes: list[ColonneCalibration] = []

    for mois in lire_liste_int(args.mois):
        for latitude in lire_liste_float(args.latitudes):
            for longitude in lire_liste_float(args.longitudes):
                donnees = extraire_colonne(paquet, latitude, longitude, mois=mois)
                surface = donnees["surface"]
                latitude_reelle = float(surface["latitude_deg"])
                longitude_reelle = float(surface["longitude_deg"])
                colonnes.append(
                    ColonneCalibration(
                        identifiant=f"lat{latitude_reelle:+06.1f}_lon{longitude_reelle:+07.1f}_m{mois:02d}",
                        donnees=donnees,
                        poids=max(0.0, math.cos(math.radians(latitude_reelle))),
                        temperature_surface_k=temperature_surface_colonne(donnees),
                    )
                )

    if sum(colonne.poids for colonne in colonnes) <= 0.0:
        raise ValueError("Les poids de colonnes sont nuls.")
    return colonnes


def couche_reference_depuis_modele(couche: dict[str, Any]) -> CoucheReference:
    pression_bas_pa = float(couche["pression_bas_pa"])
    pression_haut_pa = float(couche["pression_haut_pa"])
    delta_p_pa = pression_bas_pa - pression_haut_pa
    temperature_k = float(couche["temperature_k"])
    pression_milieu_pa = 0.5 * (pression_bas_pa + pression_haut_pa)
    if delta_p_pa <= 0.0 or pression_milieu_pa <= 0.0 or temperature_k <= 0.0:
        raise ValueError("Couche invalide pour le calibrage CO2.")

    masse_air_kg_m2 = delta_p_pa / physique.GRAVITE
    densite_air_kg_m3 = pression_milieu_pa * MASSE_MOLAIRE_AIR / (R_UNIVERSEL * temperature_k)
    epaisseur_m = masse_air_kg_m2 / densite_air_kg_m3
    return CoucheReference(
        delta_p_pa=delta_p_pa,
        pression_milieu_bar=pression_milieu_pa / 100_000.0,
        temperature_k=temperature_k,
        chemin_optique_cm=epaisseur_m * 100.0,
    )


def extraire_transmission_spectre(spectre: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        nombre_onde_cm, transmission = spectre.get("transmittance_noslit")
    except Exception:
        nombre_onde_cm, transmission = spectre.get("transmittance")
    return np.asarray(nombre_onde_cm, dtype=float), np.asarray(transmission, dtype=float)


def calculer_spectre_radis_hitran(
    bande: dict[str, Any],
    couche: CoucheReference,
    co2_ppm: float,
    wstep: float | str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from radis import calc_spectrum
    except ImportError as exc:
        raise RuntimeError(
            "RADIS n'est pas installe. Installer avec "
            "`python -m pip install -r modele3/requirements-calibrage.txt`."
        ) from exc

    wmin, wmax = plage_nombre_onde_cm(bande)
    kwargs = {
        "isotope": RADIS_ISOTOPES,
        "pressure": couche.pression_milieu_bar,
        "Tgas": couche.temperature_k,
        "mole_fraction": co2_ppm * 1e-6,
        "path_length": couche.chemin_optique_cm,
        "databank": RADIS_DATABANK,
        "medium": "air",
        "wstep": wstep,
        "truncation": RADIS_TRUNCATION_CM_1,
        "verbose": False,
    }
    try:
        spectre = calc_spectrum(wmin, wmax, molecule="CO2", **kwargs)
    except TypeError as exc:
        if "molecule" not in str(exc):
            raise
        spectre = calc_spectrum(wmin, wmax, species="CO2", **kwargs)
    return extraire_transmission_spectre(spectre)


def mesurer_tau_reference(
    colonnes: list[ColonneCalibration],
    bandes: list[dict[str, Any]],
    co2_values: list[float],
    wstep: float | str,
) -> dict[str, list[tuple[float, float]]]:
    mesures = {bande["nom"]: [] for bande in bandes}

    for colonne in colonnes:
        for couche_modele in colonne.donnees["couches"]:
            couche = couche_reference_depuis_modele(couche_modele)
            for bande in bandes:
                for co2_ppm in co2_values:
                    nombre_onde_cm, transmission = calculer_spectre_radis_hitran(
                        bande,
                        couche,
                        co2_ppm,
                        wstep,
                    )
                    transmission_moyenne = moyenne_planck_transmission(
                        nombre_onde_cm,
                        transmission,
                        couche.temperature_k,
                    )
                    x_modele = (co2_ppm / CO2_REFERENCE_PPM) * (
                        couche.delta_p_pa / PRESSION_REFERENCE_PA
                    )
                    mesures[bande["nom"]].append(
                        (x_modele, tau_equivalent_depuis_transmission(transmission_moyenne))
                    )
    return mesures


def ajuster_coefficients_par_bande(
    mesures: dict[str, list[tuple[float, float]]],
) -> dict[str, float]:
    coefficients: dict[str, float] = {}
    for nom_bande, valeurs in mesures.items():
        x = np.asarray([valeur[0] for valeur in valeurs], dtype=float)
        tau = np.asarray([valeur[1] for valeur in valeurs], dtype=float)
        coefficients[nom_bande] = coefficient_effectif_median(x, tau)
    return coefficients


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
    return numerateur / denominateur


def bandes_avec_coefficients(coefficients: dict[str, float], facteur: float) -> list[dict[str, Any]]:
    return physique.bandes_avec_coefficients_co2(coefficients, facteur=facteur)


def trouver_facteur_forcage(
    colonnes: list[ColonneCalibration],
    coefficients: dict[str, float],
    cible_w_m2: float,
) -> tuple[float, float, float]:
    if cible_w_m2 <= 0.0:
        raise ValueError("La cible de forcage doit etre positive.")

    def forcage(facteur: float) -> float:
        return calculer_forcage_echantillon(
            colonnes,
            bandes_avec_coefficients(coefficients, facteur),
            CO2_REFERENCE_PPM,
            CO2_DOUBLE_PPM,
        )

    forcage_brut = forcage(1.0)
    bas = 0.0
    haut = 1.0
    forcage_haut = forcage_brut
    while forcage_haut < cible_w_m2 and haut < FACTEUR_MAX:
        haut *= 2.0
        forcage_haut = forcage(haut)
    if forcage_haut < cible_w_m2:
        raise RuntimeError("La cible de forcage n'est pas atteinte.")

    for _ in range(ITERATIONS_FORCAGE):
        milieu = 0.5 * (bas + haut)
        if forcage(milieu) < cible_w_m2:
            bas = milieu
        else:
            haut = milieu

    facteur = 0.5 * (bas + haut)
    return facteur, forcage_brut, forcage(facteur)


def arrondir_json(objet: Any, chiffres: int = 10) -> Any:
    if isinstance(objet, float):
        return round(objet, chiffres) if math.isfinite(objet) else None
    if isinstance(objet, dict):
        return {cle: arrondir_json(valeur, chiffres) for cle, valeur in objet.items()}
    if isinstance(objet, list):
        return [arrondir_json(valeur, chiffres) for valeur in objet]
    return objet


def ecrire_snippet_python(chemin: Path, coefficients_finaux: dict[str, float]) -> None:
    lignes = [
        '"""Coefficients CO2 calibres par modele3.codes_python.calibrer_coefficients_co2."""',
        "",
        "COEFFICIENTS_CO2_CALIBRES = {",
    ]
    for nom, valeur in sorted(coefficients_finaux.items()):
        lignes.append(f'    "{nom}": {valeur:.12g},')
    lignes.extend(
        [
            "}",
            "",
            "",
            "def bandes_calibrees():",
            "    from modele3.codes_python.physique import bandes_avec_coefficients_co2",
            "",
            "    return bandes_avec_coefficients_co2(COEFFICIENTS_CO2_CALIBRES)",
            "",
        ]
    )
    chemin.write_text("\n".join(lignes), encoding="utf-8")


def construire_payload_runtime(
    coefficients_finaux: dict[str, float],
    facteur: float,
    cible_forcage: float,
    forcage_final: float,
) -> dict[str, Any]:
    return {
        "methode": (
            "coefficients CO2 effectifs calibres par HITRAN/RADIS, moyenne Planck, "
            "mediane(tau_eq / X), puis facteur global sur le forcage 280 -> 560 ppm"
        ),
        "formule_modele3": "tau_CO2 = a_CO2 * (co2_ppm / 280) * (delta_p_pa / 101325)",
        "co2_reference_ppm": CO2_REFERENCE_PPM,
        "pression_reference_pa": PRESSION_REFERENCE_PA,
        "forcage": {
            "cible_W_m2": cible_forcage,
            "final_W_m2": forcage_final,
            "facteur_global": facteur,
        },
        "coefficients": dict(sorted(coefficients_finaux.items())),
    }


def construire_resultat(
    args: argparse.Namespace,
    colonnes: list[ColonneCalibration],
    bandes: list[dict[str, Any]],
    coefficients: dict[str, float],
    facteur: float,
    forcage_brut: float,
    forcage_final: float,
    nombre_spectres: int,
) -> dict[str, Any]:
    bandes_par_nom = {bande["nom"]: bande for bande in bandes}
    return {
        "methode": (
            "HITRAN/RADIS -> moyenne Planck -> tau equivalent -> "
            "mediane(tau/X) par bande -> facteur global de forcage"
        ),
        "formule_modele3": "tau_CO2 = a_CO2 * (CO2_ppm / 280) * (delta_p / 101325)",
        "echantillon": {
            "latitudes": args.latitudes,
            "longitudes": args.longitudes,
            "mois": args.mois,
            "co2_values": args.co2_values,
            "colonnes": len(colonnes),
            "couches": sum(len(colonne.donnees["couches"]) for colonne in colonnes),
            "spectres_radis_hitran": nombre_spectres,
        },
        "forcage": {
            "cible_W_m2": args.cible_forcage,
            "brut_W_m2": forcage_brut,
            "facteur_global": facteur,
            "final_W_m2": forcage_final,
        },
        "coefficients": [
            {
                "nom": nom,
                "lambda_min_um": float(bandes_par_nom[nom]["lambda_min_um"]),
                "lambda_max_um": float(bandes_par_nom[nom]["lambda_max_um"]),
                "a_co2_hitran": valeur,
                "a_co2_final": valeur * facteur,
            }
            for nom, valeur in sorted(
                coefficients.items(),
                key=lambda item: bandes_par_nom[item[0]]["lambda_min_um"],
            )
        ],
    }


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Calibrer simplement les coefficients CO2 du modele 3 avec HITRAN/RADIS."
    )
    parseur.add_argument("--paquet", type=Path, default=DOSSIER_PAQUET_DEFAUT)
    parseur.add_argument("--latitudes", default="-45,0,45")
    parseur.add_argument("--longitudes", default="0")
    parseur.add_argument("--mois", default="1,7")
    parseur.add_argument("--co2-values", default="280,420,560,1120")
    parseur.add_argument("--wstep", type=lire_wstep, default="auto")
    parseur.add_argument("--cible-forcage", type=float, default=FORCAGE_MYHRE_2XCO2)
    parseur.add_argument(
        "--output-dir",
        type=Path,
        default=CHEMIN_COEFFICIENTS_CO2.parent,
    )
    parseur.add_argument("--dry-run", action="store_true")
    return parseur


def main() -> None:
    args = construire_parseur().parse_args()
    colonnes = charger_colonnes_calibration(args)
    bandes = physique.bandes_co2()
    co2_values = lire_liste_float(args.co2_values)
    nombre_couches = sum(len(colonne.donnees["couches"]) for colonne in colonnes)
    nombre_spectres = nombre_couches * len(bandes) * len(co2_values)

    if args.dry_run:
        print("calibrage_co2_dry_run")
        print(f"colonnes = {len(colonnes)}")
        print(f"couches = {nombre_couches}")
        print(f"bandes_co2 = {len(bandes)}")
        print(f"concentrations = {len(co2_values)}")
        print(f"spectres_radis_hitran = {nombre_spectres}")
        return

    mesures = mesurer_tau_reference(colonnes, bandes, co2_values, args.wstep)
    coefficients = ajuster_coefficients_par_bande(mesures)
    facteur, forcage_brut, forcage_final = trouver_facteur_forcage(
        colonnes,
        coefficients,
        args.cible_forcage,
    )
    coefficients_finaux = {nom: valeur * facteur for nom, valeur in coefficients.items()}
    resultat = construire_resultat(
        args,
        colonnes,
        bandes,
        coefficients,
        facteur,
        forcage_brut,
        forcage_final,
        nombre_spectres,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "calibrage_coefficients_co2.json"
    runtime_path = args.output_dir / CHEMIN_COEFFICIENTS_CO2.name
    snippet_path = args.output_dir / "coefficients_co2_calibres.py"
    json_path.write_text(json.dumps(arrondir_json(resultat), indent=2), encoding="utf-8")
    runtime_path.write_text(
        json.dumps(
            arrondir_json(
                construire_payload_runtime(
                    coefficients_finaux,
                    facteur,
                    args.cible_forcage,
                    forcage_final,
                )
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    ecrire_snippet_python(snippet_path, coefficients_finaux)

    print("calibrage_co2_ok")
    print(f"json = {json_path}")
    print(f"runtime_json = {runtime_path}")
    print(f"snippet = {snippet_path}")
    print(f"forcage_cible_W_m2 = {args.cible_forcage:.6f}")
    print(f"forcage_final_W_m2 = {forcage_final:.6f}")
    print(f"facteur_global = {facteur:.8f}")


if __name__ == "__main__":
    main()
