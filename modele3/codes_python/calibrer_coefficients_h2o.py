"""Calibrage minimal des coefficients H2O du modele 3.

Le but est de compresser un calcul HITRAN/RADIS en coefficients effectifs
compatibles avec le modele 3 :

    tau_H2O = a_h2o * (masse_h2o_kg_m2 / 10)

Pour chaque bande H2O, on calcule une transmission RADIS pour les couches ERA5
humides, on la moyenne avec le poids de Planck de la couche, on la convertit en
tau equivalent, puis on retient la mediane de tau/X. Contrairement au CO2, il
n'y a pas de recalage global sur une cible de forcage : la vapeur d'eau varie
fortement avec les profils locaux et ne fournit pas une contrainte universelle
aussi propre que le doublement 280 -> 560 ppm.
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
    from .calibrer_coefficients_co2 import (
        MASSE_MOLAIRE_AIR,
        R_UNIVERSEL,
        TRANSMISSION_MIN,
        arrondir_json,
        charger_colonnes_calibration,
        coefficient_effectif_median,
        extraire_transmission_spectre,
        lire_liste_float,
        lire_liste_int,
        lire_wstep,
        moyenne_planck_transmission,
        plage_nombre_onde_cm,
        tau_equivalent_depuis_transmission,
    )
    from .donnees import DOSSIER_PAQUET_DEFAUT
except ImportError:  # Permet aussi : python modele3/codes_python/calibrer_coefficients_h2o.py
    dossier_script = str(Path(__file__).resolve().parent)
    sys.path = [chemin for chemin in sys.path if chemin != dossier_script]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique
    from modele3.codes_python.calibrer_coefficients_co2 import (
        MASSE_MOLAIRE_AIR,
        R_UNIVERSEL,
        TRANSMISSION_MIN,
        arrondir_json,
        charger_colonnes_calibration,
        coefficient_effectif_median,
        extraire_transmission_spectre,
        lire_liste_float,
        lire_liste_int,
        lire_wstep,
        moyenne_planck_transmission,
        plage_nombre_onde_cm,
        tau_equivalent_depuis_transmission,
    )
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT


MASSE_MOLAIRE_H2O = 0.01801528
MASSE_H2O_REFERENCE_KG_M2 = physique.MASSE_H2O_REFERENCE_KG_M2
H2O_MASSE_MIN_KG_M2 = 1e-8

RADIS_DATABANK = ("hitran", "range")
RADIS_ISOTOPES_H2O = "1,2,3"
RADIS_TRUNCATION_CM_1 = 50.0


@dataclass(frozen=True)
class CoucheH2OReference:
    delta_p_pa: float
    masse_air_kg_m2: float
    masse_h2o_kg_m2: float
    pression_milieu_bar: float
    temperature_k: float
    chemin_optique_cm: float
    fraction_molaire_h2o: float


def fraction_molaire_h2o_depuis_masses(
    masse_h2o_kg_m2: float,
    masse_air_kg_m2: float,
) -> float:
    """Convertit une masse H2O colonne en fraction molaire homogene."""

    masse_h2o = float(masse_h2o_kg_m2)
    masse_air = float(masse_air_kg_m2)
    if masse_h2o <= 0.0 or masse_air <= 0.0:
        return 0.0
    if masse_h2o >= masse_air:
        raise ValueError("La masse H2O doit rester inferieure a la masse d'air.")

    q = masse_h2o / masse_air
    moles_h2o_par_kg = q / MASSE_MOLAIRE_H2O
    moles_air_sec_par_kg = (1.0 - q) / MASSE_MOLAIRE_AIR
    total = moles_h2o_par_kg + moles_air_sec_par_kg
    if total <= 0.0:
        return 0.0
    return moles_h2o_par_kg / total


def couche_reference_depuis_modele(
    couche: dict[str, Any],
    echelle_h2o: float = 1.0,
) -> CoucheH2OReference:
    pression_bas_pa = float(couche["pression_bas_pa"])
    pression_haut_pa = float(couche["pression_haut_pa"])
    delta_p_pa = pression_bas_pa - pression_haut_pa
    pression_milieu_pa = 0.5 * (pression_bas_pa + pression_haut_pa)
    temperature_k = float(couche["temperature_k"])
    if delta_p_pa <= 0.0 or pression_milieu_pa <= 0.0 or temperature_k <= 0.0:
        raise ValueError("Couche invalide pour le calibrage H2O.")

    masse_air_kg_m2 = physique.valeur_finie(couche.get("masse_air_kg_m2"))
    if masse_air_kg_m2 is None or masse_air_kg_m2 <= 0.0:
        masse_air_kg_m2 = delta_p_pa / physique.GRAVITE

    masse_h2o_source = physique.valeur_finie(couche.get("masse_h2o_kg_m2"), 0.0)
    masse_h2o_kg_m2 = max(0.0, float(masse_h2o_source) * float(echelle_h2o))
    fraction_molaire_h2o = fraction_molaire_h2o_depuis_masses(
        masse_h2o_kg_m2,
        masse_air_kg_m2,
    )

    densite_air_kg_m3 = pression_milieu_pa * MASSE_MOLAIRE_AIR / (
        R_UNIVERSEL * temperature_k
    )
    epaisseur_m = masse_air_kg_m2 / densite_air_kg_m3
    return CoucheH2OReference(
        delta_p_pa=delta_p_pa,
        masse_air_kg_m2=masse_air_kg_m2,
        masse_h2o_kg_m2=masse_h2o_kg_m2,
        pression_milieu_bar=pression_milieu_pa / 100_000.0,
        temperature_k=temperature_k,
        chemin_optique_cm=epaisseur_m * 100.0,
        fraction_molaire_h2o=fraction_molaire_h2o,
    )


def calculer_spectre_radis_hitran(
    bande: dict[str, Any],
    couche: CoucheH2OReference,
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
        "isotope": RADIS_ISOTOPES_H2O,
        "pressure": couche.pression_milieu_bar,
        "Tgas": couche.temperature_k,
        "mole_fraction": couche.fraction_molaire_h2o,
        "path_length": couche.chemin_optique_cm,
        "databank": RADIS_DATABANK,
        "medium": "air",
        "wstep": wstep,
        "truncation": RADIS_TRUNCATION_CM_1,
        "verbose": False,
    }
    try:
        spectre = calc_spectrum(wmin, wmax, molecule="H2O", **kwargs)
    except TypeError as exc:
        if "molecule" not in str(exc):
            raise
        spectre = calc_spectrum(wmin, wmax, species="H2O", **kwargs)
    return extraire_transmission_spectre(spectre)


def couche_humide(couche: CoucheH2OReference) -> bool:
    return (
        couche.masse_h2o_kg_m2 > H2O_MASSE_MIN_KG_M2
        and couche.fraction_molaire_h2o > 0.0
    )


def compter_couches_humides(colonnes: list[Any], echelles_h2o: list[float]) -> int:
    total = 0
    for colonne in colonnes:
        for couche_modele in colonne.donnees["couches"]:
            for echelle in echelles_h2o:
                if couche_humide(couche_reference_depuis_modele(couche_modele, echelle)):
                    total += 1
    return total


def mesurer_tau_reference(
    colonnes: list[Any],
    bandes: list[dict[str, Any]],
    echelles_h2o: list[float],
    wstep: float | str,
) -> dict[str, list[tuple[float, float]]]:
    mesures = {bande["nom"]: [] for bande in bandes}

    for colonne in colonnes:
        for couche_modele in colonne.donnees["couches"]:
            for echelle in echelles_h2o:
                couche = couche_reference_depuis_modele(couche_modele, echelle)
                if not couche_humide(couche):
                    continue
                x_modele = couche.masse_h2o_kg_m2 / MASSE_H2O_REFERENCE_KG_M2
                for bande in bandes:
                    nombre_onde_cm, transmission = calculer_spectre_radis_hitran(
                        bande,
                        couche,
                        wstep,
                    )
                    transmission_moyenne = moyenne_planck_transmission(
                        nombre_onde_cm,
                        transmission,
                        couche.temperature_k,
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


def ecrire_snippet_python(chemin: Path, coefficients: dict[str, float]) -> None:
    lignes = [
        '"""Coefficients H2O calibres par modele3.codes_python.calibrer_coefficients_h2o."""',
        "",
        "COEFFICIENTS_H2O_CALIBRES = {",
    ]
    for nom, valeur in sorted(coefficients.items()):
        lignes.append(f'    "{nom}": {valeur:.12g},')
    lignes.extend(
        [
            "}",
            "",
            "",
            "def bandes_calibrees():",
            "    from modele3.codes_python.physique import bandes_avec_coefficients_h2o",
            "",
            "    return bandes_avec_coefficients_h2o(COEFFICIENTS_H2O_CALIBRES)",
            "",
        ]
    )
    chemin.write_text("\n".join(lignes), encoding="utf-8")


def construire_resultat(
    args: argparse.Namespace,
    colonnes: list[Any],
    bandes: list[dict[str, Any]],
    coefficients: dict[str, float],
    nombre_couches_humides: int,
    nombre_spectres: int,
) -> dict[str, Any]:
    bandes_par_nom = {bande["nom"]: bande for bande in bandes}
    return {
        "methode": (
            "HITRAN/RADIS H2O -> moyenne Planck -> tau equivalent -> "
            "mediane(tau/X) par bande, sans recalage global de forcage"
        ),
        "formule_modele3": "tau_H2O = a_H2O * (masse_h2o_kg_m2 / 10)",
        "echantillon": {
            "latitudes": args.latitudes,
            "longitudes": args.longitudes,
            "mois": args.mois,
            "h2o_scale_values": args.h2o_scale_values,
            "colonnes": len(colonnes),
            "couches": sum(len(colonne.donnees["couches"]) for colonne in colonnes),
            "couches_humides_echantillonnees": nombre_couches_humides,
            "spectres_radis_hitran": nombre_spectres,
        },
        "normalisation": {
            "masse_h2o_reference_kg_m2": MASSE_H2O_REFERENCE_KG_M2,
            "mole_fraction": "deduite de masse_h2o_kg_m2 / masse_air_kg_m2",
            "transmission_min": TRANSMISSION_MIN,
        },
        "coefficients": [
            {
                "nom": nom,
                "lambda_min_um": float(bandes_par_nom[nom]["lambda_min_um"]),
                "lambda_max_um": float(bandes_par_nom[nom]["lambda_max_um"]),
                "a_h2o_actuel": float(bandes_par_nom[nom]["a_h2o"]),
                "a_h2o_hitran": valeur,
            }
            for nom, valeur in sorted(
                coefficients.items(),
                key=lambda item: bandes_par_nom[item[0]]["lambda_min_um"],
            )
        ],
        "limites": [
            "coefficients effectifs par grandes bandes, pas correlated-k",
            "couches supposees homogenes",
            "pas de recalage sur les flux ERA5 car les nuages et le profil thermique s'y melangent",
        ],
    }


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(
        description="Calibrer simplement les coefficients H2O du modele 3 avec HITRAN/RADIS."
    )
    parseur.add_argument("--paquet", type=Path, default=DOSSIER_PAQUET_DEFAUT)
    parseur.add_argument("--latitudes", default="-45,0,45")
    parseur.add_argument("--longitudes", default="0")
    parseur.add_argument("--mois", default="1,7")
    parseur.add_argument(
        "--h2o-scale-values",
        default="1",
        help=(
            "Echelles appliquees aux masses H2O ERA5. Garder 1 par defaut pour "
            "un calibrage strictement ancre dans les profils fournis."
        ),
    )
    parseur.add_argument("--wstep", type=lire_wstep, default="auto")
    parseur.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "ressources"
        / "calibrage_opacite_h2o",
    )
    parseur.add_argument("--dry-run", action="store_true")
    return parseur


def main() -> None:
    args = construire_parseur().parse_args()
    colonnes = charger_colonnes_calibration(args)
    bandes = physique.bandes_h2o()
    echelles_h2o = lire_liste_float(args.h2o_scale_values)
    if any(echelle <= 0.0 for echelle in echelles_h2o):
        raise ValueError("--h2o-scale-values doit contenir des valeurs strictement positives.")

    nombre_couches = sum(len(colonne.donnees["couches"]) for colonne in colonnes)
    nombre_couches_humides = compter_couches_humides(colonnes, echelles_h2o)
    nombre_spectres = nombre_couches_humides * len(bandes)

    if args.dry_run:
        print("calibrage_h2o_dry_run")
        print(f"colonnes = {len(colonnes)}")
        print(f"couches = {nombre_couches}")
        print(f"couches_humides_echantillonnees = {nombre_couches_humides}")
        print(f"bandes_h2o = {len(bandes)}")
        print(f"echelles_h2o = {len(echelles_h2o)}")
        print(f"spectres_radis_hitran = {nombre_spectres}")
        return

    mesures = mesurer_tau_reference(colonnes, bandes, echelles_h2o, args.wstep)
    coefficients = ajuster_coefficients_par_bande(mesures)
    resultat = construire_resultat(
        args,
        colonnes,
        bandes,
        coefficients,
        nombre_couches_humides,
        nombre_spectres,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "calibrage_coefficients_h2o.json"
    snippet_path = args.output_dir / "coefficients_h2o_calibres.py"
    json_path.write_text(json.dumps(arrondir_json(resultat), indent=2), encoding="utf-8")
    ecrire_snippet_python(snippet_path, coefficients)

    print("calibrage_h2o_ok")
    print(f"json = {json_path}")
    print(f"snippet = {snippet_path}")
    print(f"bandes_h2o = {len(bandes)}")
    print(f"spectres_radis_hitran = {nombre_spectres}")


if __name__ == "__main__":
    main()
