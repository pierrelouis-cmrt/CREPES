"""Calibre l'echelle des coefficients optiques du modele 2.5.

La cible retenue est le forçage effectif AR6 pour un doublement du CO2
(``F_2xCO2 = 3.93 W m-2``). Ici on l'utilise comme contrainte numerique pour les
coefficients optiques effectifs, sans ajouter de formule de forçage au modele.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


MODELE_DIR = Path(__file__).resolve().parents[1]
if str(MODELE_DIR) not in sys.path:
    sys.path.insert(0, str(MODELE_DIR))

from modele2_5 import (
    BandeSpectrale,
    calculer_forcage_doublement_co2,
    construire_bandes_co2,
)


def filtrer_bandes(
    bandes: tuple[BandeSpectrale, ...],
    famille: str | None = None,
    role_contient: str | None = None,
) -> tuple[BandeSpectrale, ...]:
    """Selectionne des bandes pour le diagnostic de contribution."""

    resultat = []
    for bande in bandes:
        if famille is not None and bande.bande_majeure != famille:
            continue
        if role_contient is not None and role_contient not in bande.role:
            continue
        resultat.append(bande)
    return tuple(resultat)


def forcage_pour_echelle(echelle: float) -> float:
    """Calcule le forçage 280 -> 560 ppm pour une echelle donnee."""

    return calculer_forcage_doublement_co2(bandes=construire_bandes_co2(echelle))


def calibrer_echelle(cible_w_m2: float, tolerance_w_m2: float) -> tuple[float, float]:
    """Trouve l'echelle donnant le forçage cible par dichotomie."""

    bas = 0.0
    haut = 1.0
    forcage_haut = forcage_pour_echelle(haut)
    # On agrandit d'abord l'intervalle jusqu'a encadrer la cible.
    while forcage_haut < cible_w_m2:
        haut *= 2.0
        forcage_haut = forcage_pour_echelle(haut)
        if haut > 1_000_000.0:
            raise RuntimeError("Impossible d'atteindre la cible de calibration.")

    meilleur = haut
    meilleur_forcage = forcage_haut
    # La dichotomie ajuste ensuite seulement le facteur multiplicatif.
    for _ in range(80):
        milieu = 0.5 * (bas + haut)
        forcage = forcage_pour_echelle(milieu)
        meilleur = milieu
        meilleur_forcage = forcage
        if abs(forcage - cible_w_m2) <= tolerance_w_m2:
            break
        if forcage < cible_w_m2:
            bas = milieu
        else:
            haut = milieu

    return meilleur, meilleur_forcage


def analyser_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibre les coefficients optiques effectifs du modele 2.5."
    )
    parser.add_argument(
        "--cible-w-m2",
        type=float,
        default=3.93,
        help="cible de forçage 280->560 ppm en W/m2",
    )
    parser.add_argument(
        "--tolerance-w-m2",
        type=float,
        default=1e-7,
        help="tolerance de calibration en W/m2",
    )
    return parser.parse_args()


def main() -> None:
    args = analyser_arguments()
    echelle, forcage = calibrer_echelle(args.cible_w_m2, args.tolerance_w_m2)
    bandes = construire_bandes_co2(echelle)

    print("calibration_coefficients_optiques")
    print(f"cible_W_m2 = {args.cible_w_m2:.6f}")
    print(f"echelle_opacite_co2 = {echelle:.10f}")
    print(f"forcage_280_560_ppm_W_m2 = {forcage:.6f}")
    print()
    print("contributions_diagnostic")
    print("selection, forcage_280_560_ppm_W_m2")
    # Ces selections montrent quelles familles de bandes portent le forcage.
    selections = {
        "toutes_sous_bandes": bandes,
        "15_um_total": filtrer_bandes(bandes, famille="15 um"),
        "15_um_coeur_sature": filtrer_bandes(
            bandes,
            famille="15 um",
            role_contient="coeur",
        ),
        "15_um_ailes": filtrer_bandes(
            bandes,
            famille="15 um",
            role_contient="aile",
        ),
        "4_3_um_total": filtrer_bandes(bandes, famille="4.3 um"),
    }
    for nom, selection in selections.items():
        if selection:
            forcage_selection = calculer_forcage_doublement_co2(bandes=selection)
        else:
            forcage_selection = 0.0
        print(f"{nom}, {forcage_selection:.6f}")


if __name__ == "__main__":
    main()
