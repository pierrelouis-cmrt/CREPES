"""Modele 5 temporaire : emission laterale sortante par couche.

Ce module ne remplace pas le modele 3. Il l'utilise comme source de colonnes
verticales puis ajoute un diagnostic d'emission laterale propre a chaque couche.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3 import physique
from modele3.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
from modele3.modele3 import calculer_colonne_radiative, construire_couches


def _arrondir(objet):
    if isinstance(objet, float):
        return round(objet, 6)
    if isinstance(objet, dict):
        return {cle: _arrondir(valeur) for cle, valeur in objet.items()}
    if isinstance(objet, list):
        return [_arrondir(valeur) for valeur in objet]
    return objet


def calculer_emission_laterale_sortante_par_couche(
    donnees_colonne,
    co2_ppm=420.0,
    bandes=None,
):
    """Calcule ce que chaque couche emet lateralement vers ses quatre cotes.

    Le calcul reutilise les memes opacites et les memes bandes infrarouges que
    le modele 3. Pour une couche donnee, les quatre cotes sont identiques :
    on calcule l'emission propre de la couche, pas un echange avec un voisin.
    """

    if bandes is None:
        bandes = physique.BANDES_INFRAROUGES

    couches_preparees = construire_couches(donnees_colonne, co2_ppm=co2_ppm)
    couches = []

    for couche in couches_preparees:
        flux_par_cote = 0.0
        diagnostics_bandes = []

        for bande in bandes:
            diagnostic = physique.opacites_couche_bande(couche, bande)
            flux_corps_noir = physique.flux_corps_noir_dans_bande(
                couche["temperature_k"],
                bande["lambda_min_um"],
                bande["lambda_max_um"],
            )
            flux_bande = diagnostic["emissivite"] * flux_corps_noir
            flux_par_cote += flux_bande
            diagnostics_bandes.append(
                {
                    "bande": bande["nom"],
                    "famille": bande["famille"],
                    "role": bande["role"],
                    "lambda_min_um": bande["lambda_min_um"],
                    "lambda_max_um": bande["lambda_max_um"],
                    "tau_CO2": diagnostic["tau_co2"],
                    "tau_H2O": diagnostic["tau_h2o"],
                    "tau_total": diagnostic["tau_total"],
                    "emissivite": diagnostic["emissivite"],
                    "flux_sortant_lateral_bande_w_m2": flux_bande,
                }
            )

        couches.append(
            {
                "nom": couche["nom"],
                "pression_bas_hpa": couche["pression_bas_hpa"],
                "pression_haut_hpa": couche["pression_haut_hpa"],
                "temperature_k": couche["temperature_k"],
                "masse_air_kg_m2": couche["masse_air_kg_m2"],
                "flux_sortant_lateral_par_cote_w_m2": flux_par_cote,
                "flux_sortant_lateral_4_cotes_w_m2": 4.0 * flux_par_cote,
                "flux_sortant_lateral_par_direction_w_m2": {
                    "nord": flux_par_cote,
                    "sud": flux_par_cote,
                    "est": flux_par_cote,
                    "ouest": flux_par_cote,
                },
                "diagnostics_bandes": diagnostics_bandes,
            }
        )

    return {
        "hypothese": "emission laterale isotrope simplifiee",
        "convention": "flux sortant emis par la couche elle-meme",
        "unite": "W m-2 de face laterale",
        "couches": couches,
    }


def calculer_colonne_avec_emission_laterale(
    paquet,
    lat,
    lon,
    mois=None,
    jour_annee=None,
    temperature_surface_k=288.15,
    co2_ppm=420.0,
    heure_solaire=12.0,
    moyenne_journaliere_sw=False,
):
    colonne = extraire_colonne(paquet, lat, lon, mois=mois, jour_annee=jour_annee)
    radiatif = calculer_colonne_radiative(
        colonne,
        temperature_surface_k=temperature_surface_k,
        co2_ppm=co2_ppm,
        jour_annee=jour_annee,
        heure_solaire=heure_solaire,
        moyenne_journaliere_sw=moyenne_journaliere_sw,
    )
    laterale = calculer_emission_laterale_sortante_par_couche(
        colonne,
        co2_ppm=co2_ppm,
    )
    return {
        "flux_radiatifs_modele3": radiatif,
        "emission_laterale_sortante": laterale,
    }


def construire_parseur():
    parseur = argparse.ArgumentParser(
        description="Modele 5 temporaire - emission laterale sortante par couche"
    )
    parseur.add_argument("--lat", type=float, default=0.0, help="Latitude en degres")
    parseur.add_argument("--lon", type=float, default=0.0, help="Longitude en degres")
    parseur.add_argument("--mois", type=int, default=7, help="Mois 1..12")
    parseur.add_argument("--jour-annee", type=int, default=None, help="Jour 1..365")
    parseur.add_argument("--temperature-surface", type=float, default=288.15)
    parseur.add_argument("--co2", type=float, default=420.0, help="CO2 en ppm")
    parseur.add_argument("--heure-solaire", type=float, default=12.0)
    parseur.add_argument("--moyenne-journaliere-sw", action="store_true")
    parseur.add_argument(
        "--paquet",
        type=Path,
        default=DOSSIER_PAQUET_DEFAUT,
        help="Dossier du paquet compact du modele 3.",
    )
    parseur.add_argument("--json", action="store_true", help="Sortie JSON complete")
    return parseur


def afficher_resume(resultat):
    radiatif = resultat["flux_radiatifs_modele3"]
    laterale = resultat["emission_laterale_sortante"]

    print("modele5_temporaire_emission_laterale")
    print(f"lat_lon = {radiatif['lat_deg']:.4f}, {radiatif['lon_deg']:.4f}")
    print(f"mois = {radiatif['mois']}")
    print(f"hypothese = {laterale['hypothese']}")
    print(f"unite_laterale = {laterale['unite']}")
    print()
    print("flux_radiatifs_modele3_W_m2")
    print(f"SW_absorbe_surface = {radiatif['SW_absorbe_surface']:.6f}")
    print(f"LW_up_surface = {radiatif['LW_up_surface']:.6f}")
    print(f"LW_down_surface = {radiatif['LW_down_surface']:.6f}")
    print(f"OLR = {radiatif['OLR']:.6f}")
    print()
    print("emission_laterale_sortante_par_couche_W_m2")
    for couche in laterale["couches"]:
        print(
            f"{couche['nom']} "
            f"{couche['pression_bas_hpa']:.1f}-{couche['pression_haut_hpa']:.1f} hPa "
            f"T={couche['temperature_k']:.2f} K "
            f"par_cote={couche['flux_sortant_lateral_par_cote_w_m2']:.6f} "
            f"total_4_cotes={couche['flux_sortant_lateral_4_cotes_w_m2']:.6f}"
        )


def main():
    args = construire_parseur().parse_args()
    paquet = charger_paquet_grille(args.paquet)
    resultat = calculer_colonne_avec_emission_laterale(
        paquet,
        args.lat,
        args.lon,
        mois=args.mois,
        jour_annee=args.jour_annee,
        temperature_surface_k=args.temperature_surface,
        co2_ppm=args.co2,
        heure_solaire=args.heure_solaire,
        moyenne_journaliere_sw=args.moyenne_journaliere_sw,
    )

    if args.json:
        print(json.dumps(_arrondir(resultat), indent=2, ensure_ascii=False))
    else:
        afficher_resume(resultat)


if __name__ == "__main__":
    main()
