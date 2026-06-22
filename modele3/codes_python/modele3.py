"""Modele 3 : colonne radiative locale.

La fonction publique `calculer_colonne_radiative` recoit une colonne deja
preparee. Elle ne lit pas les fichiers lourds et peut donc etre appelee en
boucle par le modele 4.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from . import physique
    from .donnees import (
        DOSSIER_PAQUET_DEFAUT,
        charger_paquet_grille,
        extraire_colonne,
    )
except ImportError:  # Permet aussi : python modele3/codes_python/modele3.py
    dossier_script = Path(__file__).resolve().parent
    sys.path = [
        chemin
        for chemin in sys.path
        if Path(chemin or ".").resolve() != dossier_script
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique
    from modele3.codes_python.donnees import (
        DOSSIER_PAQUET_DEFAUT,
        charger_paquet_grille,
        extraire_colonne,
    )


TEMPERATURE_SURFACE_DEFAUT_K = physique.TEMPERATURE_SURFACE_DEFAUT_K
CO2_DEFAUT_PPM = physique.CO2_DEFAUT_PPM


def _arrondir(objet):
    if isinstance(objet, float):
        return round(objet, 6)
    if isinstance(objet, dict):
        return {cle: _arrondir(valeur) for cle, valeur in objet.items()}
    if isinstance(objet, list):
        return [_arrondir(valeur) for valeur in objet]
    return objet


def construire_couches(donnees, co2_ppm=CO2_DEFAUT_PPM):
    if not donnees.get("couches"):
        raise ValueError("La colonne doit contenir des couches pretraitees.")

    couches = []
    for indice, couche in enumerate(donnees["couches"], start=1):
        copie = dict(couche)
        copie.setdefault("nom", f"couche_{indice:02d}")
        copie["co2_ppm"] = co2_ppm
        if "pression_bas_pa" not in copie:
            copie["pression_bas_pa"] = copie["pression_bas_hpa"] * 100.0
        if "pression_haut_pa" not in copie:
            copie["pression_haut_pa"] = copie["pression_haut_hpa"] * 100.0
        p_bas = physique.valeur_finie(copie.get("pression_bas_pa"))
        p_haut = physique.valeur_finie(copie.get("pression_haut_pa"))
        if p_bas is None or p_haut is None or p_bas <= p_haut:
            raise ValueError(
                f"Couche {copie['nom']} invalide: "
                f"pression_bas_pa={copie.get('pression_bas_pa')}, "
                f"pression_haut_pa={copie.get('pression_haut_pa')}"
            )
        copie["pression_bas_pa"] = p_bas
        copie["pression_haut_pa"] = p_haut
        if "masse_air_kg_m2" not in copie:
            delta_p = p_bas - p_haut
            copie["masse_air_kg_m2"] = physique.masse_air_depuis_delta_p(delta_p)
        masse_air = physique.valeur_finie(copie.get("masse_air_kg_m2"))
        if masse_air is None or masse_air <= 0.0:
            raise ValueError(f"Couche {copie['nom']} invalide: masse_air_kg_m2={masse_air}")
        copie["masse_air_kg_m2"] = masse_air
        humidite = physique.valeur_finie(copie.get("humidite_specifique_kgkg"))
        if humidite is None or humidite < 0.0:
            raise ValueError(
                f"Couche {copie['nom']} invalide: "
                f"humidite_specifique_kgkg={copie.get('humidite_specifique_kgkg')}"
            )
        copie["humidite_specifique_kgkg"] = humidite
        if "masse_h2o_kg_m2" not in copie:
            copie["masse_h2o_kg_m2"] = physique.masse_h2o_colonne(
                humidite,
                masse_air,
            )
        masse_h2o = physique.valeur_finie(copie.get("masse_h2o_kg_m2"))
        if masse_h2o is None or masse_h2o < 0.0:
            raise ValueError(f"Couche {copie['nom']} invalide: masse_h2o_kg_m2={masse_h2o}")
        temperature = physique.valeur_finie(copie.get("temperature_k"))
        if temperature is None:
            raise ValueError(
                f"Couche {copie['nom']} invalide: temperature_k={copie.get('temperature_k')}"
            )
        copie["masse_h2o_kg_m2"] = masse_h2o
        copie["temperature_k"] = temperature
        couches.append(copie)
    return couches


def _propager_flux_montant(flux_surface_bande, bande, couches):
    flux = flux_surface_bande
    for couche in couches:
        diagnostic = physique.opacites_couche_bande(couche, bande)
        emission_couche = physique.flux_corps_noir_dans_bande(
            couche["temperature_k"],
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        flux = diagnostic["transmission"] * flux + diagnostic["emissivite"] * emission_couche
    return flux


def _propager_flux_descendant(bande, couches):
    flux = 0.0
    for couche in reversed(couches):
        diagnostic = physique.opacites_couche_bande(couche, bande)
        emission_couche = physique.flux_corps_noir_dans_bande(
            couche["temperature_k"],
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        flux = diagnostic["transmission"] * flux + diagnostic["emissivite"] * emission_couche
    return flux


def construire_flux_validation_modele(resultat):
    flux_validation = {}
    sw_toa_mensuel = physique.valeur_finie(resultat.get("sw_toa_moyen_mensuel_w_m2"))
    if sw_toa_mensuel is not None:
        sw_down_mensuel = sw_toa_mensuel * resultat["transmissivite_sw_mensuelle"]
        flux_validation["SW_down_surface_mensuel"] = sw_down_mensuel
        flux_validation["SW_absorbe_surface_mensuel"] = (
            sw_down_mensuel * (1.0 - resultat["albedo_surface"])
        )
    return flux_validation


def construire_notes_validation(resultat, validation_flux, flux_validation_modele):
    notes = []
    references_sw = any(
        nom in validation_flux
        for nom in ("era5_sw_down_surface_w_m2", "era5_sw_net_surface_w_m2")
    )
    if references_sw and not flux_validation_modele:
        notes.append(
            "Comparaison shortwave ignoree: les references ERA5 sont mensuelles "
            "et aucun SW_TOA moyen mensuel n'est disponible."
        )
    elif references_sw and resultat.get("mode_shortwave") != "moyenne_mensuelle_paquet":
        notes.append(
            "Le court-onde de flux_W_m2 est en mode "
            f"{resultat.get('mode_shortwave')}; les ecarts SW utilisent le "
            "diagnostic mensuel equivalent."
        )
    return notes


def comparaison_validation(resultat, validation_flux, flux_validation_modele=None):
    if flux_validation_modele is None:
        flux_validation_modele = {}
    comparaison = {}
    if "era5_lw_down_surface_w_m2" in validation_flux:
        comparaison["ecart_LW_down_surface_W_m2"] = (
            resultat["LW_down_surface"] - validation_flux["era5_lw_down_surface_w_m2"]
        )
    if (
        "era5_sw_down_surface_w_m2" in validation_flux
        and "SW_down_surface_mensuel" in flux_validation_modele
    ):
        comparaison["ecart_SW_down_surface_mensuel_W_m2"] = (
            flux_validation_modele["SW_down_surface_mensuel"]
            - validation_flux["era5_sw_down_surface_w_m2"]
        )
    if (
        "era5_sw_net_surface_w_m2" in validation_flux
        and "SW_absorbe_surface_mensuel" in flux_validation_modele
    ):
        comparaison["ecart_SW_absorbe_surface_mensuel_W_m2"] = (
            flux_validation_modele["SW_absorbe_surface_mensuel"]
            - validation_flux["era5_sw_net_surface_w_m2"]
        )
    if "era5_olr_w_m2" in validation_flux:
        comparaison["ecart_OLR_W_m2"] = resultat["OLR"] - validation_flux["era5_olr_w_m2"]
    return comparaison


def calculer_flux_shortwave(
    surface,
    jour_annee,
    heure_solaire,
    moyenne_journaliere_sw,
    utiliser_moyenne_mensuelle=False,
):
    sw_toa_mensuel = physique.valeur_finie(surface.get("sw_toa_moyen_mensuel_w_m2"))
    if moyenne_journaliere_sw:
        if utiliser_moyenne_mensuelle and sw_toa_mensuel is not None:
            sw_toa_local = sw_toa_mensuel
            mode_shortwave = "moyenne_mensuelle_paquet"
        else:
            sw_toa_local = physique.flux_solaire_moyen_journalier(
                surface["latitude_deg"],
                jour_annee,
            )
            mode_shortwave = "moyenne_journaliere_jour_representatif"
    else:
        sw_toa_local = physique.flux_solaire_incident(
            surface["latitude_deg"],
            jour_annee,
            heure_solaire,
        )
        mode_shortwave = "instantane_jour_representatif"

    albedo_surface = physique.albedo_surface_corrige_neige_glace(
        surface.get("albedo_surface"),
        surface.get("snow_ice_fraction"),
    )
    transmissivite_sw = physique.fraction(surface["transmissivite_sw_mensuelle"], defaut=0.0)
    # Le flux shortwave descend jusqu'a la surface puis l'albedo en renvoie une partie.
    sw_down_surface = sw_toa_local * transmissivite_sw
    sw_absorbe_surface = sw_down_surface * (1.0 - albedo_surface)

    return {
        "SW_TOA_local": sw_toa_local,
        "SW_down_surface": sw_down_surface,
        "SW_absorbe_surface": sw_absorbe_surface,
        "albedo_surface": albedo_surface,
        "transmissivite_sw_mensuelle": transmissivite_sw,
        "mode_shortwave": mode_shortwave,
    }


def calculer_colonne_radiative(
    donnees,
    temperature_surface_k=TEMPERATURE_SURFACE_DEFAUT_K,
    co2_ppm=CO2_DEFAUT_PPM,
    jour_annee=None,
    heure_solaire=12.0,
    moyenne_journaliere_sw=False,
    bandes=None,
):
    if bandes is None:
        bandes = physique.BANDES_INFRAROUGES

    surface = donnees["surface"]
    jour_annee_argument = jour_annee
    if jour_annee is None:
        jour_annee = surface.get("jour_annee") or physique.jour_milieu_mois(surface["mois"])
    jour_representatif_mois = jour_annee_argument is None and surface.get("jour_annee") is None

    couches = construire_couches(donnees, co2_ppm=co2_ppm)
    emissivite_surface = physique.EMISSIVITE_SURFACE_CONSTANTE
    flux_surface_total = physique.flux_lw_surface(temperature_surface_k, emissivite_surface)

    flux_surface_bandes = 0.0
    flux_sommet_bandes = 0.0
    lw_down_surface = 0.0
    diagnostics_bandes = []

    for bande in bandes:
        # Chaque bande longwave est propagee vers le sommet puis vers la surface.
        flux_surface_bande = emissivite_surface * physique.flux_corps_noir_dans_bande(
            temperature_surface_k,
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        flux_sommet = _propager_flux_montant(flux_surface_bande, bande, couches)
        flux_descendant = _propager_flux_descendant(bande, couches)

        flux_surface_bandes += flux_surface_bande
        flux_sommet_bandes += flux_sommet
        lw_down_surface += flux_descendant

        tau_co2_total = 0.0
        tau_h2o_total = 0.0
        # Ces sommes servent surtout a lire rapidement ce qui absorbe dans la bande.
        for couche in couches:
            diagnostic = physique.opacites_couche_bande(couche, bande)
            tau_co2_total += diagnostic["tau_co2"]
            tau_h2o_total += diagnostic["tau_h2o"]

        diagnostics_bandes.append(
            {
                "bande": bande["nom"],
                "famille": bande["famille"],
                "role": bande["role"],
                "lambda_min_um": bande["lambda_min_um"],
                "lambda_max_um": bande["lambda_max_um"],
                "tau_CO2_total": tau_co2_total,
                "tau_H2O_total": tau_h2o_total,
                "flux_surface_W_m2": flux_surface_bande,
                "flux_sommet_W_m2": flux_sommet,
                "flux_descendant_surface_W_m2": flux_descendant,
            }
        )

    olr = max(0.0, flux_surface_total - flux_surface_bandes) + flux_sommet_bandes
    lw_down_absorbe_surface = emissivite_surface * lw_down_surface

    shortwave = calculer_flux_shortwave(
        surface,
        jour_annee,
        heure_solaire,
        moyenne_journaliere_sw,
        utiliser_moyenne_mensuelle=jour_representatif_mois,
    )
    sw_absorbe_surface = shortwave["SW_absorbe_surface"]

    resultat = {
        "lat_deg": surface["latitude_deg"],
        "lon_deg": surface["longitude_deg"],
        "mois": surface["mois"],
        "jour_annee": jour_annee,
        "heure_solaire": heure_solaire,
        "moyenne_journaliere_sw": moyenne_journaliere_sw,
        "mode_shortwave": shortwave["mode_shortwave"],
        "temperature_surface_k": temperature_surface_k,
        "co2_ppm": co2_ppm,
        "SW_TOA_local": shortwave["SW_TOA_local"],
        "SW_down_surface": shortwave["SW_down_surface"],
        "SW_absorbe_surface": sw_absorbe_surface,
        "LW_up_surface": flux_surface_total,
        "LW_down_surface": lw_down_surface,
        "LW_down_absorbe_surface": lw_down_absorbe_surface,
        "OLR": olr,
        "flux_net_radiatif_surface": (
            sw_absorbe_surface + lw_down_absorbe_surface - flux_surface_total
        ),
        "albedo_surface": shortwave["albedo_surface"],
        "sw_toa_moyen_mensuel_w_m2": surface.get("sw_toa_moyen_mensuel_w_m2"),
        "transmissivite_sw_mensuelle": shortwave["transmissivite_sw_mensuelle"],
        "emissivite_surface": emissivite_surface,
        "sources": {
            "albedo_surface": surface.get("source_albedo_surface", "inconnue"),
            "transmissivite_sw_mensuelle": surface.get(
                "source_transmissivite_sw_mensuelle",
                "inconnue",
            ),
            "emissivite_surface": "constante_0.98",
        },
        "diagnostics_bandes": diagnostics_bandes,
        "validation_flux": donnees.get("validation_flux", {}),
    }
    resultat["flux_validation_modele"] = construire_flux_validation_modele(resultat)
    resultat["comparaison_validation"] = comparaison_validation(
        resultat,
        resultat["validation_flux"],
        resultat["flux_validation_modele"],
    )
    resultat["notes_validation"] = construire_notes_validation(
        resultat,
        resultat["validation_flux"],
        resultat["flux_validation_modele"],
    )
    return resultat


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Modele 3 - colonne radiative locale")
    parseur.add_argument("--lat", type=float, default=0.0, help="Latitude en degres")
    parseur.add_argument("--lon", type=float, default=0.0, help="Longitude en degres")
    parseur.add_argument("--mois", type=int, default=7, help="Mois 1..12")
    parseur.add_argument("--jour-annee", type=int, default=None, help="Jour 1..365")
    parseur.add_argument("--heure-solaire", type=float, default=12.0, help="Heure solaire locale")
    parseur.add_argument(
        "--moyenne-journaliere-sw",
        action="store_true",
        help="Moyenne la formule solaire instantanee sur 24 h",
    )
    parseur.add_argument(
        "--temperature-surface",
        type=float,
        default=TEMPERATURE_SURFACE_DEFAUT_K,
        help="Temperature de surface imposee en K",
    )
    parseur.add_argument("--co2", type=float, default=CO2_DEFAUT_PPM, help="CO2 en ppm")
    parseur.add_argument(
        "--paquet",
        type=Path,
        default=DOSSIER_PAQUET_DEFAUT,
        help="Dossier du paquet .npz compact",
    )
    parseur.add_argument("--json", action="store_true", help="Sortie JSON complete")
    return parseur


def afficher_resultat(donnees, resultat):
    surface = donnees["surface"]
    print("modele3_colonne_radiative_locale")
    print(f"source_donnees = {donnees.get('source', 'inconnue')}")
    print(f"lat_lon = {resultat['lat_deg']:.4f}, {resultat['lon_deg']:.4f}")
    print(f"mois = {resultat['mois']}")
    print(f"jour_annee = {resultat['jour_annee']}")
    print(f"heure_solaire = {resultat['heure_solaire']:.2f}")
    print(f"moyenne_journaliere_sw = {resultat['moyenne_journaliere_sw']}")
    print(f"T_surface_K = {resultat['temperature_surface_k']:.3f}")
    print(f"CO2_ppm = {resultat['co2_ppm']:.3f}")
    print(f"p_surface_hPa = {surface['pression_surface_pa'] / 100.0:.3f}")
    print(f"albedo_surface = {resultat['albedo_surface']:.4f}")
    print(f"transmissivite_sw_mensuelle = {resultat['transmissivite_sw_mensuelle']:.4f}")
    print(f"emissivite_surface = {resultat['emissivite_surface']:.4f}")
    print()
    print("flux_W_m2")
    print(f"SW_TOA_local = {resultat['SW_TOA_local']:.6f}")
    print(f"SW_down_surface = {resultat['SW_down_surface']:.6f}")
    print(f"SW_absorbe_surface = {resultat['SW_absorbe_surface']:.6f}")
    print(f"LW_up_surface = {resultat['LW_up_surface']:.6f}")
    print(f"LW_down_surface = {resultat['LW_down_surface']:.6f}")
    print(f"LW_down_absorbe_surface = {resultat['LW_down_absorbe_surface']:.6f}")
    print(f"OLR = {resultat['OLR']:.6f}")
    print(f"flux_net_radiatif_surface = {resultat['flux_net_radiatif_surface']:.6f}")

    if resultat["validation_flux"]:
        print()
        print("validation_era5_mensuelle_W_m2")
        for nom, valeur in sorted(resultat["validation_flux"].items()):
            print(f"{nom} = {valeur:.6f}")
        if resultat["flux_validation_modele"]:
            print()
            print("validation_modele_mensuelle_W_m2")
            for nom, valeur in sorted(resultat["flux_validation_modele"].items()):
                print(f"{nom} = {valeur:.6f}")
        if resultat["comparaison_validation"]:
            print()
            print("comparaison_validation_W_m2")
            for nom, valeur in sorted(resultat["comparaison_validation"].items()):
                print(f"{nom} = {valeur:.6f}")
        if resultat["notes_validation"]:
            print()
            print("notes_validation")
            for note in resultat["notes_validation"]:
                print(f"- {note}")


def main():
    args = construire_parseur().parse_args()
    paquet = charger_paquet_grille(args.paquet)
    donnees = extraire_colonne(
        paquet,
        args.lat,
        args.lon,
        mois=args.mois,
        jour_annee=args.jour_annee,
    )

    resultat = calculer_colonne_radiative(
        donnees,
        temperature_surface_k=args.temperature_surface,
        co2_ppm=args.co2,
        jour_annee=args.jour_annee,
        heure_solaire=args.heure_solaire,
        moyenne_journaliere_sw=args.moyenne_journaliere_sw,
    )

    if args.json:
        print(json.dumps(_arrondir(resultat), indent=2, ensure_ascii=False))
    else:
        afficher_resultat(donnees, resultat)


if __name__ == "__main__":
    main()
