"""Modele 3 : colonne radiative locale.

Ce fichier garde la logique generale du modele : construire la colonne,
propager les flux entre les couches, produire les diagnostics et exposer la CLI.
Les formules physiques elementaires sont dans physique/calculs.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .donnees import charger_colonne_locale, charger_donnees_extraites
    from .physique import calculs as physique
except ImportError:  # Permet aussi : python modele3/modele3.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3.donnees import charger_colonne_locale, charger_donnees_extraites
    from modele3.physique import calculs as physique


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


# =============================================================================
# Construction de la colonne locale
# =============================================================================


def construire_bords_pression_hpa(pression_surface_hpa):
    """Construit [p_surface] + les niveaux de reference inferieurs."""

    if pression_surface_hpa <= 1.0:
        raise ValueError("La pression de surface doit etre superieure a 1 hPa.")

    bords = [float(pression_surface_hpa)]
    for pression in physique.PRESSION_BORDS_REFERENCE_HPA:
        if pression < pression_surface_hpa:
            bords.append(pression)

    if len(bords) < 2:
        raise ValueError("La colonne doit contenir au moins une couche.")
    return bords


def fraction_nuage_couche(donnees, pression_bas_hpa, pression_haut_hpa):
    """Choisit la fraction nuageuse de la couche avec la logique low/mid/high."""

    surface = donnees["surface"]
    pression_surface_hpa = surface["pression_surface_pa"] / 100.0
    pression_milieu_hpa = 0.5 * (pression_bas_hpa + pression_haut_hpa)
    ratio = pression_milieu_hpa / pression_surface_hpa

    if ratio >= 0.80 and surface.get("low_cloud") is not None:
        return physique.fraction(surface.get("low_cloud"))
    if ratio >= 0.45 and surface.get("medium_cloud") is not None:
        return physique.fraction(surface.get("medium_cloud"))
    if ratio < 0.45 and surface.get("high_cloud") is not None:
        return physique.fraction(surface.get("high_cloud"))

    profil = donnees["profil"]
    fractions_nuageuses = profil.get("fractions_nuageuses")
    if fractions_nuageuses:
        return physique.fraction(
            physique.moyenne_pression(
                profil["pressions_hpa"],
                fractions_nuageuses,
                pression_bas_hpa,
                pression_haut_hpa,
            )
        )

    return physique.fraction(surface.get("cloud_total"))


def construire_couches(donnees, co2_ppm=CO2_DEFAUT_PPM):
    """Construit les couches locales utilisees par le transfert radiatif."""

    profil = donnees["profil"]
    surface = donnees["surface"]
    bords = construire_bords_pression_hpa(surface["pression_surface_pa"] / 100.0)
    couches = []

    for indice, (pression_bas_hpa, pression_haut_hpa) in enumerate(
        zip(bords[:-1], bords[1:]),
        start=1,
    ):
        delta_p_pa = (pression_bas_hpa - pression_haut_hpa) * 100.0
        if delta_p_pa <= 0.0:
            continue

        temperature_k = physique.moyenne_pression(
            profil["pressions_hpa"],
            profil["temperatures_k"],
            pression_bas_hpa,
            pression_haut_hpa,
        )
        humidite = physique.moyenne_pression(
            profil["pressions_hpa"],
            profil["humidites_specifiques_kgkg"],
            pression_bas_hpa,
            pression_haut_hpa,
        )
        humidite = max(0.0, humidite)
        masse_air = physique.masse_air_depuis_delta_p(delta_p_pa)
        masse_h2o = physique.masse_h2o_colonne(humidite, masse_air)

        couches.append(
            {
                "nom": f"couche_{indice:02d}",
                "pression_bas_pa": pression_bas_hpa * 100.0,
                "pression_haut_pa": pression_haut_hpa * 100.0,
                "pression_bas_hpa": pression_bas_hpa,
                "pression_haut_hpa": pression_haut_hpa,
                "temperature_k": temperature_k,
                "humidite_specifique_kgkg": humidite,
                "co2_ppm": co2_ppm,
                "fraction_nuageuse": fraction_nuage_couche(
                    donnees,
                    pression_bas_hpa,
                    pression_haut_hpa,
                ),
                "masse_air_kg_m2": masse_air,
                "masse_h2o_kg_m2": masse_h2o,
            }
        )

    return couches


# =============================================================================
# Propagation des flux dans la colonne
# =============================================================================


def propager_flux_montant(flux_surface_bande, bande, couches):
    """Propage un flux infrarouge montant de la surface vers le sommet."""

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


def propager_flux_descendant(bande, couches):
    """Propage l'emission atmospherique descendante vers la surface."""

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


def comparaison_validation(resultat, validation_flux):
    """Compare les flux principaux aux donnees ERA5 disponibles."""

    comparaison = {}
    if "avg_sdlwrf" in validation_flux:
        comparaison["ecart_LW_down_surface_W_m2"] = (
            resultat["LW_down_surface"] - validation_flux["avg_sdlwrf"]
        )
    if "avg_snswrf" in validation_flux:
        comparaison["ecart_SW_absorbe_surface_W_m2"] = (
            resultat["SW_absorbe_surface"] - validation_flux["avg_snswrf"]
        )
    if "avg_tnlwrf" in validation_flux:
        olr_era5 = validation_flux["avg_tnlwrf"]
        if olr_era5 < 0.0:
            olr_era5 = -olr_era5
        comparaison["OLR_ERA5_W_m2"] = olr_era5
        comparaison["ecart_OLR_W_m2"] = resultat["OLR"] - olr_era5
    return comparaison


def calculer_colonne_radiative(
    donnees,
    temperature_surface_k=TEMPERATURE_SURFACE_DEFAUT_K,
    co2_ppm=CO2_DEFAUT_PPM,
    jour_annee=None,
    heure_solaire=12.0,
    moyenne_journaliere_sw=False,
    bandes=None,
):
    """Calcule les flux radiatifs montants et descendants d'une colonne."""

    if bandes is None:
        bandes = physique.BANDES_INFRAROUGES

    surface = donnees["surface"]
    if jour_annee is None:
        jour_annee = physique.jour_milieu_mois(surface["mois"])

    couches = construire_couches(donnees, co2_ppm=co2_ppm)
    emissivite_surface = surface["emissivite_surface"]
    flux_surface_total = physique.flux_lw_surface(
        temperature_surface_k,
        emissivite_surface,
    )

    flux_surface_bandes = 0.0
    flux_sommet_bandes = 0.0
    lw_down_surface = 0.0
    diagnostics_bandes = []
    diagnostics_couches_bandes = []

    for bande in bandes:
        flux_surface_bande = emissivite_surface * physique.flux_corps_noir_dans_bande(
            temperature_surface_k,
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        flux_sommet = propager_flux_montant(flux_surface_bande, bande, couches)
        flux_descendant = propager_flux_descendant(bande, couches)

        flux_surface_bandes += flux_surface_bande
        flux_sommet_bandes += flux_sommet
        lw_down_surface += flux_descendant

        diagnostics_bandes.append(
            {
                "bande": bande["nom"],
                "famille": bande["famille"],
                "role": bande["role"],
                "lambda_min_um": bande["lambda_min_um"],
                "lambda_max_um": bande["lambda_max_um"],
                "flux_surface_W_m2": flux_surface_bande,
                "flux_sommet_W_m2": flux_sommet,
                "flux_descendant_surface_W_m2": flux_descendant,
            }
        )
        for couche in couches:
            diagnostics_couches_bandes.append(
                physique.opacites_couche_bande(couche, bande)
            )

    olr = max(0.0, flux_surface_total - flux_surface_bandes) + flux_sommet_bandes
    lw_down_absorbe_surface = emissivite_surface * lw_down_surface

    cloud_total = surface.get("cloud_total")
    if cloud_total is None and couches:
        cloud_total = max(couche["fraction_nuageuse"] for couche in couches)
    albedo_cloud = physique.albedo_nuage_effectif(cloud_total)

    if moyenne_journaliere_sw:
        sw_incident_surface = physique.flux_solaire_moyen_journalier(
            surface["latitude_deg"],
            jour_annee,
        )
    else:
        sw_incident_surface = physique.flux_solaire_incident(
            surface["latitude_deg"],
            jour_annee,
            heure_solaire,
        )

    sw_absorbe_surface = physique.flux_sw_absorbe_surface(
        sw_incident_surface,
        surface["albedo_surface"],
        albedo_cloud,
    )

    resultat = {
        "lat_deg": surface["latitude_deg"],
        "lon_deg": surface["longitude_deg"],
        "mois": surface["mois"],
        "jour_annee": jour_annee,
        "heure_solaire": heure_solaire,
        "moyenne_journaliere_sw": moyenne_journaliere_sw,
        "temperature_surface_k": temperature_surface_k,
        "co2_ppm": co2_ppm,
        "SW_incident_surface": sw_incident_surface,
        "SW_absorbe_surface": sw_absorbe_surface,
        "LW_up_surface": flux_surface_total,
        "LW_down_surface": lw_down_surface,
        "LW_down_absorbe_surface": lw_down_absorbe_surface,
        "OLR": olr,
        "flux_net_radiatif_surface": (
            sw_absorbe_surface + lw_down_absorbe_surface - flux_surface_total
        ),
        "albedo_surface": surface["albedo_surface"],
        "albedo_cloud": albedo_cloud,
        "emissivite_surface": emissivite_surface,
        "couches": couches,
        "diagnostics_bandes": diagnostics_bandes,
        "diagnostics_couches_bandes": diagnostics_couches_bandes,
        "validation_flux": donnees.get("validation_flux", {}),
    }
    resultat["comparaison_validation"] = comparaison_validation(
        resultat,
        resultat["validation_flux"],
    )
    return resultat


# =============================================================================
# Interface en ligne de commande
# =============================================================================


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Modele 3 - colonne radiative locale")
    parseur.add_argument("--lat", type=float, default=48.8566, help="Latitude en degres")
    parseur.add_argument("--lon", type=float, default=2.3522, help="Longitude en degres")
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
        "--donnees-extraites",
        type=Path,
        default=None,
        help="Extrait JSON compact a utiliser a la place des gros fichiers locaux",
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
    print(f"albedo_cloud = {resultat['albedo_cloud']:.4f}")
    print(f"emissivite_surface = {resultat['emissivite_surface']:.4f}")
    print()
    print("flux_W_m2")
    print(f"SW_incident_surface = {resultat['SW_incident_surface']:.6f}")
    print(f"SW_absorbe_surface = {resultat['SW_absorbe_surface']:.6f}")
    print(f"LW_up_surface = {resultat['LW_up_surface']:.6f}")
    print(f"LW_down_surface = {resultat['LW_down_surface']:.6f}")
    print(f"LW_down_absorbe_surface = {resultat['LW_down_absorbe_surface']:.6f}")
    print(f"OLR = {resultat['OLR']:.6f}")
    print(f"flux_net_radiatif_surface = {resultat['flux_net_radiatif_surface']:.6f}")

    if resultat["validation_flux"]:
        print()
        print("validation_ERA5_W_m2")
        for nom, valeur in sorted(resultat["validation_flux"].items()):
            print(f"{nom} = {valeur:.6f}")
        for nom, valeur in sorted(resultat["comparaison_validation"].items()):
            print(f"{nom} = {valeur:.6f}")

    print()
    print("couches")
    print("nom, pression_hPa, T_K, q_kgkg, masse_H2O_kg_m2, cloud_fraction")
    for couche in resultat["couches"]:
        print(
            f"{couche['nom']}, "
            f"{couche['pression_bas_hpa']:.3f}-{couche['pression_haut_hpa']:.3f}, "
            f"{couche['temperature_k']:.3f}, "
            f"{couche['humidite_specifique_kgkg']:.8f}, "
            f"{couche['masse_h2o_kg_m2']:.6f}, "
            f"{couche['fraction_nuageuse']:.4f}"
        )


def main():
    args = construire_parseur().parse_args()
    if args.donnees_extraites is not None:
        donnees = charger_donnees_extraites(args.donnees_extraites)
    else:
        donnees = charger_colonne_locale(
            lat=args.lat,
            lon=args.lon,
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
