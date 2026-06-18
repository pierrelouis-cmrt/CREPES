"""Modele 3 : colonne radiative locale.

Ce fichier garde la logique generale du modele : construire la colonne,
propager les flux entre les couches, produire les diagnostics et exposer la CLI.
Les formules physiques elementaires sont dans physique/calculs.py.
"""

from __future__ import annotations  # permet d'utiliser des annotations de type "modernes" (ex: list[float]) meme sur de vieilles versions de Python

import argparse  # pour construire l'interface en ligne de commande (CLI)
import json      # pour serialiser le resultat en JSON (option --json)
import sys
from pathlib import Path

# Bloc d'import "double mode" : le module peut etre importe comme un sous-package
# (import relatif avec ".") OU execute directement comme script (python modele3/modele3.py).
# Dans ce second cas, l'import relatif echoue (pas de package parent connu), donc on
# rattrape l'ImportError et on bricole sys.path pour retrouver le dossier parent et
# refaire un import absolu equivalent.
try:
    from .donnees import charger_colonne_locale, charger_donnees_extraites
    from .physique import calculs as physique
except ImportError:  # Permet aussi : python modele3/modele3.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3.donnees import charger_colonne_locale, charger_donnees_extraites
    from modele3.physique import calculs as physique


# Raccourcis vers des constantes physiques par defaut definies dans le module physique,
# pour pouvoir les utiliser directement comme valeurs par defaut des arguments CLI.
TEMPERATURE_SURFACE_DEFAUT_K = physique.TEMPERATURE_SURFACE_DEFAUT_K
CO2_DEFAUT_PPM = physique.CO2_DEFAUT_PPM


def _arrondir(objet):
    """Arrondit recursivement tous les flottants d'une structure (dict/list/valeur).

    Utilise uniquement avant l'export JSON, pour avoir une sortie lisible
    (6 decimales) sans modifier les valeurs utilisees en interne pour les calculs.
    """
    if isinstance(objet, float):
        return round(objet, 6)
    if isinstance(objet, dict):
        # On reconstruit un nouveau dict en appliquant l'arrondi a chaque valeur (recursif)
        return {cle: _arrondir(valeur) for cle, valeur in objet.items()}
    if isinstance(objet, list):
        # Idem pour les listes : on arrondit chaque element (qui peut lui-meme etre un dict/list)
        return [_arrondir(valeur) for valeur in objet]
    return objet  # types non concernes (str, int, bool, None...) : renvoyes tels quels


# =============================================================================
# Construction de la colonne locale
# =============================================================================


def construire_bords_pression_hpa(pression_surface_hpa):
    """Construit [p_surface] + les niveaux de reference inferieurs.

    Cette liste de "bords" delimite les couches verticales : chaque paire de
    bords consecutifs (bords[i], bords[i+1]) definira une couche atmospherique.
    """

    if pression_surface_hpa <= 1.0:
        # Garde-fou : une pression de surface <= 1 hPa n'a pas de sens physique ici
        raise ValueError("La pression de surface doit etre superieure a 1 hPa.")

    bords = [float(pression_surface_hpa)]  # le premier bord est toujours la surface elle-meme
    for pression in physique.PRESSION_BORDS_REFERENCE_HPA:
        # On ne garde que les niveaux de reference strictement au-dessus de la surface
        # (en pression decroissante = altitude croissante), pour ne pas avoir de couche
        # "negative" sous le sol
        if pression < pression_surface_hpa:
            bords.append(pression)

    if len(bords) < 2:
        # Il faut au moins 2 bords pour definir une couche (surface + un niveau au-dessus)
        raise ValueError("La colonne doit contenir au moins une couche.")
    return bords


def fraction_nuage_couche(donnees, pression_bas_hpa, pression_haut_hpa):
    """Choisit la fraction nuageuse de la couche avec la logique low/mid/high."""

    surface = donnees["surface"]
    pression_surface_hpa = surface["pression_surface_pa"] / 100.0  # conversion Pa -> hPa
    pression_milieu_hpa = 0.5 * (pression_bas_hpa + pression_haut_hpa)  # pression "moyenne" de la couche
    ratio = pression_milieu_hpa / pression_surface_hpa  # position relative de la couche dans la colonne (1 = surface, ~0 = sommet)

    # Logique de selection par tranche d'altitude relative :
    # - couche basse (ratio >= 0.80, proche de la surface) -> utiliser la donnee "low_cloud" si dispo
    if ratio >= 0.80 and surface.get("low_cloud") is not None:
        return physique.fraction(surface.get("low_cloud"))
    # - couche moyenne -> "medium_cloud"
    if ratio >= 0.45 and surface.get("medium_cloud") is not None:
        return physique.fraction(surface.get("medium_cloud"))
    # - couche haute (ratio < 0.45, proche du sommet) -> "high_cloud"
    if ratio < 0.45 and surface.get("high_cloud") is not None:
        return physique.fraction(surface.get("high_cloud"))

    # Si aucune des donnees low/medium/high n'est disponible, on essaie un profil
    # vertical detaille de fraction nuageuse (si fourni), en faisant la moyenne
    # ponderee par pression sur l'intervalle de la couche
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

    # Dernier recours : la fraction nuageuse totale (une seule valeur pour toute la colonne)
    return physique.fraction(surface.get("cloud_total"))


def construire_couches(donnees, co2_ppm=CO2_DEFAUT_PPM):
    """Construit les couches locales utilisees par le transfert radiatif."""

    profil = donnees["profil"]
    surface = donnees["surface"]
    # Bords de pression (hPa) qui vont decouper la colonne en couches successives
    bords = construire_bords_pression_hpa(surface["pression_surface_pa"] / 100.0)
    couches = []

    # On parcourt les paires de bords consecutifs (bords[0]-bords[1], bords[1]-bords[2], ...)
    # chaque paire = une couche, numerotee a partir de 1
    for indice, (pression_bas_hpa, pression_haut_hpa) in enumerate(
        zip(bords[:-1], bords[1:]),
        start=1,
    ):
        # Epaisseur de la couche en pression, convertie en Pa (delta_p > 0 car pression_bas > pression_haut)
        delta_p_pa = (pression_bas_hpa - pression_haut_hpa) * 100.0
        if delta_p_pa <= 0.0:
            # Securite : si jamais deux bords identiques/inverses se retrouvent cote a cote, on ignore la couche
            continue

        # Temperature moyenne de la couche, ponderee par pression a partir du profil vertical complet
        temperature_k = physique.moyenne_pression(
            profil["pressions_hpa"],
            profil["temperatures_k"],
            pression_bas_hpa,
            pression_haut_hpa,
        )
        # Humidite specifique moyenne de la couche (meme principe de ponderation)
        humidite = physique.moyenne_pression(
            profil["pressions_hpa"],
            profil["humidites_specifiques_kgkg"],
            pression_bas_hpa,
            pression_haut_hpa,
        )
        humidite = max(0.0, humidite)  # on interdit une humidite negative (artefact d'interpolation possible)
        # Masse d'air de la couche (par unite de surface), deduite directement de son epaisseur en pression
        masse_air = physique.masse_air_depuis_delta_p(delta_p_pa)
        # Masse de vapeur d'eau dans la couche = humidite specifique * masse d'air totale de la couche
        masse_h2o = physique.masse_h2o_colonne(humidite, masse_air)

        couches.append(
            {
                "nom": f"couche_{indice:02d}",  # ex: couche_01, couche_02...
                "pression_bas_pa": pression_bas_hpa * 100.0,
                "pression_haut_pa": pression_haut_hpa * 100.0,
                "pression_bas_hpa": pression_bas_hpa,
                "pression_haut_hpa": pression_haut_hpa,
                "temperature_k": temperature_k,
                "humidite_specifique_kgkg": humidite,
                "co2_ppm": co2_ppm,  # CO2 suppose uniforme sur toute la colonne (bien melange)
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

    flux = flux_surface_bande  # flux de depart, emis par la surface dans cette bande spectrale
    # On traverse les couches dans l'ordre naturel (du bas vers le haut, car "couches"
    # est construit de la surface vers le sommet) : le flux monte donc couche par couche
    for couche in couches:
        # Pour chaque couche : transmission (fraction du flux qui la traverse sans interaction)
        # et emissivite (fraction qu'elle absorbe/reemet), dependant de la bande spectrale
        diagnostic = physique.opacites_couche_bande(couche, bande)
        # Emission propre de la couche (corps noir a sa temperature, integree sur la bande)
        emission_couche = physique.flux_corps_noir_dans_bande(
            couche["temperature_k"],
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        # Bilan radiatif simple (type "two-stream") : ce qui sort de la couche = ce qui la
        # traverse (transmission * flux entrant) + ce qu'elle emet elle-meme (emissivite * emission)
        flux = diagnostic["transmission"] * flux + diagnostic["emissivite"] * emission_couche
    return flux


def propager_flux_descendant(bande, couches):
    """Propage l'emission atmospherique descendante vers la surface."""

    flux = 0.0  # rien ne vient "d'au-dessus du sommet" (pas de flux entrant initial)
    # On parcourt les couches en sens inverse (du sommet vers la surface, via reversed())
    # car le flux descendant se construit en partant du haut de la colonne
    for couche in reversed(couches):
        diagnostic = physique.opacites_couche_bande(couche, bande)
        emission_couche = physique.flux_corps_noir_dans_bande(
            couche["temperature_k"],
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        # Meme bilan que pour le flux montant, mais applique en descendant couche par couche
        flux = diagnostic["transmission"] * flux + diagnostic["emissivite"] * emission_couche
    return flux


def comparaison_validation(resultat, validation_flux):
    """Compare les flux principaux aux donnees ERA5 disponibles."""

    comparaison = {}
    # avg_sdlwrf = ERA5 "surface downward longwave radiation flux" (rayonnement IR descendant a la surface)
    if "avg_sdlwrf" in validation_flux:
        comparaison["ecart_LW_down_surface_W_m2"] = (
            resultat["LW_down_surface"] - validation_flux["avg_sdlwrf"]
        )
    # avg_snswrf = ERA5 "surface net shortwave radiation flux" (rayonnement solaire net absorbe en surface)
    if "avg_snswrf" in validation_flux:
        comparaison["ecart_SW_absorbe_surface_W_m2"] = (
            resultat["SW_absorbe_surface"] - validation_flux["avg_snswrf"]
        )
    # avg_tnlwrf = ERA5 "top net longwave radiation flux" (flux IR net au sommet de l'atmosphere)
    if "avg_tnlwrf" in validation_flux:
        olr_era5 = validation_flux["avg_tnlwrf"]
        if olr_era5 < 0.0:
            # ERA5 donne ce flux avec un signe (convention "net descendant"), on le repasse
            # en valeur positive pour le comparer directement a notre OLR (qui est sortant)
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
        # Par defaut, on utilise le decoupage spectral standard defini dans le module physique
        bandes = physique.BANDES_INFRAROUGES

    surface = donnees["surface"]
    if jour_annee is None:
        # Si le jour de l'annee n'est pas precise, on prend le jour "representatif" du mois fourni
        jour_annee = physique.jour_milieu_mois(surface["mois"])

    # Construction de la colonne verticale (liste de couches avec T, humidite, CO2, nuages...)
    couches = construire_couches(donnees, co2_ppm=co2_ppm)
    emissivite_surface = surface["emissivite_surface"]
    # Flux total emis par la surface (loi de Stefan-Boltzmann ponderee par l'emissivite)
    flux_surface_total = physique.flux_lw_surface(
        temperature_surface_k,
        emissivite_surface,
    )

    # Accumulateurs sur l'ensemble des bandes spectrales infrarouges
    flux_surface_bandes = 0.0   # somme du flux de surface, bande par bande
    flux_sommet_bandes = 0.0    # somme du flux montant arrive au sommet, bande par bande
    lw_down_surface = 0.0       # somme du flux descendant arrivant a la surface, bande par bande
    diagnostics_bandes = []
    diagnostics_couches_bandes = []

    for bande in bandes:
        # Flux emis par la surface, restreint a cette bande spectrale
        flux_surface_bande = emissivite_surface * physique.flux_corps_noir_dans_bande(
            temperature_surface_k,
            bande["lambda_min_um"],
            bande["lambda_max_um"],
        )
        # Propagation montante (surface -> sommet) et descendante (sommet -> surface) pour cette bande
        flux_sommet = propager_flux_montant(flux_surface_bande, bande, couches)
        flux_descendant = propager_flux_descendant(bande, couches)

        flux_surface_bandes += flux_surface_bande
        flux_sommet_bandes += flux_sommet
        lw_down_surface += flux_descendant

        # On garde une trace detaillee par bande (utile pour les diagnostics / le mode --json)
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
        # Et un diagnostic d'opacite (transmission/emissivite) pour chaque couche, dans cette bande
        for couche in couches:
            diagnostics_couches_bandes.append(
                physique.opacites_couche_bande(couche, bande)
            )

    # OLR (Outgoing Longwave Radiation) = rayonnement IR sortant au sommet de l'atmosphere.
    # On ajoute la part du flux de surface qui n'est couverte par aucune des bandes modelisees
    # (flux_surface_total - flux_surface_bandes, jamais negatif grace au max(0, ...))
    # a la part qui a effectivement traverse/ete reemise par les couches dans les bandes modelisees.
    olr = max(0.0, flux_surface_total - flux_surface_bandes) + flux_sommet_bandes
    # Flux IR descendant reellement absorbe par la surface (pondere par son emissivite)
    lw_down_absorbe_surface = emissivite_surface * lw_down_surface

    # Fraction nuageuse "globale" de la colonne : donnee directement si disponible,
    # sinon on prend le maximum des fractions nuageuses calculees couche par couche
    cloud_total = surface.get("cloud_total")
    if cloud_total is None and couches:
        cloud_total = max(couche["fraction_nuageuse"] for couche in couches)
    # Albedo effectif induit par la presence de nuages
    albedo_cloud = physique.albedo_nuage_effectif(cloud_total)

    if moyenne_journaliere_sw:
        # Flux solaire incident moyenne sur 24h (utile pour des bilans energetiques journaliers)
        sw_incident_surface = physique.flux_solaire_moyen_journalier(
            surface["latitude_deg"],
            jour_annee,
        )
    else:
        # Flux solaire incident instantane, a une heure solaire locale donnee
        sw_incident_surface = physique.flux_solaire_incident(
            surface["latitude_deg"],
            jour_annee,
            heure_solaire,
        )

    # Flux solaire effectivement absorbe en surface, apres reflexion par l'albedo de surface et des nuages
    sw_absorbe_surface = physique.flux_sw_absorbe_surface(
        sw_incident_surface,
        surface["albedo_surface"],
        albedo_cloud,
    )

    # Dictionnaire de resultats final, regroupant les parametres d'entree, les flux calcules
    # et les diagnostics detailles (couches, bandes, comparaison avec ERA5 si dispo)
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
        # Bilan radiatif net en surface : ce qui est absorbe (solaire + IR descendant) moins ce qui est emis
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
    # Ajout de la comparaison avec les flux ERA5 disponibles (si "validation_flux" est non vide)
    resultat["comparaison_validation"] = comparaison_validation(
        resultat,
        resultat["validation_flux"],
    )
    return resultat


# =============================================================================
# Interface en ligne de commande
# =============================================================================


def construire_parseur():
    """Definit tous les arguments acceptes par la CLI du modele 3."""
    parseur = argparse.ArgumentParser(description="Modele 3 - colonne radiative locale")
    parseur.add_argument("--lat", type=float, default=48.8566, help="Latitude en degres")
    parseur.add_argument("--lon", type=float, default=2.3522, help="Longitude en degres")
    parseur.add_argument("--mois", type=int, default=7, help="Mois 1..12")
    parseur.add_argument("--jour-annee", type=int, default=None, help="Jour 1..365")
    parseur.add_argument("--heure-solaire", type=float, default=12.0, help="Heure solaire locale")
    parseur.add_argument(
        "--moyenne-journaliere-sw",
        action="store_true",  # drapeau booleen : present = True, absent = False
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
    """Affiche le resultat de maniere lisible dans le terminal (mode texte, par opposition a --json)."""
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
        # Section optionnelle : seulement si des donnees ERA5 de validation sont presentes
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
        # Une ligne resumee par couche, du bas vers le haut de la colonne
        print(
            f"{couche['nom']}, "
            f"{couche['pression_bas_hpa']:.3f}-{couche['pression_haut_hpa']:.3f}, "
            f"{couche['temperature_k']:.3f}, "
            f"{couche['humidite_specifique_kgkg']:.8f}, "
            f"{couche['masse_h2o_kg_m2']:.6f}, "
            f"{couche['fraction_nuageuse']:.4f}"
        )


def main():
    """Point d'entree de la CLI : parse les arguments, charge les donnees, calcule, affiche."""
    args = construire_parseur().parse_args()
    if args.donnees_extraites is not None:
        # Mode "leger" : on lit un extrait JSON deja prepare, plutot que les gros fichiers source
        donnees = charger_donnees_extraites(args.donnees_extraites)
    else:
        # Mode "complet" : on va chercher la colonne locale correspondant a lat/lon/mois/jour
        # dans les jeux de donnees bruts (ex: ERA5)
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
        # Sortie machine-readable : on arrondit d'abord pour avoir un JSON propre
        print(json.dumps(_arrondir(resultat), indent=2, ensure_ascii=False))
    else:
        # Sortie humaine, formatee en texte
        afficher_resultat(donnees, resultat)


if __name__ == "__main__":
    main()