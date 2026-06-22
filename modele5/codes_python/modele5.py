"""Modele 5 : evolution de T_surface avec echanges radiatifs horizontaux.

Le moteur reprend exactement le bilan de surface vectorise du modele 4 rapide.
Il ajoute un echange infrarouge entre les faces verticales des couches
atmospheriques de deux colonnes voisines. Les profils de reference proviennent
du paquet ERA5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from modele3.codes_python import physique
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille
    from modele4.codes_python import rapide as modele4_rapide
    from modele4.codes_python import surface
    from modele4.codes_python.modele4 import RZSM_MODELE0_DEFAUT
except ImportError:  # Permet aussi : python modele5/modele5.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille
    from modele4.codes_python import rapide as modele4_rapide
    from modele4.codes_python import surface
    from modele4.codes_python.modele4 import RZSM_MODELE0_DEFAUT


SORTIE_DEFAUT = Path(__file__).resolve().parents[1] / "sorties" / "simulation_modele5.npz"
RAYON_TERRE_M = 6_371_000.0
CONSTANTE_GAZ_AIR_SEC_J_KG_K = 287.05
NOMBRE_PAS_PLANCK = 96


@dataclass
class ConfigurationModele5:
    """Configuration du moteur de temperature avec couplage horizontal.

    ``facteur_horizontal`` permet de desactiver le nouveau terme avec ``0``.
    """

    jours: float = 1.0
    dt_s: float = 1800.0
    sortie_heures: float = 4.0
    co2_ppm: float = physique.CO2_DEFAUT_PPM
    temperature_initiale_k: float | None = None
    max_latitudes: int | None = None
    max_longitudes: int | None = None
    rzsm_csv: Path | None = None
    facteur_latent: float = 1.0
    mode_convection: str = "toutes"
    vent_m_s: float = surface.VENT_DEFAUT_M_S
    temperature_air_defaut_k: float = surface.TEMPERATURE_AIR_DEFAUT_K
    facteur_horizontal: float = 1.0
    afficher_progression: bool = True


def _flux_planck_bande_vectoriel(temperature_k, lambda_min_um, lambda_max_um):
    """Integre Planck sur une bande pour un tableau de temperatures.

    Cette version numpy est reservee au precalcul de grille. Elle reprend la
    meme grandeur hemispherique que ``physique.flux_corps_noir_dans_bande``
    tout en evitant des millions de boucles Python.
    """

    temperature = np.maximum(np.asarray(temperature_k, dtype=np.float64), 1.0)
    bornes = np.linspace(
        float(lambda_min_um) * 1e-6,
        float(lambda_max_um) * 1e-6,
        NOMBRE_PAS_PLANCK + 1,
        dtype=np.float64,
    )
    longueurs = 0.5 * (bornes[:-1] + bornes[1:])
    pas = (bornes[-1] - bornes[0]) / NOMBRE_PAS_PLANCK
    longueur = longueurs[:, None, None]
    exposant = (
        physique.PLANCK
        * physique.VITESSE_LUMIERE
        / (longueur * physique.BOLTZMANN * temperature[None, :, :])
    )
    luminance = (
        2.0
        * physique.PLANCK
        * physique.VITESSE_LUMIERE**2
        / longueur**5
        / np.expm1(np.minimum(exposant, 700.0))
    )
    return np.sum(np.pi * luminance, axis=0) * pas


def _epaisseur_hypsometrique_m(pression_bas_hpa, pression_haut_hpa, temperature_k):
    """Epaisseur geometrique approximatee d'une couche de pression."""

    valide = (
        np.isfinite(pression_bas_hpa)
        & np.isfinite(pression_haut_hpa)
        & np.isfinite(temperature_k)
        & (pression_bas_hpa > pression_haut_hpa)
        & (pression_haut_hpa > 0.0)
    )
    resultat = np.zeros_like(temperature_k, dtype=np.float64)
    rapport = np.ones_like(temperature_k, dtype=np.float64)
    rapport[valide] = pression_bas_hpa[valide] / pression_haut_hpa[valide]
    resultat[valide] = (
        CONSTANTE_GAZ_AIR_SEC_J_KG_K
        * np.maximum(temperature_k[valide], 1.0)
        / physique.GRAVITE
        * np.log(rapport[valide])
    )
    return resultat


def _precalculer_couches_horizontales(paquet, config, lat_indices, lon_indices, mois_utiles):
    """Prepare emission laterale, transmission verticale et geometrie des couches."""

    donnees = paquet["donnees"]
    bandes = physique.BANDES_INFRAROUGES
    indices_lat = list(lat_indices)
    indices_lon = list(lon_indices)
    n_mois = len(mois_utiles)
    n_couches = donnees["temperature_couche_k"].shape[1]
    n_bandes = len(bandes)
    shape_couche = (n_mois, n_couches, len(indices_lat), len(indices_lon))
    shape_bande = (n_mois, n_couches, n_bandes, len(indices_lat), len(indices_lon))

    epaisseur_m = np.zeros(shape_couche, dtype=np.float32)
    emission_ref = np.zeros(shape_bande, dtype=np.float32)
    transmission_vers_surface = np.zeros(shape_bande, dtype=np.float32)

    for indice_mois, mois in enumerate(mois_utiles):
        source = int(mois) - 1
        def selectionner_couches(nom):
            """Selectionne sans laisser l'indexation avancee permuter les axes."""

            valeurs = np.asarray(donnees[nom][source], dtype=np.float64)
            valeurs = np.take(valeurs, indices_lat, axis=1)
            return np.take(valeurs, indices_lon, axis=2)

        p_bas = np.asarray(
            selectionner_couches("pression_bas_couche_hpa"), dtype=np.float64
        )
        p_haut = np.asarray(
            selectionner_couches("pression_haut_couche_hpa"), dtype=np.float64
        )
        temperature = np.asarray(
            selectionner_couches("temperature_couche_k"), dtype=np.float64
        )
        masse_h2o = np.asarray(
            selectionner_couches("masse_h2o_couche_kg_m2"), dtype=np.float64
        )
        valide = (
            np.isfinite(p_bas)
            & np.isfinite(p_haut)
            & np.isfinite(temperature)
            & (p_bas > p_haut)
            & (p_haut > 0.0)
        )
        temperature = np.where(valide, np.maximum(temperature, 1.0), 1.0)
        masse_h2o = np.where(np.isfinite(masse_h2o), np.maximum(masse_h2o, 0.0), 0.0)
        epaisseur = _epaisseur_hypsometrique_m(p_bas, p_haut, temperature)
        epaisseur_m[indice_mois] = epaisseur

        # On garde en memoire ce que chaque couche peut emettre vers ses voisines.
        transmission_cumulee = np.ones((n_bandes, len(indices_lat), len(indices_lon)))
        delta_p_pa = np.maximum((p_bas - p_haut) * 100.0, 0.0)
        for couche in range(n_couches):
            for bande_indice, bande in enumerate(bandes):
                tau_co2 = (
                    bande["a_co2"]
                    * (config.co2_ppm / physique.CO2_REFERENCE_PPM)
                    * (delta_p_pa[couche] / 101_325.0)
                )
                tau_h2o = bande["a_h2o"] * (
                    masse_h2o[couche] / physique.MASSE_H2O_REFERENCE_KG_M2
                )
                transmission = np.exp(-physique.FACTEUR_DIFFUSIF * (tau_co2 + tau_h2o))
                transmission = np.where(valide[couche], transmission, 1.0)
                emissivite = 1.0 - transmission
                corps_noir = _flux_planck_bande_vectoriel(
                    temperature[couche],
                    bande["lambda_min_um"],
                    bande["lambda_max_um"],
                )
                emission_ref[indice_mois, couche, bande_indice] = emissivite * corps_noir
                transmission_vers_surface[indice_mois, couche, bande_indice] = (
                    transmission_cumulee[bande_indice]
                    * physique.EMISSIVITE_SURFACE_CONSTANTE
                )
                # La couche suivante voit seulement ce qui traverse deja celles du dessus.
                transmission_cumulee[bande_indice] *= transmission

    return {
        "epaisseur_m": epaisseur_m,
        "emission_ref": emission_ref,
        "transmission_vers_surface": transmission_vers_surface,
    }


def _geometrie_grille(latitudes, longitudes):
    """Aires de maille et longueurs des interfaces d'une grille reguliere."""

    if len(latitudes) == 0 or len(longitudes) == 0:
        raise ValueError("La grille horizontale ne peut pas etre vide.")
    if len(latitudes) > 1:
        dlat = abs(np.deg2rad(float(latitudes[1] - latitudes[0])))
    else:
        dlat = np.deg2rad(5.0)
    if len(longitudes) > 1:
        dlon = abs(np.deg2rad(float(longitudes[1] - longitudes[0])))
    else:
        dlon = np.deg2rad(5.0)

    lat_rad = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    lat_sud = np.maximum(lat_rad - 0.5 * dlat, -0.5 * np.pi)
    lat_nord = np.minimum(lat_rad + 0.5 * dlat, 0.5 * np.pi)
    # Les mailles retrecissent vers les poles, donc leur aire depend de la latitude.
    aire = RAYON_TERRE_M**2 * dlon * (np.sin(lat_nord) - np.sin(lat_sud))
    longueur_meridienne = RAYON_TERRE_M * dlat
    longueurs_nord = RAYON_TERRE_M * np.cos(lat_nord[:-1]) * dlon
    return {
        "aire_m2": aire,
        "longueur_meridienne_m": longueur_meridienne,
        "longueurs_nord_m": np.maximum(longueurs_nord, 0.0),
    }


def calculer_convergence_horizontale(emission_face, epaisseur_m, geometrie, periodique_longitude):
    """Retourne la convergence laterale par couche et par bande.

    ``emission_face[k, b, lat, lon]`` est un flux sortant en W m-2 de face
    verticale. Chaque interface est calculee une seule fois : le gain de l'une
    des deux colonnes est exactement la perte de l'autre en watts.
    """

    emission = np.asarray(emission_face, dtype=np.float64)
    epaisseur = np.asarray(epaisseur_m, dtype=np.float64)
    if emission.ndim != 4:
        raise ValueError("emission_face doit avoir la forme (couche, bande, lat, lon).")
    if epaisseur.shape != (emission.shape[0], emission.shape[2], emission.shape[3]):
        raise ValueError("epaisseur_m doit avoir la forme (couche, lat, lon).")

    _, _, n_lat, n_lon = emission.shape
    convergence = np.zeros_like(emission)
    aire = np.asarray(geometrie["aire_m2"], dtype=np.float64)

    # Interfaces nord-sud : pas de condition periodique aux poles.
    for i in range(max(0, n_lat - 1)):
        hauteur_face = 0.5 * (epaisseur[:, i, :] + epaisseur[:, i + 1, :])
        puissance = (
            (emission[:, :, i + 1, :] - emission[:, :, i, :])
            * hauteur_face[:, None, :]
            * geometrie["longueurs_nord_m"][i]
        )
        convergence[:, :, i, :] += puissance / aire[i]
        convergence[:, :, i + 1, :] -= puissance / aire[i + 1]

    # Interfaces est-ouest. La longitude est periodique uniquement sur la
    # grille globale ; une sous-grille de developpement garde des bords fermes.
    nombre_interfaces = n_lon if periodique_longitude else max(0, n_lon - 1)
    for j in range(nombre_interfaces):
        voisin = (j + 1) % n_lon
        hauteur_face = 0.5 * (epaisseur[:, :, j] + epaisseur[:, :, voisin])
        puissance = (
            (emission[:, :, :, voisin] - emission[:, :, :, j])
            * hauteur_face[:, None, :]
            * geometrie["longueur_meridienne_m"]
        )
        convergence[:, :, :, j] += puissance / aire[None, :]
        convergence[:, :, :, voisin] -= puissance / aire[None, :]

    return convergence


def _configuration_modele4(config):
    return modele4_rapide.ConfigurationRapide(
        jours=config.jours,
        dt_s=config.dt_s,
        sortie_heures=config.sortie_heures,
        co2_ppm=config.co2_ppm,
        temperature_initiale_k=config.temperature_initiale_k,
        max_latitudes=config.max_latitudes,
        max_longitudes=config.max_longitudes,
        rzsm_csv=config.rzsm_csv,
        facteur_latent=config.facteur_latent,
        mode_convection=config.mode_convection,
        vent_m_s=config.vent_m_s,
        temperature_air_defaut_k=config.temperature_air_defaut_k,
        afficher_progression=False,
    )


def simuler(paquet, config=None):
    """Simule la grille modele 5 et retourne le meme noyau de sortie que le 4."""

    if config is None:
        config = ConfigurationModele5()
    if config.jours <= 0 or config.dt_s <= 0 or config.sortie_heures <= 0:
        raise ValueError("jours, dt_s et sortie_heures doivent etre strictement positifs.")
    if config.facteur_horizontal < 0.0:
        raise ValueError("facteur_horizontal doit etre positif ou nul.")
    donnees = paquet["donnees"]
    lat_indices = modele4_rapide._indices_grille(
        len(donnees["lat_deg"]), maximum=config.max_latitudes
    )
    lon_indices = modele4_rapide._indices_grille(
        len(donnees["lon_deg"]), maximum=config.max_longitudes
    )
    latitudes = np.asarray(donnees["lat_deg"][list(lat_indices)], dtype=np.float64)
    longitudes = np.asarray(donnees["lon_deg"][list(lon_indices)], dtype=np.float64)
    nombre_pas = max(1, int(round(config.jours * 86400.0 / config.dt_s)))
    sortie_pas = max(1, int(round(config.sortie_heures * 3600.0 / config.dt_s)))
    mois_utiles = modele4_rapide._mois_utiles(nombre_pas, config.dt_s)
    index_mois = {mois: indice for indice, mois in enumerate(mois_utiles)}

    config_4 = _configuration_modele4(config)
    champs_surface = modele4_rapide._precalculer_champs(
        paquet, config_4, latitudes, longitudes, mois_utiles
    )
    champs_couches = _precalculer_couches_horizontales(
        paquet, config, lat_indices, lon_indices, mois_utiles
    )
    geometrie = _geometrie_grille(latitudes, longitudes)
    periodique_longitude = len(lon_indices) == len(donnees["lon_deg"])

    temperature = champs_surface["temperature_initiale"].copy()
    sorties_temperature = [temperature.astype(np.float32)]
    temps_sortie = [0.0]
    jours_sortie = [1.0]
    heures_sortie = [0.0]
    diagnostics_somme = {
        "SW_absorbe_surface": np.zeros_like(temperature),
        "LW_down_absorbe_surface": np.zeros_like(temperature),
        "LW_up_surface": np.zeros_like(temperature),
        "flux_latent": np.zeros_like(temperature),
        "flux_convection": np.zeros_like(temperature),
        "flux_horizontal_net_surface": np.zeros_like(temperature),
        "flux_horizontal_atmosphere": np.zeros_like(temperature),
        "flux_net_surface": np.zeros_like(temperature),
    }

    if config.afficher_progression:
        modele4_rapide._afficher_progression(0, nombre_pas, "pas")

    for pas in range(nombre_pas):
        t_sec = pas * config.dt_s
        jour_annee = int(t_sec // 86400.0) % 365 + 1
        indice_mois = index_mois[physique.mois_depuis_jour_annee(jour_annee)]
        albedo = champs_surface["albedo"][indice_mois]
        tau_sw = champs_surface["tau_sw"][indice_mois]
        lw_down = champs_surface["lw_down_absorbe"][indice_mois]
        temperature_air = champs_surface["temperature_air"][indice_mois]
        flux_latent = champs_surface["flux_latent"][indice_mois]
        capacite = champs_surface["capacite"][indice_mois]

        convergence = calculer_convergence_horizontale(
            champs_couches["emission_ref"][indice_mois],
            champs_couches["epaisseur_m"][indice_mois],
            geometrie,
            periodique_longitude,
        )
        flux_horizontal_atmosphere = np.sum(convergence, axis=(0, 1))
        flux_horizontal_surface = config.facteur_horizontal * np.sum(
            convergence * champs_couches["transmission_vers_surface"][indice_mois],
            axis=(0, 1),
        )
        # Seule la part transmise jusqu'au sol entre dans le bilan de surface.

        cosinus = modele4_rapide._cosinus_solaire_grille(
            latitudes, longitudes, jour_annee, t_sec
        )
        sw_absorbe = physique.CONSTANTE_SOLAIRE * cosinus * tau_sw * (1.0 - albedo)
        lw_up = physique.flux_lw_surface(temperature, physique.EMISSIVITE_SURFACE_CONSTANTE)
        flux_convection, h_convection = modele4_rapide._flux_convection_vectoriel(
            temperature, temperature_air, config_4
        )
        flux_net = (
            sw_absorbe
            + lw_down
            - lw_up
            - flux_latent
            - flux_convection
            + flux_horizontal_surface
        )
        d_lw = (
            4.0
            * physique.EMISSIVITE_SURFACE_CONSTANTE
            * physique.SIGMA
            * np.maximum(temperature, 1.0) ** 3
        )
        temperature = temperature + config.dt_s * flux_net / (
            capacite + config.dt_s * (d_lw + h_convection)
        )
        # La temperature avance d'un pas de temps avec le bilan net de surface.
        if not np.isfinite(temperature).all():
            raise FloatingPointError("Temperature non finie dans le modele 5.")

        diagnostics_somme["SW_absorbe_surface"] += sw_absorbe
        diagnostics_somme["LW_down_absorbe_surface"] += lw_down
        diagnostics_somme["LW_up_surface"] += lw_up
        diagnostics_somme["flux_latent"] += flux_latent
        diagnostics_somme["flux_convection"] += flux_convection
        diagnostics_somme["flux_horizontal_net_surface"] += flux_horizontal_surface
        diagnostics_somme["flux_horizontal_atmosphere"] += flux_horizontal_atmosphere
        diagnostics_somme["flux_net_surface"] += flux_net

        if (pas + 1) % sortie_pas == 0 or pas == nombre_pas - 1:
            t_sortie = (pas + 1) * config.dt_s
            sorties_temperature.append(temperature.astype(np.float32))
            temps_sortie.append(t_sortie)
            jours_sortie.append(t_sortie / 86400.0 + 1.0)
            heures_sortie.append((t_sortie / 3600.0) % 24.0)
        if config.afficher_progression:
            modele4_rapide._afficher_progression(pas + 1, nombre_pas, "pas")

    diagnostics_moyens = {
        nom: valeurs / nombre_pas for nom, valeurs in diagnostics_somme.items()
    }
    return {
        "temps_s": np.array(temps_sortie, dtype=np.float64),
        "jours": np.array(jours_sortie, dtype=np.float64),
        "heures": np.array(heures_sortie, dtype=np.float64),
        "temperature_surface_k": np.stack(sorties_temperature).astype(np.float32),
        "lat_deg": latitudes,
        "lon_deg": longitudes,
        "mois_precalcules": champs_surface["mois_utiles"],
        "capacite_surface_j_m2_k": champs_surface["capacite"].astype(np.float32),
        "diagnostics_moyens": diagnostics_moyens,
        "metadata": {
            "modele": "modele5",
            "description": "modele4 rapide + echanges radiatifs lateraux couche-a-couche",
            "jours": config.jours,
            "dt_s": config.dt_s,
            "sortie_heures": config.sortie_heures,
            "co2_ppm": config.co2_ppm,
            "mode_convection": config.mode_convection,
            "facteur_latent": config.facteur_latent,
            "vent_m_s": config.vent_m_s,
            "facteur_horizontal": config.facteur_horizontal,
            "condition_longitude": "periodique" if periodique_longitude else "bords_fermes_sous_grille",
            "condition_latitude": "bords_fermes_aux_poles_ou_aux_bords_sous_grille",
            "source_paquet": str(paquet["npz_path"]),
            "source_capacite": surface.source_capacite_surface(config.rzsm_csv),
            "lat_indices": list(lat_indices),
            "lon_indices": list(lon_indices),
            "mois_precalcules": [int(mois) for mois in champs_surface["mois_utiles"]],
            "schema_temperature": "semi-implicite lineaire LW_up + convection, horizontal explicite",
        },
    }


def enregistrer_resultat(resultat, chemin):
    """Ecrit la sortie NPZ, compatible avec les champs principaux du modele 4."""

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = resultat["diagnostics_moyens"]
    np.savez_compressed(
        chemin,
        temps_s=resultat["temps_s"],
        jours=resultat["jours"],
        heures=resultat["heures"],
        temperature_surface_k=resultat["temperature_surface_k"],
        lat_deg=resultat["lat_deg"],
        lon_deg=resultat["lon_deg"],
        mois_precalcules=resultat["mois_precalcules"],
        capacite_surface_j_m2_k=resultat["capacite_surface_j_m2_k"],
        sw_absorbe_surface_moyen_w_m2=diagnostics["SW_absorbe_surface"],
        lw_down_absorbe_surface_moyen_w_m2=diagnostics["LW_down_absorbe_surface"],
        lw_up_surface_moyen_w_m2=diagnostics["LW_up_surface"],
        flux_latent_moyen_w_m2=diagnostics["flux_latent"],
        flux_convection_moyen_w_m2=diagnostics["flux_convection"],
        flux_horizontal_net_surface_moyen_w_m2=diagnostics["flux_horizontal_net_surface"],
        flux_horizontal_atmosphere_moyen_w_m2=diagnostics["flux_horizontal_atmosphere"],
        flux_net_surface_moyen_w_m2=diagnostics["flux_net_surface"],
        metadata_json=json.dumps(resultat["metadata"], ensure_ascii=True, indent=2),
    )
    return chemin


def construire_parseur():
    parseur = argparse.ArgumentParser(
        description="Modele 5 - temperature de surface avec echanges radiatifs horizontaux"
    )
    parseur.add_argument("--paquet", type=Path, default=DOSSIER_PAQUET_DEFAUT)
    parseur.add_argument("--output", type=Path, default=SORTIE_DEFAUT)
    parseur.add_argument("--jours", type=float, default=1.0)
    parseur.add_argument("--dt", type=float, default=1800.0)
    parseur.add_argument("--sortie-heures", type=float, default=4.0)
    parseur.add_argument("--co2", type=float, default=physique.CO2_DEFAUT_PPM)
    parseur.add_argument("--temperature-initiale", type=float, default=None)
    parseur.add_argument("--max-latitudes", type=int, default=None)
    parseur.add_argument("--max-longitudes", type=int, default=None)
    parseur.add_argument(
        "--rzsm-csv",
        type=Path,
        default=None,
        help=f"CSV RZSM du modele 0. Exemple: {RZSM_MODELE0_DEFAUT}",
    )
    parseur.add_argument("--facteur-latent", type=float, default=1.0)
    parseur.add_argument(
        "--convection", choices=("aucune", "forcee", "naturelle", "toutes"), default="toutes"
    )
    parseur.add_argument("--vent", type=float, default=surface.VENT_DEFAUT_M_S)
    parseur.add_argument("--temperature-air", type=float, default=surface.TEMPERATURE_AIR_DEFAUT_K)
    parseur.add_argument(
        "--facteur-horizontal",
        type=float,
        default=1.0,
        help="Multiplicateur de l'echange horizontal ; 0 reproduit le modele 4 rapide.",
    )
    parseur.add_argument("--no-progress", action="store_true")
    return parseur


def main():
    args = construire_parseur().parse_args()
    paquet = charger_paquet_grille(args.paquet)
    config = ConfigurationModele5(
        jours=args.jours,
        dt_s=args.dt,
        sortie_heures=args.sortie_heures,
        co2_ppm=args.co2,
        temperature_initiale_k=args.temperature_initiale,
        max_latitudes=args.max_latitudes,
        max_longitudes=args.max_longitudes,
        rzsm_csv=args.rzsm_csv,
        facteur_latent=args.facteur_latent,
        mode_convection=args.convection,
        vent_m_s=args.vent,
        temperature_air_defaut_k=args.temperature_air,
        facteur_horizontal=args.facteur_horizontal,
        afficher_progression=not args.no_progress,
    )
    resultat = simuler(paquet, config)
    chemin = enregistrer_resultat(resultat, args.output)
    temperature = resultat["temperature_surface_k"]
    print("modele5_simulation_ok")
    print(f"sortie = {chemin}")
    print(f"shape_T_surface = {temperature.shape}")
    print(f"T_min_K = {float(np.nanmin(temperature)):.3f}")
    print(f"T_max_K = {float(np.nanmax(temperature)):.3f}")
    print(f"T_moyenne_finale_K = {float(np.nanmean(temperature[-1])):.3f}")
    print(
        "flux_horizontal_surface_moyen_W_m2 = "
        f"{float(np.nanmean(resultat['diagnostics_moyens']['flux_horizontal_net_surface'])):.6f}"
    )


if __name__ == "__main__":
    main()
