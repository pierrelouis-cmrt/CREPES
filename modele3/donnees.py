"""Chargement du paquet compact du modele 3."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import physique


DOSSIER_PAQUET_DEFAUT = (
    Path(__file__).resolve().parent
    / "ressources"
    / "donnees_precalculees"
    / "grille_5deg_2024"
)
FICHIER_NPZ_DEFAUT = "donnees_colonnes_5deg_2024.npz"
EMISSIVITE_SURFACE = physique.EMISSIVITE_SURFACE_CONSTANTE
ALBEDO_SURFACE_SECOURS = physique.ALBEDO_SURFACE_SECOURS
LONGITUDE_CONVENTION = "-180..180"


def _float_ou_none(valeur):
    if valeur is None:
        return None
    if hasattr(valeur, "item"):
        try:
            valeur = valeur.item()
        except ValueError:
            pass
    try:
        flottant = float(valeur)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(flottant):
        return None
    return flottant


def _dequantifier(nom, tableau, metadata):
    variable = metadata.get("variables", {}).get(nom)
    if not variable or "scale_factor" not in variable:
        return np.array(tableau)

    scale = float(variable["scale_factor"])
    offset = float(variable.get("offset", 0.0))
    missing = variable.get("valeur_manquante")
    valeurs = np.array(tableau, dtype=np.float64)
    if missing is not None:
        valeurs[valeurs == float(missing)] = np.nan
    return valeurs * scale + offset


def _chemins_paquet(chemin):
    chemin = Path(chemin)
    if chemin.is_dir():
        metadata_path = chemin / "metadata.json"
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as fichier:
                metadata = json.load(fichier)
            npz_name = metadata.get("fichier_npz", FICHIER_NPZ_DEFAUT)
            return chemin, metadata_path, chemin / npz_name
        return chemin, metadata_path, chemin / FICHIER_NPZ_DEFAUT
    return chemin.parent, chemin.parent / "metadata.json", chemin


def charger_paquet_grille(chemin=DOSSIER_PAQUET_DEFAUT):
    dossier, metadata_path, npz_path = _chemins_paquet(chemin)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata introuvable: {metadata_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"paquet NPZ introuvable: {npz_path}")

    with metadata_path.open(encoding="utf-8") as fichier:
        metadata = json.load(fichier)

    donnees = {}
    with np.load(npz_path) as npz:
        for nom in npz.files:
            donnees[nom] = _dequantifier(nom, npz[nom], metadata)

    return {
        "dossier": dossier,
        "metadata_path": metadata_path,
        "npz_path": npz_path,
        "metadata": metadata,
        "donnees": donnees,
    }


def _indice_plus_proche(valeurs, cible):
    valeurs = np.asarray(valeurs, dtype=float)
    return int(np.nanargmin(np.abs(valeurs - cible)))


def _distance_longitude_deg(valeurs, cible):
    valeurs = np.asarray(valeurs, dtype=float)
    return np.abs(((valeurs - float(cible) + 180.0) % 360.0) - 180.0)


def _indice_longitude_plus_proche(valeurs, cible):
    return int(np.nanargmin(_distance_longitude_deg(valeurs, cible)))


def _extraire_mensuel(tableau, indice_lat, indice_lon, mois=None, jour_annee=None):
    tableau = np.asarray(tableau)
    if tableau.ndim == 2:
        valeurs = tableau[:, indice_lat]
    elif tableau.ndim == 3:
        valeurs = tableau[:, indice_lat, indice_lon]
    elif tableau.ndim == 4:
        valeurs = tableau[:, :, indice_lat, indice_lon]
    else:
        raise ValueError(f"Tableau mensuel de dimension inattendue: {tableau.shape}")

    if jour_annee is not None:
        mois_a, mois_b, poids_b = physique.poids_interpolation_mensuelle(jour_annee)
        return (1.0 - poids_b) * valeurs[mois_a] + poids_b * valeurs[mois_b]

    if mois is None:
        raise ValueError("mois ou jour_annee doit etre fourni.")
    if not 1 <= mois <= 12:
        raise ValueError("mois doit etre entre 1 et 12.")
    return valeurs[mois - 1]


def _source_variable(paquet, nom, defaut="inconnue"):
    return paquet["metadata"].get("variables", {}).get(nom, {}).get("source", defaut)


def extraire_colonne(paquet, lat, lon, mois=None, jour_annee=None):
    donnees = paquet["donnees"]
    latitudes = donnees["lat_deg"]
    longitudes = donnees["lon_deg"]
    indice_lat = _indice_plus_proche(latitudes, lat)
    indice_lon = _indice_longitude_plus_proche(longitudes, lon)
    latitude = float(latitudes[indice_lat])
    longitude = float(longitudes[indice_lon])

    if jour_annee is not None:
        mois_sortie = physique.mois_depuis_jour_annee(jour_annee)
    elif mois is not None:
        mois_sortie = int(mois)
    else:
        raise ValueError("mois ou jour_annee doit etre fourni.")

    def mensuel(nom):
        return _extraire_mensuel(donnees[nom], indice_lat, indice_lon, mois, jour_annee)

    pression_surface_hpa = float(mensuel("pression_surface_hpa"))
    snow_ice_fraction = None
    if "snow_ice_fraction" in donnees:
        snow_ice_fraction = _float_ou_none(mensuel("snow_ice_fraction"))
        if snow_ice_fraction is not None:
            snow_ice_fraction = physique.fraction(snow_ice_fraction)
    albedo_brut = mensuel("albedo_surface")
    albedo_surface = physique.albedo_surface_corrige_neige_glace(
        albedo_brut,
        snow_ice_fraction,
    )
    source_albedo_surface = _source_variable(paquet, "albedo_surface")
    if (
        physique.fraction(albedo_brut, defaut=ALBEDO_SURFACE_SECOURS) <= 0.0
        and snow_ice_fraction is not None
        and snow_ice_fraction > physique.SEUIL_FRACTION_NEIGE_GLACE_ALBEDO
    ):
        source_albedo_surface += " + correction zero neige/glace"
    transmissivite_sw = physique.fraction(mensuel("transmissivite_sw_mensuelle"), defaut=0.0)
    sw_toa_moyen = _float_ou_none(mensuel("sw_toa_moyen_mensuel_w_m2"))

    surface = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "mois": mois_sortie,
        "jour_annee": jour_annee,
        "pression_surface_pa": pression_surface_hpa * 100.0,
        "pression_surface_hpa": pression_surface_hpa,
        "albedo_surface": albedo_surface,
        "sw_toa_moyen_mensuel_w_m2": sw_toa_moyen,
        "transmissivite_sw_mensuelle": transmissivite_sw,
        "emissivite_surface": EMISSIVITE_SURFACE,
        "source_albedo_surface": source_albedo_surface,
        "source_transmissivite_sw_mensuelle": _source_variable(
            paquet,
            "transmissivite_sw_mensuelle",
        ),
        "source_emissivite_surface": "constante_0.98",
    }

    for nom in (
        "land_fraction",
        "snow_ice_fraction",
        "temperature_2m_k",
        "skin_temperature_k",
    ):
        if nom in donnees:
            if nom == "snow_ice_fraction":
                surface[nom] = snow_ice_fraction
            else:
                valeur = _float_ou_none(mensuel(nom))
                if nom.endswith("fraction"):
                    valeur = None if valeur is None else physique.fraction(valeur)
                surface[nom] = valeur

    couches = []
    diagnostics_donnees = {
        "convention_longitude": paquet["metadata"].get("conventions", {}).get(
            "longitude_deg",
            LONGITUDE_CONVENTION,
        ),
        "couches_ignorees_incompletes": 0,
        "couches_ignorees_non_positives": 0,
        "couches_non_positives_exemples": [],
    }
    pression_bas = mensuel("pression_bas_couche_hpa")
    pression_haut = mensuel("pression_haut_couche_hpa")
    temperature = mensuel("temperature_couche_k")
    humidite = mensuel("humidite_specifique_couche_kgkg")
    masse_air = mensuel("masse_air_couche_kg_m2")
    masse_h2o = mensuel("masse_h2o_couche_kg_m2")

    for indice in range(len(pression_haut)):
        p_bas = _float_ou_none(pression_bas[indice])
        p_haut = _float_ou_none(pression_haut[indice])
        temperature_k = _float_ou_none(temperature[indice])
        humidite_kgkg = _float_ou_none(humidite[indice])
        masse_air_kg_m2 = _float_ou_none(masse_air[indice])
        masse_h2o_kg_m2 = _float_ou_none(masse_h2o[indice])
        if (
            p_bas is None
            or p_haut is None
            or temperature_k is None
            or humidite_kgkg is None
            or masse_air_kg_m2 is None
            or masse_h2o_kg_m2 is None
        ):
            diagnostics_donnees["couches_ignorees_incompletes"] += 1
            continue
        if p_bas <= p_haut:
            diagnostics_donnees["couches_ignorees_non_positives"] += 1
            if len(diagnostics_donnees["couches_non_positives_exemples"]) < 5:
                diagnostics_donnees["couches_non_positives_exemples"].append(
                    {
                        "indice_source": int(indice),
                        "pression_bas_hpa": p_bas,
                        "pression_haut_hpa": p_haut,
                    }
                )
            continue
        if masse_air_kg_m2 <= 0.0:
            diagnostics_donnees["couches_ignorees_non_positives"] += 1
            if len(diagnostics_donnees["couches_non_positives_exemples"]) < 5:
                diagnostics_donnees["couches_non_positives_exemples"].append(
                    {
                        "indice_source": int(indice),
                        "pression_bas_hpa": p_bas,
                        "pression_haut_hpa": p_haut,
                        "masse_air_kg_m2": masse_air_kg_m2,
                    }
                )
            continue
        couches.append(
            {
                "nom": f"couche_{len(couches) + 1:02d}",
                "pression_bas_hpa": p_bas,
                "pression_haut_hpa": p_haut,
                "pression_bas_pa": p_bas * 100.0,
                "pression_haut_pa": p_haut * 100.0,
                "temperature_k": temperature_k,
                "humidite_specifique_kgkg": max(0.0, humidite_kgkg),
                "masse_air_kg_m2": masse_air_kg_m2,
                "masse_h2o_kg_m2": max(0.0, masse_h2o_kg_m2),
            }
        )

    validation_flux = {}
    correspondance_flux = {
        "era5_lw_down_surface_w_m2": "era5_lw_down_surface_w_m2",
        "era5_sw_net_surface_w_m2": "era5_sw_net_surface_w_m2",
        "era5_olr_w_m2": "era5_olr_w_m2",
        "era5_sw_down_surface_w_m2": "era5_sw_down_surface_w_m2",
    }
    for nom_tableau, nom_sortie in correspondance_flux.items():
        if nom_tableau in donnees:
            valeur = _float_ou_none(mensuel(nom_tableau))
            if valeur is not None:
                validation_flux[nom_sortie] = valeur

    return {
        "surface": surface,
        "couches": couches,
        "validation_flux": validation_flux,
        "diagnostics_donnees": diagnostics_donnees,
        "source": f"paquet {paquet['npz_path'].name}",
        "indices_grille": {"lat": indice_lat, "lon": indice_lon},
    }


def iterer_colonnes(paquet, mois=None, jour_annee=None):
    latitudes = paquet["donnees"]["lat_deg"]
    longitudes = paquet["donnees"]["lon_deg"]
    for latitude in latitudes:
        for longitude in longitudes:
            yield extraire_colonne(paquet, float(latitude), float(longitude), mois, jour_annee)
