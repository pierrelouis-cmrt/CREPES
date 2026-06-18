"""Chargement des donnees compactes du modele 3.1.

Le calcul normal lit un paquet `.npz` deja prepare. Les fichiers ERA5/CERES
lourds sont lus seulement par `generer_donnees.py`, jamais par le calculateur
de colonne ou le modele 4.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import physique


RACINE_DEPOT = Path(__file__).resolve().parents[1]
DOSSIER_PAQUET_DEFAUT = (
    Path(__file__).resolve().parent / "donnees_precalculees" / "grille_5deg_2024"
)
FICHIER_NPZ_DEFAUT = "donnees_colonnes_5deg_2024.npz"
EMISSIVITE_SURFACE = physique.EMISSIVITE_SURFACE_CONSTANTE
ALBEDO_SURFACE_SECOURS = 0.30


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
    indice_lon = _indice_plus_proche(longitudes, lon)
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
    albedo_surface = physique.fraction(mensuel("albedo_surface"), defaut=ALBEDO_SURFACE_SECOURS)
    albedo_nuages = physique.fraction(
        mensuel("albedo_nuages_effectif"),
        defaut=0.0,
        maximum=0.95,
    )
    transmissivite_sw = None
    if "transmissivite_sw_mensuelle" in donnees:
        transmissivite_sw = physique.fraction(
            mensuel("transmissivite_sw_mensuelle"),
            defaut=0.0,
        )
    sw_toa_moyen = None
    if "sw_toa_moyen_mensuel_w_m2" in donnees:
        sw_toa_moyen = _float_ou_none(mensuel("sw_toa_moyen_mensuel_w_m2"))

    surface = {
        "latitude_deg": latitude,
        "longitude_deg": longitude,
        "mois": mois_sortie,
        "jour_annee": jour_annee,
        "pression_surface_pa": pression_surface_hpa * 100.0,
        "pression_surface_hpa": pression_surface_hpa,
        "albedo_surface": albedo_surface,
        "albedo_nuages_effectif": albedo_nuages,
        "sw_toa_moyen_mensuel_w_m2": sw_toa_moyen,
        "transmissivite_sw_mensuelle": transmissivite_sw,
        "emissivite_surface": EMISSIVITE_SURFACE,
        "source_albedo_surface": _source_variable(paquet, "albedo_surface"),
        "source_albedo_nuages_effectif": _source_variable(paquet, "albedo_nuages_effectif"),
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
        "cloud_total",
        "low_cloud",
        "medium_cloud",
        "high_cloud",
    ):
        if nom in donnees:
            valeur = _float_ou_none(mensuel(nom))
            if nom.endswith("cloud") or nom.endswith("fraction") or nom in {"cloud_total"}:
                valeur = None if valeur is None else physique.fraction(valeur)
            surface[nom] = valeur

    couches = []
    pression_bas = mensuel("pression_bas_couche_hpa")
    pression_haut = mensuel("pression_haut_couche_hpa")
    temperature = mensuel("temperature_couche_k")
    humidite = mensuel("humidite_specifique_couche_kgkg")
    fraction_nuageuse = mensuel("fraction_nuageuse_couche")
    masse_air = mensuel("masse_air_couche_kg_m2")
    masse_h2o = mensuel("masse_h2o_couche_kg_m2")

    for indice in range(len(pression_haut)):
        p_bas = _float_ou_none(pression_bas[indice])
        p_haut = _float_ou_none(pression_haut[indice])
        if p_bas is None or p_haut is None or p_bas <= p_haut:
            continue
        couches.append(
            {
                "nom": f"couche_{len(couches) + 1:02d}",
                "pression_bas_hpa": p_bas,
                "pression_haut_hpa": p_haut,
                "pression_bas_pa": p_bas * 100.0,
                "pression_haut_pa": p_haut * 100.0,
                "temperature_k": float(temperature[indice]),
                "humidite_specifique_kgkg": max(0.0, float(humidite[indice])),
                "fraction_nuageuse": physique.fraction(fraction_nuageuse[indice]),
                "masse_air_kg_m2": max(0.0, float(masse_air[indice])),
                "masse_h2o_kg_m2": max(0.0, float(masse_h2o[indice])),
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
        "source": f"paquet {paquet['npz_path'].name}",
        "indices_grille": {"lat": indice_lat, "lon": indice_lon},
    }


def iterer_colonnes(paquet, mois=None, jour_annee=None):
    latitudes = paquet["donnees"]["lat_deg"]
    longitudes = paquet["donnees"]["lon_deg"]
    for latitude in latitudes:
        for longitude in longitudes:
            yield extraire_colonne(paquet, float(latitude), float(longitude), mois, jour_annee)


def charger_donnees_extraites(chemin):
    with Path(chemin).open(encoding="utf-8") as fichier:
        donnees = json.load(fichier)
    return normaliser_colonne_legacy(donnees)


def normaliser_colonne_legacy(donnees):
    """Adapte un JSON modele 3 sans recreer les coefficients supprimes."""

    surface = donnees.setdefault("surface", {})
    surface["emissivite_surface"] = EMISSIVITE_SURFACE
    surface["source_emissivite_surface"] = "constante_0.98"
    if "albedo_surface" not in surface:
        surface["albedo_surface"] = ALBEDO_SURFACE_SECOURS
        surface["source_albedo_surface"] = "secours_0.30"
    else:
        surface.setdefault("source_albedo_surface", "json_fourni")
    if "albedo_nuages_effectif" not in surface:
        surface["albedo_nuages_effectif"] = 0.0
        surface["source_albedo_nuages_effectif"] = "absent_json_0.0"
    donnees.setdefault("validation_flux", {})
    return donnees
