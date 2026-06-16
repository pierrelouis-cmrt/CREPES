"""Chargement simple des donnees du modele 3.

Le coeur physique du modele est dans modele3.py. Ici on prepare seulement un
dictionnaire avec trois blocs :

- surface
- profil
- validation_flux
"""

from __future__ import annotations

import json
import math
from pathlib import Path

try:
    import xarray as xr
except ImportError:  # Le modele fonctionne quand meme avec le JSON ou le secours.
    xr = None


RACINE_DEPOT = Path(__file__).resolve().parents[1]
RESSOURCES_RACINE = RACINE_DEPOT / "ressources"
EXTRAIT_PARIS_DEFAUT = Path(__file__).resolve().parent / "donnees_exemple" / "paris_2024_m07.json"

PRESSION_SURFACE_DEFAUT_PA = 101_325.0
ALBEDO_SURFACE_DEFAUT = 0.30
EMISSIVITE_SURFACE_DEFAUT = 0.98
EMISSIVITE_OCEAN = 0.985
EMISSIVITE_NEIGE_GLACE = 0.98

JOURS_CUMULES_MOIS = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def mois_depuis_jour_annee(jour_annee):
    if not 1 <= jour_annee <= 365:
        raise ValueError("jour_annee doit etre entre 1 et 365.")
    mois = 1
    for seuil in JOURS_CUMULES_MOIS[1:]:
        if jour_annee > seuil:
            mois += 1
    return min(mois, 12)


def _float_ou_none(valeur):
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


def _fraction(valeur):
    flottant = _float_ou_none(valeur)
    if flottant is None:
        return None
    return max(0.0, min(1.0, flottant))


def _liste_float(valeurs):
    if hasattr(valeurs, "tolist"):
        valeurs = valeurs.tolist()

    sortie = []

    def ajouter(valeur):
        if isinstance(valeur, list):
            for element in valeur:
                ajouter(element)
        else:
            flottant = _float_ou_none(valeur)
            if flottant is not None:
                sortie.append(flottant)

    ajouter(valeurs)
    return sortie


def _normaliser_longitude_era5(longitude_deg):
    return longitude_deg % 360.0


def _selection_mois(ds, mois):
    if not 1 <= mois <= 12:
        raise ValueError("mois doit etre entre 1 et 12.")
    if "valid_time" in ds.coords:
        return {"valid_time": ds.valid_time.values[mois - 1]}
    if "time" in ds.coords:
        return {"time": ds.time.values[mois - 1]}
    return {}


def _extraire_point(ds, variable, lat, lon, mois):
    selection = _selection_mois(ds, mois)
    selection["latitude"] = lat
    selection["longitude"] = _normaliser_longitude_era5(lon)
    return ds[variable].sel(selection, method="nearest").values


def _profil_de_secours():
    pressions = [1000, 925, 850, 700, 500, 300, 200, 100, 50, 20, 10, 1]
    temperatures = []
    humidites = []

    for pression in pressions:
        altitude_m = 44330.0 * (1.0 - (pression / 1013.25) ** (1.0 / 5.255))
        temperatures.append(max(216.65, 288.15 - 0.0065 * altitude_m))
        humidites.append(max(2e-6, 0.0075 * (pression / 1000.0) ** 4))

    return {
        "pressions_hpa": pressions,
        "temperatures_k": temperatures,
        "humidites_specifiques_kgkg": humidites,
        "fractions_nuageuses": None,
    }


def _surface_de_secours(lat, lon, mois):
    return {
        "latitude_deg": lat,
        "longitude_deg": lon,
        "mois": mois,
        "pression_surface_pa": PRESSION_SURFACE_DEFAUT_PA,
        "albedo_surface": ALBEDO_SURFACE_DEFAUT,
        "emissivite_surface": EMISSIVITE_SURFACE_DEFAUT,
        "cloud_total": 0.0,
        "low_cloud": None,
        "medium_cloud": None,
        "high_cloud": None,
        "land_fraction": None,
        "snow_ice_fraction": None,
        "temperature_2m_k": None,
        "skin_temperature_k": None,
    }


def _trouver_fichiers_era5(ressources_dir):
    if xr is None or not ressources_dir.exists():
        return None, None, None

    fichier_profil = None
    fichier_surface = None
    fichier_flux = None

    for chemin in sorted(ressources_dir.glob("**/*.nc")):
        try:
            with xr.open_dataset(chemin, decode_times=False) as ds:
                variables = set(ds.data_vars)
                if {"t", "q"}.issubset(variables) and "pressure_level" in ds.coords:
                    fichier_profil = chemin
                if {"sp", "fal"}.issubset(variables):
                    fichier_surface = chemin
                if {"avg_sdlwrf", "avg_snswrf", "avg_tnlwrf"}.issubset(variables):
                    fichier_flux = chemin
        except Exception:
            continue

    return fichier_profil, fichier_surface, fichier_flux


def _emissivite_simple(land_fraction, snow_ice_fraction):
    if snow_ice_fraction is not None and snow_ice_fraction > 0.5:
        return EMISSIVITE_NEIGE_GLACE
    if land_fraction is not None and land_fraction < 0.5:
        return EMISSIVITE_OCEAN
    return EMISSIVITE_SURFACE_DEFAUT


def _charger_depuis_era5(lat, lon, mois, ressources_dir):
    if xr is None:
        return None

    fichier_profil, fichier_surface, fichier_flux = _trouver_fichiers_era5(ressources_dir)
    if fichier_profil is None or fichier_surface is None:
        return None

    with xr.open_dataset(fichier_profil, decode_times=True) as ds:
        pressions = _liste_float(ds["pressure_level"].values)
        temperatures = _liste_float(_extraire_point(ds, "t", lat, lon, mois))
        humidites = [max(0.0, valeur) for valeur in _liste_float(_extraire_point(ds, "q", lat, lon, mois))]
        fractions_nuageuses = None
        if "cc" in ds.data_vars:
            fractions_nuageuses = [_fraction(v) or 0.0 for v in _liste_float(_extraire_point(ds, "cc", lat, lon, mois))]

    surface_vars = {}
    with xr.open_dataset(fichier_surface, decode_times=True) as ds:
        for nom in ds.data_vars:
            surface_vars[nom] = _float_ou_none(_extraire_point(ds, nom, lat, lon, mois))

    land_fraction = _fraction(surface_vars.get("lsm"))
    sea_ice = _fraction(surface_vars.get("siconc")) or 0.0
    snow_depth = surface_vars.get("sd") or 0.0
    snow_ice_fraction = max(sea_ice, 1.0 if snow_depth > 0.01 else 0.0)

    albedo = _fraction(surface_vars.get("fal"))
    snow_albedo = _fraction(surface_vars.get("asn"))
    if snow_depth > 0.01 and snow_albedo is not None:
        albedo = max(albedo or ALBEDO_SURFACE_DEFAUT, snow_albedo)

    validation_flux = {}
    if fichier_flux is not None:
        with xr.open_dataset(fichier_flux, decode_times=True) as ds:
            for nom in ("avg_sdlwrf", "avg_snswrf", "avg_tnlwrf", "avg_sdswrf"):
                if nom in ds.data_vars:
                    valeur = _float_ou_none(_extraire_point(ds, nom, lat, lon, mois))
                    if valeur is not None:
                        validation_flux[nom] = valeur

    return {
        "surface": {
            "latitude_deg": lat,
            "longitude_deg": lon,
            "mois": mois,
            "pression_surface_pa": surface_vars.get("sp") or PRESSION_SURFACE_DEFAUT_PA,
            "albedo_surface": albedo or ALBEDO_SURFACE_DEFAUT,
            "emissivite_surface": _emissivite_simple(land_fraction, snow_ice_fraction),
            "cloud_total": _fraction(surface_vars.get("tcc")),
            "low_cloud": _fraction(surface_vars.get("lcc")),
            "medium_cloud": _fraction(surface_vars.get("mcc")),
            "high_cloud": _fraction(surface_vars.get("hcc")),
            "land_fraction": land_fraction,
            "snow_ice_fraction": snow_ice_fraction,
            "temperature_2m_k": _float_ou_none(surface_vars.get("t2m")),
            "skin_temperature_k": _float_ou_none(surface_vars.get("skt")),
        },
        "profil": {
            "pressions_hpa": pressions,
            "temperatures_k": temperatures,
            "humidites_specifiques_kgkg": humidites,
            "fractions_nuageuses": fractions_nuageuses,
        },
        "validation_flux": validation_flux,
        "source": f"ERA5 local: {fichier_profil.name}, {fichier_surface.name}",
    }


def charger_donnees_extraites(chemin):
    with Path(chemin).open(encoding="utf-8") as fichier:
        return json.load(fichier)


def sauvegarder_donnees_extraites(donnees, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8") as fichier:
        json.dump(donnees, fichier, indent=2, ensure_ascii=False)
        fichier.write("\n")


def charger_colonne_locale(
    lat=48.8566,
    lon=2.3522,
    mois=7,
    jour_annee=None,
    ressources_dir=RESSOURCES_RACINE,
    utiliser_extrait_defaut=True,
):
    if jour_annee is not None:
        mois = mois_depuis_jour_annee(jour_annee)

    donnees_era5 = _charger_depuis_era5(lat, lon, mois, Path(ressources_dir))
    if donnees_era5 is not None:
        return donnees_era5

    if utiliser_extrait_defaut and EXTRAIT_PARIS_DEFAUT.exists():
        extrait = charger_donnees_extraites(EXTRAIT_PARIS_DEFAUT)
        surface = extrait["surface"]
        meme_point = abs(surface["latitude_deg"] - lat) < 0.1 and abs(surface["longitude_deg"] - lon) < 0.1
        if meme_point and surface["mois"] == mois:
            return extrait

    return {
        "surface": _surface_de_secours(lat, lon, mois),
        "profil": _profil_de_secours(),
        "validation_flux": {},
        "source": "secours analytique simple",
    }
