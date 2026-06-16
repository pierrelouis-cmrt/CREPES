"""Genere le paquet compact du modele 3.1.

Le script lit les donnees lourdes locales de `ressources/` et les ressources
albedo/CERES copiees dans `ressources/albedo`, puis ecrit un paquet `.npz`
compact utilisable par le modele 3.1 et le modele 4.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - message CLI
    raise SystemExit("xarray est requis pour generer les donnees 3.1.") from exc

try:
    from . import physique
except ImportError:  # Permet aussi : python modele3_1/generer_donnees.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3_1 import physique


RACINE_DEPOT = Path(__file__).resolve().parents[1]
RESSOURCES_DEFAUT = RACINE_DEPOT / "ressources"
ALBEDO_DIR_DEFAUT = RESSOURCES_DEFAUT / "albedo"
SORTIE_DEFAUT = Path(__file__).resolve().parent / "donnees_precalculees" / "grille_5deg_2024"
FICHIER_NPZ = "donnees_colonnes_5deg_2024.npz"
FICHIER_CERES = "CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc"


QUANTIFICATION = {
    "pression_surface_hpa": ("uint16", 0.1, 0.0, 65535, "hPa"),
    "pression_bas_couche_hpa": ("uint16", 0.1, 0.0, 65535, "hPa"),
    "pression_haut_couche_hpa": ("uint16", 0.1, 0.0, 65535, "hPa"),
    "temperature_couche_k": ("int16", 0.01, 250.0, -32768, "K"),
    "temperature_2m_k": ("int16", 0.01, 250.0, -32768, "K"),
    "skin_temperature_k": ("int16", 0.01, 250.0, -32768, "K"),
    "masse_air_couche_kg_m2": ("uint16", 0.1, 0.0, 65535, "kg m-2"),
    "humidite_specifique_couche_kgkg": ("uint16", 5e-7, 0.0, 65535, "kg kg-1"),
    "masse_h2o_couche_kg_m2": ("uint16", 0.001, 0.0, 65535, "kg m-2"),
    "fraction_nuageuse_couche": ("uint16", 1e-4, 0.0, 65535, "1"),
    "albedo_surface": ("uint16", 1e-4, 0.0, 65535, "1"),
    "albedo_nuages_effectif": ("uint16", 1e-4, 0.0, 65535, "1"),
    "land_fraction": ("uint16", 1e-4, 0.0, 65535, "1"),
    "snow_ice_fraction": ("uint16", 1e-4, 0.0, 65535, "1"),
    "cloud_total": ("uint16", 1e-4, 0.0, 65535, "1"),
    "low_cloud": ("uint16", 1e-4, 0.0, 65535, "1"),
    "medium_cloud": ("uint16", 1e-4, 0.0, 65535, "1"),
    "high_cloud": ("uint16", 1e-4, 0.0, 65535, "1"),
    "era5_lw_down_surface_w_m2": ("int16", 0.1, 0.0, -32768, "W m-2"),
    "era5_sw_net_surface_w_m2": ("int16", 0.1, 0.0, -32768, "W m-2"),
    "era5_olr_w_m2": ("int16", 0.1, 0.0, -32768, "W m-2"),
    "era5_sw_down_surface_w_m2": ("int16", 0.1, 0.0, -32768, "W m-2"),
}


def _message(texte):
    print(texte, flush=True)


def construire_grille(resolution):
    if resolution <= 0 or 180 % resolution != 0 or 360 % resolution != 0:
        raise ValueError("La resolution doit diviser 180 et 360.")
    latitudes = np.arange(-90.0 + resolution / 2.0, 90.0, resolution, dtype=np.float32)
    longitudes = np.arange(-180.0 + resolution / 2.0, 180.0, resolution, dtype=np.float32)
    poids_lat = np.cos(np.deg2rad(latitudes)).astype(np.float64)
    poids = poids_lat[:, None] * np.ones((1, len(longitudes)), dtype=np.float64)
    poids /= poids.sum()
    return latitudes, longitudes, poids.astype(np.float32)


def _selection_temps(ds, annee):
    coord = "valid_time" if "valid_time" in ds.coords else "time"
    valeurs = ds[coord]
    try:
        masque = valeurs.dt.year == annee
        ds_annee = ds.sel({coord: masque})
        if ds_annee.sizes.get(coord, 0) >= 12:
            return ds_annee.isel({coord: slice(0, 12)})
    except Exception:
        pass
    if ds.sizes.get(coord, 0) < 12:
        raise ValueError(f"Moins de 12 pas mensuels dans {coord}.")
    return ds.isel({coord: slice(0, 12)})


def _selection_grille(ds, latitudes, longitudes):
    longitudes_era5 = np.asarray(longitudes, dtype=float) % 360.0
    return ds.sel(latitude=latitudes, longitude=longitudes_era5, method="nearest")


def trouver_fichiers_era5(ressources_dir):
    fichier_profil = None
    fichier_surface = None
    fichier_flux = None
    for chemin in sorted(Path(ressources_dir).glob("**/*.nc")):
        if "CERES" in chemin.name:
            continue
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


def _borne_fraction(tableau, maximum=1.0):
    return np.clip(np.nan_to_num(tableau, nan=0.0), 0.0, maximum).astype(np.float32)


def _ouvrir_selection_surface(fichier, latitudes, longitudes, annee):
    with xr.open_dataset(fichier, decode_times=True) as ds:
        ds = _selection_temps(ds, annee)
        ds = _selection_grille(ds, latitudes, longitudes)
        ds.load()
        return ds


def charger_surface_era5(fichier_surface, latitudes, longitudes, annee):
    ds = _ouvrir_selection_surface(fichier_surface, latitudes, longitudes, annee)
    surface = {
        "pression_surface_hpa": (ds["sp"].to_numpy().astype(np.float32) / 100.0),
        "temperature_2m_k": ds["t2m"].to_numpy().astype(np.float32) if "t2m" in ds else np.nan,
        "skin_temperature_k": ds["skt"].to_numpy().astype(np.float32) if "skt" in ds else np.nan,
        "land_fraction": _borne_fraction(ds["lsm"].to_numpy().astype(np.float32)) if "lsm" in ds else np.nan,
        "cloud_total": _borne_fraction(ds["tcc"].to_numpy().astype(np.float32)) if "tcc" in ds else np.nan,
        "low_cloud": _borne_fraction(ds["lcc"].to_numpy().astype(np.float32)) if "lcc" in ds else np.nan,
        "medium_cloud": _borne_fraction(ds["mcc"].to_numpy().astype(np.float32)) if "mcc" in ds else np.nan,
        "high_cloud": _borne_fraction(ds["hcc"].to_numpy().astype(np.float32)) if "hcc" in ds else np.nan,
    }
    sea_ice = _borne_fraction(ds["siconc"].to_numpy().astype(np.float32)) if "siconc" in ds else 0.0
    snow_depth = ds["sd"].to_numpy().astype(np.float32) if "sd" in ds else 0.0
    surface["snow_ice_fraction"] = np.maximum(sea_ice, (snow_depth > 0.01).astype(np.float32))
    ds.close()
    return surface


def charger_flux_era5(fichier_flux, latitudes, longitudes, annee, allow_fallbacks):
    variables = {
        "avg_sdlwrf": "era5_lw_down_surface_w_m2",
        "avg_snswrf": "era5_sw_net_surface_w_m2",
        "avg_tnlwrf": "era5_olr_w_m2",
        "avg_sdswrf": "era5_sw_down_surface_w_m2",
    }
    shape = (12, len(latitudes), len(longitudes))
    if fichier_flux is None:
        if not allow_fallbacks:
            raise FileNotFoundError("Fichier flux ERA5 introuvable.")
        return {nom: np.full(shape, np.nan, dtype=np.float32) for nom in variables.values()}

    ds = _ouvrir_selection_surface(fichier_flux, latitudes, longitudes, annee)
    sortie = {}
    for entree, sortie_nom in variables.items():
        if entree in ds:
            valeurs = ds[entree].to_numpy().astype(np.float32)
            if entree == "avg_tnlwrf":
                valeurs = np.abs(valeurs)
            sortie[sortie_nom] = valeurs
        elif allow_fallbacks:
            sortie[sortie_nom] = np.full(shape, np.nan, dtype=np.float32)
        else:
            raise KeyError(f"Variable ERA5 flux manquante: {entree}")
    ds.close()
    return sortie


def _lire_csv_albedo(path):
    with Path(path).open(newline="", encoding="utf-8") as fichier:
        lecteur = csv.reader(fichier)
        header = next(lecteur)
        longitudes = np.array([float(valeur) for valeur in header[1:]], dtype=np.float64)
        latitudes = []
        lignes = []
        for ligne in lecteur:
            latitudes.append(float(ligne[0]))
            valeurs = []
            for valeur in ligne[1:]:
                if valeur == "":
                    valeurs.append(np.nan)
                else:
                    valeurs.append(float(valeur))
            lignes.append(valeurs)
    return np.array(latitudes, dtype=np.float64), longitudes, np.array(lignes, dtype=np.float64)


def _nearest_matrix(source_lat, source_lon, valeurs, target_lat, target_lon, allow_fallbacks, fallback):
    sortie = np.empty((len(target_lat), len(target_lon)), dtype=np.float32)
    valid = np.isfinite(valeurs)
    for i, lat in enumerate(target_lat):
        i_src = int(np.nanargmin(np.abs(source_lat - lat)))
        for j, lon in enumerate(target_lon):
            j_src = int(np.nanargmin(np.abs(source_lon - lon)))
            valeur = valeurs[i_src, j_src]
            if not math.isfinite(float(valeur)):
                if not allow_fallbacks:
                    raise ValueError(f"Valeur albedo manquante pres de lat={lat}, lon={lon}.")
                valeur = fallback
            if valid.any() and not math.isfinite(float(valeur)):
                valeur = fallback
            sortie[i, j] = np.clip(valeur, 0.0, 1.0)
    return sortie


def charger_albedo_surface(albedo_dir, latitudes, longitudes, allow_fallbacks):
    fichiers = [Path(albedo_dir) / f"albedo{mois:02d}.csv" for mois in range(1, 13)]
    manquants = [fichier for fichier in fichiers if not fichier.exists()]
    if manquants and not allow_fallbacks:
        raise FileNotFoundError("CSV albedo manquants: " + ", ".join(str(f) for f in manquants))
    if manquants:
        return np.full((12, len(latitudes), len(longitudes)), 0.30, dtype=np.float32)

    cartes = []
    for fichier in fichiers:
        source_lat, source_lon, valeurs = _lire_csv_albedo(fichier)
        cartes.append(
            _nearest_matrix(
                source_lat,
                source_lon,
                valeurs,
                latitudes,
                longitudes,
                allow_fallbacks,
                0.30,
            )
        )
    return np.stack(cartes, axis=0)


def charger_albedo_nuages(albedo_dir, latitudes, longitudes, annee, allow_fallbacks):
    fichier = Path(albedo_dir) / FICHIER_CERES
    if not fichier.exists():
        if not allow_fallbacks:
            raise FileNotFoundError(f"Fichier CERES introuvable: {fichier}")
        return np.zeros((12, len(latitudes), len(longitudes)), dtype=np.float32)

    with xr.open_dataset(fichier, decode_times=True) as ds:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
        cloud = xr.where(
            ds["solar_mon"] > 1e-6,
            (ds["toa_sw_all_mon"] - ds["toa_sw_clr_c_mon"]) / ds["solar_mon"],
            0.0,
        )
        try:
            cloud = cloud.sel(time=cloud.time.dt.year == annee)
        except Exception:
            pass
        if cloud.sizes.get("time", 0) < 12:
            if not allow_fallbacks:
                raise ValueError("CERES ne contient pas 12 mois utilisables.")
            return np.zeros((12, len(latitudes), len(longitudes)), dtype=np.float32)
        cloud = cloud.isel(time=slice(0, 12))
        cloud = cloud.sel(lat=latitudes, lon=longitudes, method="nearest")
        valeurs = cloud.to_numpy().astype(np.float32)
    return _borne_fraction(valeurs, maximum=0.95)


def _moyenne_profile(p_levels_hpa, valeurs, p_bas, p_haut):
    if p_bas <= p_haut:
        return np.nan
    p = np.asarray(p_levels_hpa, dtype=float)
    v = np.asarray(valeurs, dtype=float)
    masque = np.isfinite(p) & np.isfinite(v)
    if not masque.any():
        return np.nan
    p = p[masque]
    v = v[masque]
    ordre = np.argsort(p)
    p = p[ordre]
    v = v[ordre]
    points = [p_haut, p_bas]
    points.extend(float(x) for x in p if p_haut < x < p_bas)
    points = np.array(sorted(set(points)), dtype=float)
    interp = np.interp(points, p, v, left=v[0], right=v[-1])
    return float(np.trapezoid(interp, points) / (p_bas - p_haut))


def charger_profils_et_couches(fichier_profil, surface, latitudes, longitudes, annee):
    with xr.open_dataset(fichier_profil, decode_times=True) as ds:
        ds = _selection_temps(ds, annee)
        ds = _selection_grille(ds, latitudes, longitudes)
        ds.load()
        p_levels = ds["pressure_level"].to_numpy().astype(np.float64)
        temperature = ds["t"].to_numpy().astype(np.float32)
        humidite = np.maximum(ds["q"].to_numpy().astype(np.float32), 0.0)
        nuage = _borne_fraction(ds["cc"].to_numpy().astype(np.float32)) if "cc" in ds else None

    shape = (12, len(physique.PRESSION_BORDS_REFERENCE_HPA), len(latitudes), len(longitudes))
    sortie = {
        "pression_bas_couche_hpa": np.full(shape, np.nan, dtype=np.float32),
        "pression_haut_couche_hpa": np.full(shape, np.nan, dtype=np.float32),
        "temperature_couche_k": np.full(shape, np.nan, dtype=np.float32),
        "humidite_specifique_couche_kgkg": np.full(shape, np.nan, dtype=np.float32),
        "fraction_nuageuse_couche": np.full(shape, np.nan, dtype=np.float32),
        "masse_air_couche_kg_m2": np.full(shape, np.nan, dtype=np.float32),
        "masse_h2o_couche_kg_m2": np.full(shape, np.nan, dtype=np.float32),
    }

    p_surface = surface["pression_surface_hpa"]
    for mois in range(12):
        _message(f"  couches verticales mois {mois + 1}/12")
        for i in range(len(latitudes)):
            for j in range(len(longitudes)):
                ps = float(p_surface[mois, i, j])
                if not math.isfinite(ps) or ps <= 1.0:
                    continue
                bords = [ps] + [p for p in physique.PRESSION_BORDS_REFERENCE_HPA if p < ps]
                for p_bas, p_haut in zip(bords[:-1], bords[1:]):
                    indice_couche = physique.PRESSION_BORDS_REFERENCE_HPA.index(p_haut)
                    t_moy = _moyenne_profile(p_levels, temperature[mois, :, i, j], p_bas, p_haut)
                    q_moy = _moyenne_profile(p_levels, humidite[mois, :, i, j], p_bas, p_haut)
                    if not math.isfinite(t_moy) or not math.isfinite(q_moy):
                        continue
                    if nuage is None:
                        cc_moy = 0.0
                    else:
                        cc_moy = _moyenne_profile(p_levels, nuage[mois, :, i, j], p_bas, p_haut)
                    delta_p_pa = (p_bas - p_haut) * 100.0
                    masse_air = physique.masse_air_depuis_delta_p(delta_p_pa)
                    masse_h2o = physique.masse_h2o_colonne(q_moy, masse_air)
                    sortie["pression_bas_couche_hpa"][mois, indice_couche, i, j] = p_bas
                    sortie["pression_haut_couche_hpa"][mois, indice_couche, i, j] = p_haut
                    sortie["temperature_couche_k"][mois, indice_couche, i, j] = t_moy
                    sortie["humidite_specifique_couche_kgkg"][mois, indice_couche, i, j] = q_moy
                    sortie["fraction_nuageuse_couche"][mois, indice_couche, i, j] = np.clip(cc_moy, 0.0, 1.0)
                    sortie["masse_air_couche_kg_m2"][mois, indice_couche, i, j] = masse_air
                    sortie["masse_h2o_couche_kg_m2"][mois, indice_couche, i, j] = masse_h2o
    return sortie


def _quantifier(nom, valeurs):
    if nom not in QUANTIFICATION:
        return np.asarray(valeurs)
    dtype, scale, offset, missing, _unite = QUANTIFICATION[nom]
    valeurs = np.asarray(valeurs, dtype=np.float64)
    masque = ~np.isfinite(valeurs)
    valeurs_quantifiables = np.where(masque, offset, valeurs)
    quantifie = np.rint((valeurs_quantifiables - offset) / scale)
    if dtype == "uint16":
        minimum, maximum = 0, 65534
    else:
        minimum, maximum = -32767, 32767
    valides = quantifie[~masque]
    if valides.size and (valides.min() < minimum or valides.max() > maximum):
        raise ValueError(
            f"{nom} depasse la quantification {dtype}: "
            f"{valides.min()}..{valides.max()} autorise {minimum}..{maximum}"
        )
    quantifie = np.clip(quantifie, minimum, maximum).astype(dtype)
    quantifie[masque] = np.array(missing, dtype=dtype)
    return quantifie


def _metadata_variable(nom, source):
    dtype, scale, offset, missing, unite = QUANTIFICATION[nom]
    return {
        "dtype_stocke": dtype,
        "unite_physique": unite,
        "scale_factor": scale,
        "offset": offset,
        "valeur_manquante": missing,
        "source": source,
    }


def ecrire_paquet(sortie_dir, tableaux, metadata, overwrite):
    sortie_dir = Path(sortie_dir)
    npz_path = sortie_dir / FICHIER_NPZ
    metadata_path = sortie_dir / "metadata.json"
    readme_path = sortie_dir / "README.md"
    if (npz_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"La sortie existe deja: {sortie_dir}. Ajouter --overwrite.")
    sortie_dir.mkdir(parents=True, exist_ok=True)

    tableaux_quantifies = {nom: _quantifier(nom, valeurs) for nom, valeurs in tableaux.items()}
    np.savez_compressed(npz_path, **tableaux_quantifies)
    with metadata_path.open("w", encoding="utf-8") as fichier:
        json.dump(metadata, fichier, indent=2, ensure_ascii=False)
        fichier.write("\n")
    readme_path.write_text(
        "# Donnees precalculees 3.1\n\n"
        "Paquet compact genere depuis les ressources racine du depot. Ce dossier\n"
        "est la source normale du calcul 3.1 et la future entree grille du modele 4.\n\n"
        f"- Fichier: `{FICHIER_NPZ}`\n"
        f"- Resolution: {metadata['resolution_deg']} degres\n"
        f"- Annee: {metadata['annee']}\n"
        "- Grille: 36 latitudes x 72 longitudes x 12 mois\n"
        "- Usage normal: `modele3_1.donnees.charger_paquet_grille`.\n\n"
        "## Provenance\n\n"
        "| Champ | Source active | Transformation |\n"
        "| --- | --- | --- |\n"
        "| Profils `T`, `q`, `cc` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3.1. |\n"
        "| Surface et nuages | ERA5 single levels, `ressources/**/*.nc` | Selection au plus proche sur grille 5 degres. |\n"
        "| Flux de validation | ERA5 flux moyens | Stockes pour comparaison, jamais pour recalibrer. |\n"
        "| Albedo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Selection mensuelle au plus proche. |\n"
        "| Albedo nuages | `ressources/albedo/CERES_EBAF-TOA_Ed4.2.1_Subset_202401-202501.nc` | `(toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon`. |\n\n"
        "Les fichiers `ressources/albedo/*` sont des copies racine des donnees utiles\n"
        "historiquement presentes dans le modele 0. Le code 3.1 ne lit pas le dossier\n"
        "`modele0_maintenance/`.\n\n"
        "## Contenu\n\n"
        "Le `.npz` contient seulement les champs necessaires au calcul normal :\n"
        "coordonnees, poids de surface, pression de surface, albedos, diagnostics\n"
        "surface, flux ERA5 de validation et couches pretraitees. Les facteurs de\n"
        "quantification, unites et sources sont dans `metadata.json`.\n",
        encoding="utf-8",
    )
    return npz_path


def generer(args):
    latitudes, longitudes, poids_surface = construire_grille(args.resolution)
    fichier_profil, fichier_surface, fichier_flux = trouver_fichiers_era5(args.ressources_dir)

    _message("Sources detectees")
    _message(f"  profil ERA5  = {fichier_profil}")
    _message(f"  surface ERA5 = {fichier_surface}")
    _message(f"  flux ERA5    = {fichier_flux}")
    _message(f"  albedo       = {args.albedo_dir}")

    if args.dry_run:
        return None
    if fichier_profil is None or fichier_surface is None:
        raise FileNotFoundError("Fichiers ERA5 profil/surface requis introuvables.")

    _message("Chargement surface ERA5")
    surface = charger_surface_era5(fichier_surface, latitudes, longitudes, args.annee)
    _message("Chargement flux ERA5")
    flux = charger_flux_era5(fichier_flux, latitudes, longitudes, args.annee, args.allow_fallbacks)
    _message("Chargement albedo surface CSV")
    albedo_surface = charger_albedo_surface(
        args.albedo_dir,
        latitudes,
        longitudes,
        args.allow_fallbacks,
    )
    _message("Chargement albedo nuages CERES")
    albedo_nuages = charger_albedo_nuages(
        args.albedo_dir,
        latitudes,
        longitudes,
        args.annee,
        args.allow_fallbacks,
    )
    _message("Construction des couches pretraitees")
    couches = charger_profils_et_couches(fichier_profil, surface, latitudes, longitudes, args.annee)

    tableaux = {
        "lat_deg": latitudes.astype(np.float32),
        "lon_deg": longitudes.astype(np.float32),
        "mois": np.arange(1, 13, dtype=np.int16),
        "poids_surface": poids_surface,
        "pression_bords_reference_hpa": np.array(
            physique.PRESSION_BORDS_REFERENCE_HPA,
            dtype=np.float32,
        ),
        **surface,
        "albedo_surface": _borne_fraction(albedo_surface),
        "albedo_nuages_effectif": _borne_fraction(albedo_nuages, maximum=0.95),
        **flux,
        **couches,
    }

    metadata = {
        "nom": "modele3_1_grille_5deg_2024",
        "annee": args.annee,
        "resolution_deg": args.resolution,
        "shape": {
            "mois": 12,
            "lat": len(latitudes),
            "lon": len(longitudes),
            "couches": len(physique.PRESSION_BORDS_REFERENCE_HPA),
        },
        "fichier_npz": FICHIER_NPZ,
        "emissivite_surface": {
            "valeur": physique.EMISSIVITE_SURFACE_CONSTANTE,
            "source": "constante_modele3_1",
        },
        "sources_fichiers": {
            "era5_profil": str(Path(fichier_profil).relative_to(RACINE_DEPOT)),
            "era5_surface": str(Path(fichier_surface).relative_to(RACINE_DEPOT)),
            "era5_flux": str(Path(fichier_flux).relative_to(RACINE_DEPOT)) if fichier_flux else None,
            "albedo_dir": str(Path(args.albedo_dir).relative_to(RACINE_DEPOT)),
            "ceres": str((Path(args.albedo_dir) / FICHIER_CERES).relative_to(RACINE_DEPOT)),
        },
        "references_externes": {
            "era5_pressure_levels_monthly": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means",
            "era5_single_levels_monthly": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means",
            "nasa_power_parameters": "https://power.larc.nasa.gov/docs/tutorials/parameters/",
            "nasa_power_methodology": "https://power.larc.nasa.gov/docs/methodology/",
            "ceres_ebaf_toa_ed4_2_1": "https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1",
        },
        "variables": {
            nom: _metadata_variable(nom, source)
            for nom, source in {
                "pression_surface_hpa": "ERA5 sp",
                "pression_bas_couche_hpa": "pretraitement ERA5 pression_surface + bords 3.1",
                "pression_haut_couche_hpa": "bords 3.1",
                "temperature_couche_k": "ERA5 t moyenne par couche",
                "temperature_2m_k": "ERA5 t2m",
                "skin_temperature_k": "ERA5 skt",
                "masse_air_couche_kg_m2": "delta_p / g",
                "humidite_specifique_couche_kgkg": "ERA5 q moyenne par couche",
                "masse_h2o_couche_kg_m2": "q * delta_p / g",
                "fraction_nuageuse_couche": "ERA5 cc moyenne par couche",
                "albedo_surface": "ressources/albedo/albedo01.csv..albedo12.csv",
                "albedo_nuages_effectif": "CERES EBAF-TOA, (toa_sw_all - toa_sw_clr_c) / solar",
                "land_fraction": "ERA5 lsm",
                "snow_ice_fraction": "ERA5 siconc ou sd > 0.01 m",
                "cloud_total": "ERA5 tcc",
                "low_cloud": "ERA5 lcc",
                "medium_cloud": "ERA5 mcc",
                "high_cloud": "ERA5 hcc",
                "era5_lw_down_surface_w_m2": "ERA5 avg_sdlwrf",
                "era5_sw_net_surface_w_m2": "ERA5 avg_snswrf",
                "era5_olr_w_m2": "abs(ERA5 avg_tnlwrf)",
                "era5_sw_down_surface_w_m2": "ERA5 avg_sdswrf",
            }.items()
        },
        "notes": [
            "Les fichiers albedo/CERES sont lus depuis ressources/albedo, copie racine des donnees utiles du modele 0.",
            "Le modele 3.1 n'utilise pas de coefficient nuageux court-onde ou long-onde cache.",
            "Les poids_surface sont normalises pour sommer a 1 sur la grille.",
        ],
    }

    _message("Ecriture du paquet compact")
    npz_path = ecrire_paquet(args.output, tableaux, metadata, args.overwrite)
    _message(f"Paquet ecrit: {npz_path}")
    return npz_path


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Genere le paquet compact modele 3.1.")
    parseur.add_argument("--resolution", type=int, default=5)
    parseur.add_argument("--annee", type=int, default=2024)
    parseur.add_argument("--ressources-dir", type=Path, default=RESSOURCES_DEFAUT)
    parseur.add_argument("--albedo-dir", type=Path, default=ALBEDO_DIR_DEFAUT)
    parseur.add_argument("--output", type=Path, default=SORTIE_DEFAUT)
    parseur.add_argument("--overwrite", action="store_true")
    parseur.add_argument("--dry-run", action="store_true")
    parseur.add_argument("--allow-fallbacks", action="store_true")
    return parseur


def main():
    args = construire_parseur().parse_args()
    generer(args)


if __name__ == "__main__":
    main()
