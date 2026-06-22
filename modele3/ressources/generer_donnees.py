"""Genere le paquet compact du modele 3."""

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
    raise SystemExit("xarray est requis pour generer les donnees 3.") from exc

try:
    from ..codes_python import physique
except ImportError:  # Permet aussi : python modele3/ressources/generer_donnees.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique


RACINE_DEPOT = Path(__file__).resolve().parents[2]
RESSOURCES_DEFAUT = RACINE_DEPOT / "ressources"
ALBEDO_DIR_DEFAUT = RESSOURCES_DEFAUT / "albedo"
SORTIE_DEFAUT = Path(__file__).resolve().parent / "donnees_precalculees" / "grille_5deg_2024"
FICHIER_NPZ = "donnees_colonnes_5deg_2024.npz"
LONGITUDE_CONVENTION = "-180..180"
EPAISSEUR_MIN_COUCHE_HPA = 0.1


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
    "albedo_surface": ("uint16", 1e-4, 0.0, 65535, "1"),
    "sw_toa_moyen_mensuel_w_m2": ("uint16", 0.1, 0.0, 65535, "W m-2"),
    "transmissivite_sw_mensuelle": ("uint16", 1e-4, 0.0, 65535, "1"),
    "land_fraction": ("uint16", 1e-4, 0.0, 65535, "1"),
    "snow_ice_fraction": ("uint16", 1e-4, 0.0, 65535, "1"),
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


def normaliser_longitudes_180(longitudes):
    return ((np.asarray(longitudes, dtype=float) + 180.0) % 360.0) - 180.0


def _distance_longitude_deg(source_lon, cible_lon):
    return np.abs(((np.asarray(source_lon, dtype=float) - float(cible_lon) + 180.0) % 360.0) - 180.0)


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
    longitude_source = np.asarray(ds["longitude"].to_numpy(), dtype=float)
    if np.nanmin(longitude_source) >= 0.0 and np.nanmax(longitude_source) > 180.0:
        longitudes_selection = np.asarray(longitudes, dtype=float) % 360.0
    else:
        longitudes_selection = normaliser_longitudes_180(longitudes)
    return ds.sel(latitude=latitudes, longitude=longitudes_selection, method="nearest")


def trouver_fichiers_era5(ressources_dir):
    fichier_profil = None
    fichier_surface = None
    fichier_flux = None
    for chemin in sorted(Path(ressources_dir).glob("**/*.nc")):
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


def corriger_albedo_neige_glace(albedo_surface, snow_ice_fraction):
    """Remplace les albedos nuls sur neige/glace par un repli physique simple."""

    albedo = _borne_fraction(albedo_surface)
    neige_glace = _borne_fraction(snow_ice_fraction)
    masque = (albedo <= 0.0) & (
        neige_glace > physique.SEUIL_FRACTION_NEIGE_GLACE_ALBEDO
    )
    fallback = physique.ALBEDO_SURFACE_SECOURS + neige_glace * (
        physique.ALBEDO_NEIGE_GLACE_SECOURS - physique.ALBEDO_SURFACE_SECOURS
    )
    corrige = np.where(masque, fallback, albedo).astype(np.float32)
    valeurs_corrigees = corrige[masque]
    diagnostics = {
        "zeros_neige_glace_corriges": int(np.count_nonzero(masque)),
        "seuil_fraction_neige_glace": physique.SEUIL_FRACTION_NEIGE_GLACE_ALBEDO,
        "albedo_surface_secours": physique.ALBEDO_SURFACE_SECOURS,
        "albedo_neige_glace_secours": physique.ALBEDO_NEIGE_GLACE_SECOURS,
        "fallback_min": float(valeurs_corrigees.min()) if valeurs_corrigees.size else None,
        "fallback_max": float(valeurs_corrigees.max()) if valeurs_corrigees.size else None,
    }
    return corrige, diagnostics


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
    source_lon = normaliser_longitudes_180(source_lon)
    target_lon = normaliser_longitudes_180(target_lon)
    valid = np.isfinite(valeurs)
    for i, lat in enumerate(target_lat):
        i_src = int(np.nanargmin(np.abs(source_lat - lat)))
        for j, lon in enumerate(target_lon):
            j_src = int(np.nanargmin(_distance_longitude_deg(source_lon, lon)))
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


def calculer_sw_toa_moyen_mensuel(latitudes, nombre_pas_horaires=96):
    """Moyenne mensuelle de S0 * max(cos(i), 0) sur des jours solaires complets."""

    bornes_mois = physique.JOURS_CUMULES_MOIS + [365]
    sortie = np.zeros((12, len(latitudes)), dtype=np.float32)
    for mois in range(12):
        jour_debut = bornes_mois[mois] + 1
        jour_fin = bornes_mois[mois + 1]
        for indice_lat, latitude in enumerate(latitudes):
            total = 0.0
            nombre_points = 0
            for jour_annee in range(jour_debut, jour_fin + 1):
                for indice_heure in range(nombre_pas_horaires):
                    heure = 24.0 * (indice_heure + 0.5) / nombre_pas_horaires
                    total += physique.flux_solaire_incident(float(latitude), jour_annee, heure)
                    nombre_points += 1
            sortie[mois, indice_lat] = total / nombre_points
    return sortie


def calculer_transmissivite_sw(era5_sw_down_surface, sw_toa_moyen_mensuel):
    denominateur = sw_toa_moyen_mensuel[:, :, None]
    avec_soleil = denominateur > 1e-6
    brut = np.divide(
        era5_sw_down_surface,
        denominateur,
        out=np.zeros_like(era5_sw_down_surface, dtype=np.float32),
        where=avec_soleil,
    )
    brut = np.nan_to_num(brut, nan=0.0, posinf=1.0, neginf=0.0)
    transmissivite = np.clip(brut, 0.0, 1.0).astype(np.float32)
    diagnostics = {
        "valeurs_bornees_min": int(np.count_nonzero(brut < 0.0)),
        "valeurs_bornees_max": int(np.count_nonzero(brut > 1.0)),
        "valeurs_sans_soleil": int(np.count_nonzero(~avec_soleil)),
    }
    return transmissivite, diagnostics


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


def _bords_couches_valides(p_surface_hpa):
    bords = [float(p_surface_hpa)]
    for pression_hpa in physique.PRESSION_BORDS_REFERENCE_HPA:
        if pression_hpa < p_surface_hpa - EPAISSEUR_MIN_COUCHE_HPA:
            bords.append(float(pression_hpa))
    return bords


def diagnostiquer_couches_pretraitees(couches):
    p_bas = couches["pression_bas_couche_hpa"]
    p_haut = couches["pression_haut_couche_hpa"]
    valides = np.isfinite(p_bas) & np.isfinite(p_haut)
    delta = np.where(valides, p_bas - p_haut, np.nan)
    return {
        "epaisseur_min_hpa": EPAISSEUR_MIN_COUCHE_HPA,
        "couches_valides": int(np.count_nonzero(valides)),
        "couches_nulles_ou_negatives": int(np.count_nonzero(valides & (p_bas <= p_haut))),
        "delta_p_min_hpa": float(np.nanmin(delta)) if np.isfinite(delta).any() else None,
    }


def charger_profils_et_couches(fichier_profil, surface, latitudes, longitudes, annee):
    with xr.open_dataset(fichier_profil, decode_times=True) as ds:
        ds = _selection_temps(ds, annee)
        ds = _selection_grille(ds, latitudes, longitudes)
        ds.load()
        p_levels = ds["pressure_level"].to_numpy().astype(np.float64)
        temperature = ds["t"].to_numpy().astype(np.float32)
        humidite = np.maximum(ds["q"].to_numpy().astype(np.float32), 0.0)

    shape = (12, len(physique.PRESSION_BORDS_REFERENCE_HPA), len(latitudes), len(longitudes))
    sortie = {
        "pression_bas_couche_hpa": np.full(shape, np.nan, dtype=np.float32),
        "pression_haut_couche_hpa": np.full(shape, np.nan, dtype=np.float32),
        "temperature_couche_k": np.full(shape, np.nan, dtype=np.float32),
        "humidite_specifique_couche_kgkg": np.full(shape, np.nan, dtype=np.float32),
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
                bords = _bords_couches_valides(ps)
                for p_bas, p_haut in zip(bords[:-1], bords[1:]):
                    indice_couche = physique.PRESSION_BORDS_REFERENCE_HPA.index(p_haut)
                    t_moy = _moyenne_profile(p_levels, temperature[mois, :, i, j], p_bas, p_haut)
                    q_moy = _moyenne_profile(p_levels, humidite[mois, :, i, j], p_bas, p_haut)
                    if not math.isfinite(t_moy) or not math.isfinite(q_moy):
                        continue
                    delta_p_pa = (p_bas - p_haut) * 100.0
                    masse_air = physique.masse_air_depuis_delta_p(delta_p_pa)
                    masse_h2o = physique.masse_h2o_colonne(q_moy, masse_air)
                    sortie["pression_bas_couche_hpa"][mois, indice_couche, i, j] = p_bas
                    sortie["pression_haut_couche_hpa"][mois, indice_couche, i, j] = p_haut
                    sortie["temperature_couche_k"][mois, indice_couche, i, j] = t_moy
                    sortie["humidite_specifique_couche_kgkg"][mois, indice_couche, i, j] = q_moy
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
        "# Donnees precalculees 3\n\n"
        "Paquet compact genere depuis les ressources racine du depot. Ce dossier\n"
        "est la source normale du calcul 3 et l'entree grille du modele 4.\n\n"
        f"- Fichier: `{FICHIER_NPZ}`\n"
        f"- Resolution: {metadata['resolution_deg']} degres\n"
        f"- Annee: {metadata['annee']}\n"
        "- Grille: 36 latitudes x 72 longitudes x 12 mois\n"
        "- Usage normal: `modele3.codes_python.donnees.charger_paquet_grille`.\n\n"
        "## Provenance\n\n"
        "| Champ | Source active | Transformation |\n"
        "| --- | --- | --- |\n"
        "| Profils `T`, `q` | ERA5 pression, `ressources/*.nc` | Moyennes par couche de pression 3. |\n"
        "| Surface | ERA5 single levels, `ressources/**/*.nc` | Selection au plus proche sur grille 5 degres. |\n"
        "| Flux de validation | ERA5 flux moyens | Stockes pour comparaison, jamais pour recalibrer. |\n"
        "| Transmissivite SW | Geometrie solaire 3 + ERA5 `avg_sdswrf` | `ERA5 SW_down / moyenne_mensuelle(S0*cos(i))`, borne `[0, 1]`. |\n"
        "| Albedo surface | `ressources/albedo/albedo01.csv` ... `albedo12.csv` | Longitudes normalisees -180..180, selection mensuelle au plus proche, puis correction des zeros sur neige/glace. |\n"
        "\n"
        "Les fichiers `ressources/albedo/*` sont des copies racine des donnees utiles\n"
        "historiquement presentes dans le modele 0. Le code 3 ne lit pas le dossier\n"
        "`modele0_maintenance/`. Les valeurs d'albedo nulles sur des mailles\n"
        "neige/glace viennent surtout de mois polaires ou le rapport source\n"
        "`SW_UP / SW_DOWN` n'est pas observable ; elles sont remplacees par un\n"
        "melange simple entre `0.30` et `0.65` selon la fraction neige/glace.\n\n"
        "Les couches verticales dont l'epaisseur serait inferieure a 0.1 hPa sont\n"
        "ignorees avant stockage pour eviter des couches nulles apres quantification.\n\n"
        "## Contenu\n\n"
        "Le `.npz` contient seulement les champs necessaires au calcul normal :\n"
        "coordonnees, poids de surface, pression de surface, albedo, transmissivite\n"
        "court-onde mensuelle, champs surface utiles, flux ERA5 de validation et\n"
        "couches pretraitees. Les facteurs de quantification, unites et sources\n"
        "sont dans `metadata.json`.\n",
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
    _message("Calcul transmissivite court-onde mensuelle")
    sw_toa_moyen = calculer_sw_toa_moyen_mensuel(latitudes)
    transmissivite_sw, diagnostics_transmissivite = calculer_transmissivite_sw(
        flux["era5_sw_down_surface_w_m2"],
        sw_toa_moyen,
    )
    albedo_surface, diagnostics_albedo = corriger_albedo_neige_glace(
        albedo_surface,
        surface["snow_ice_fraction"],
    )
    _message("Construction des couches pretraitees")
    couches = charger_profils_et_couches(fichier_profil, surface, latitudes, longitudes, args.annee)
    diagnostics_couches = diagnostiquer_couches_pretraitees(couches)

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
        "sw_toa_moyen_mensuel_w_m2": sw_toa_moyen,
        "transmissivite_sw_mensuelle": transmissivite_sw,
        **flux,
        **couches,
    }

    metadata = {
        "nom": "modele3_grille_5deg_2024",
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
            "source": "constante_modele3",
        },
        "conventions": {
            "longitude_deg": LONGITUDE_CONVENTION,
            "albedo_nearest_neighbor": "longitudes source et cible normalisees -180..180",
            "shortwave_mensuel": "transmissivite et SW_TOA moyennes sur le mois complet",
        },
        "diagnostics_generation": {
            "transmissivite_sw": diagnostics_transmissivite,
            "albedo_surface": diagnostics_albedo,
            "couches_verticales": diagnostics_couches,
        },
        "sources_fichiers": {
            "era5_profil": str(Path(fichier_profil).relative_to(RACINE_DEPOT)),
            "era5_surface": str(Path(fichier_surface).relative_to(RACINE_DEPOT)),
            "era5_flux": str(Path(fichier_flux).relative_to(RACINE_DEPOT)) if fichier_flux else None,
            "albedo_dir": str(Path(args.albedo_dir).relative_to(RACINE_DEPOT)),
        },
        "references_externes": {
            "era5_pressure_levels_monthly": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means",
            "era5_single_levels_monthly": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means",
            "nasa_power_parameters": "https://power.larc.nasa.gov/docs/tutorials/parameters/",
            "nasa_power_methodology": "https://power.larc.nasa.gov/docs/methodology/",
        },
        "variables": {
            nom: _metadata_variable(nom, source)
            for nom, source in {
                "pression_surface_hpa": "ERA5 sp",
                "pression_bas_couche_hpa": "pretraitement ERA5 pression_surface + bords 3, couches <0.1 hPa ignorees",
                "pression_haut_couche_hpa": "bords 3, couches <0.1 hPa ignorees",
                "temperature_couche_k": "ERA5 t moyenne par couche",
                "temperature_2m_k": "ERA5 t2m",
                "skin_temperature_k": "ERA5 skt",
                "masse_air_couche_kg_m2": "delta_p / g",
                "humidite_specifique_couche_kgkg": "ERA5 q moyenne par couche",
                "masse_h2o_couche_kg_m2": "q * delta_p / g",
                "albedo_surface": "ressources/albedo/albedo01.csv..albedo12.csv, longitudes normalisees -180..180 + correction zero neige/glace",
                "sw_toa_moyen_mensuel_w_m2": "geometrie solaire modele 3, S0=1361 W m-2",
                "transmissivite_sw_mensuelle": "ERA5 avg_sdswrf / sw_toa_moyen_mensuel_w_m2",
                "land_fraction": "ERA5 lsm",
                "snow_ice_fraction": "ERA5 siconc ou sd > 0.01 m",
                "era5_lw_down_surface_w_m2": "ERA5 avg_sdlwrf",
                "era5_sw_net_surface_w_m2": "ERA5 avg_snswrf",
                "era5_olr_w_m2": "abs(ERA5 avg_tnlwrf)",
                "era5_sw_down_surface_w_m2": "ERA5 avg_sdswrf",
            }.items()
        },
        "notes": [
            "Les CSV d'albedo sont lus depuis ressources/albedo, copie racine des donnees utiles du modele 0.",
            "Les albedos nuls sur neige/glace sont remplaces par un repli physique 0.30..0.65, car le rapport SW_UP/SW_DOWN source est non observable en nuit polaire.",
            "Le modele 3 n'utilise pas de coefficient nuageux court-onde ou long-onde.",
            "transmissivite_sw_mensuelle corrige le court-onde surface avec ERA5, sans remplacer S0*cos(i).",
            "Les couches verticales <0.1 hPa sont ignorees avant stockage.",
            "Les poids_surface sont normalises pour sommer a 1 sur la grille.",
        ],
    }

    _message("Ecriture du paquet compact")
    npz_path = ecrire_paquet(args.output, tableaux, metadata, args.overwrite)
    _message(f"Paquet ecrit: {npz_path}")
    return npz_path


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Genere le paquet compact modele 3.")
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
