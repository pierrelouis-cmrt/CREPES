"""Modele 4 : evolution d'une grille de temperature de surface."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from modele3 import physique
    from modele3.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.modele3 import calculer_colonne_radiative
except ImportError:  # Permet aussi : python modele4/modele4.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3 import physique
    from modele3.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.modele3 import calculer_colonne_radiative

try:
    from . import surface
except ImportError:  # Permet aussi : python modele4/modele4.py
    from modele4 import surface


SORTIE_DEFAUT = Path(__file__).resolve().parent / "sorties" / "simulation_modele4.npz"
MOIS_DEFAUT = tuple(range(1, 13))
RZSM_MODELE0_DEFAUT = surface.RZSM_MODELE0_DEFAUT


@dataclass
class ConfigurationModele4:
    jours: float = 1.0
    dt_s: float = 1800.0
    co2_ppm: float = physique.CO2_DEFAUT_PPM
    temperature_initiale_k: float | None = None
    frequence_sortie_pas: int = 48
    iterations_implicites: int = 4
    indices_lat: tuple[int, ...] | None = None
    indices_lon: tuple[int, ...] | None = None
    max_latitudes: int | None = None
    max_longitudes: int | None = None
    rzsm_csv: Path | None = RZSM_MODELE0_DEFAUT
    afficher_progression: bool = False
    surface: surface.ConfigurationSurface = surface.ConfigurationSurface()


def _indices_grille(taille, indices=None, maximum=None):
    if indices is not None:
        selection = [int(indice) for indice in indices]
    elif maximum is not None:
        selection = list(range(min(int(maximum), taille)))
    else:
        selection = list(range(taille))

    for indice in selection:
        if indice < 0 or indice >= taille:
            raise ValueError(f"Indice de grille hors limites: {indice} pour taille {taille}")
    return tuple(selection)


def _jour_et_heure(t_sec, lon_deg):
    jour_zero = int(t_sec // 86400.0)
    jour_annee = jour_zero % 365 + 1
    heure_solaire = ((t_sec / 3600.0) + lon_deg / 15.0) % 24.0
    return jour_annee, heure_solaire


def _temperature_initiale(colonne, temperature_imposee=None):
    if temperature_imposee is not None:
        return float(temperature_imposee)

    surface_colonne = colonne["surface"]
    for nom in ("skin_temperature_k", "temperature_2m_k"):
        valeur = surface_colonne.get(nom)
        if valeur is not None and np.isfinite(valeur):
            return float(valeur)
    return physique.TEMPERATURE_SURFACE_DEFAUT_K


def _construire_grille_initiale(paquet, config, lat_indices, lon_indices):
    latitudes = paquet["donnees"]["lat_deg"][list(lat_indices)]
    longitudes = paquet["donnees"]["lon_deg"][list(lon_indices)]
    grille_rzsm = (
        surface.charger_grille_rzsm(config.rzsm_csv) if config.rzsm_csv is not None else None
    )

    temperature = np.empty((len(lat_indices), len(lon_indices)), dtype=np.float64)
    capacite = np.empty_like(temperature)

    for i, latitude in enumerate(latitudes):
        for j, longitude in enumerate(longitudes):
            colonne = extraire_colonne(
                paquet,
                float(latitude),
                float(longitude),
                jour_annee=1,
            )
            temperature[i, j] = _temperature_initiale(
                colonne,
                temperature_imposee=config.temperature_initiale_k,
            )
            rzsm = surface.rzsm_plus_proche(
                grille_rzsm,
                colonne["surface"]["latitude_deg"],
                colonne["surface"]["longitude_deg"],
            )
            capacite[i, j] = surface.capacite_surface(colonne["surface"], rzsm=rzsm)

    return latitudes, longitudes, temperature, capacite


def _pas_implicite_cellule(
    colonne,
    temperature_depart_k,
    capacite,
    jour_annee,
    heure_solaire,
    config,
    moyenne_journaliere_sw=False,
):
    surface_colonne = colonne["surface"]
    temperature_air_k = surface.temperature_air(surface_colonne, config.surface)
    q_latent = surface.flux_latent_moyen(
        surface_colonne,
        facteur=config.surface.facteur_latent,
    )

    x = float(temperature_depart_k)
    dernier_flux = None
    for _ in range(max(1, int(config.iterations_implicites))):
        radiatif = calculer_colonne_radiative(
            colonne,
            temperature_surface_k=x,
            co2_ppm=config.co2_ppm,
            jour_annee=jour_annee,
            heure_solaire=heure_solaire,
            moyenne_journaliere_sw=moyenne_journaliere_sw,
        )
        q_convection = surface.flux_convection(x, temperature_air_k, config.surface)
        flux_net = (
            radiatif["SW_absorbe_surface"]
            + radiatif["LW_down_absorbe_surface"]
            - radiatif["LW_up_surface"]
            - q_latent
            - q_convection
        )
        dernier_flux = radiatif, q_latent, q_convection, flux_net

        residu = x - temperature_depart_k - config.dt_s * flux_net / capacite
        d_lw_up_d_t = (
            4.0
            * physique.EMISSIVITE_SURFACE_CONSTANTE
            * physique.SIGMA
            * max(x, 1.0) ** 3
        )
        d_conv_d_t = surface.derivee_flux_convection(x, temperature_air_k, config.surface)
        d_flux_net_d_t = -d_lw_up_d_t - d_conv_d_t
        derivee_residu = 1.0 - config.dt_s * d_flux_net_d_t / capacite
        if abs(derivee_residu) < 1e-12:
            break
        correction = residu / derivee_residu
        x -= correction
        if abs(correction) < 1e-6:
            break

    if not np.isfinite(x):
        raise FloatingPointError("Temperature non finie dans le modele 4.")

    radiatif, q_latent, q_convection, flux_net = dernier_flux
    return x, {
        "SW_absorbe_surface": radiatif["SW_absorbe_surface"],
        "LW_down_absorbe_surface": radiatif["LW_down_absorbe_surface"],
        "LW_up_surface": radiatif["LW_up_surface"],
        "flux_latent": q_latent,
        "flux_convection": q_convection,
        "flux_net_surface": flux_net,
    }


def _afficher_progression(etapes_effectuees, nombre_etapes, largeur=32, unite="etapes"):
    fraction = etapes_effectuees / max(1, nombre_etapes)
    remplis = int(round(largeur * fraction))
    barre = "#" * remplis + "-" * (largeur - remplis)
    pourcentage = 100.0 * fraction
    sys.stderr.write(
        f"\rmodele4 [{barre}] {pourcentage:6.2f}% "
        f"({etapes_effectuees}/{nombre_etapes} {unite})"
    )
    if etapes_effectuees >= nombre_etapes:
        sys.stderr.write("\n")
    sys.stderr.flush()


def simuler(paquet, config=None):
    """Lance une simulation modele 4 et retourne les tableaux resultats."""

    if config is None:
        config = ConfigurationModele4()
    if config.dt_s <= 0:
        raise ValueError("dt_s doit etre strictement positif.")
    if config.jours <= 0:
        raise ValueError("jours doit etre strictement positif.")
    if config.frequence_sortie_pas <= 0:
        raise ValueError("frequence_sortie_pas doit etre strictement positive.")

    donnees = paquet["donnees"]
    lat_indices = _indices_grille(
        len(donnees["lat_deg"]),
        indices=config.indices_lat,
        maximum=config.max_latitudes,
    )
    lon_indices = _indices_grille(
        len(donnees["lon_deg"]),
        indices=config.indices_lon,
        maximum=config.max_longitudes,
    )
    latitudes, longitudes, temperature, capacite = _construire_grille_initiale(
        paquet,
        config,
        lat_indices,
        lon_indices,
    )

    nombre_pas = int(round(config.jours * 86400.0 / config.dt_s))
    if nombre_pas < 1:
        nombre_pas = 1

    if config.afficher_progression:
        _afficher_progression(0, nombre_pas, unite="pas")

    sorties_temperature = [temperature.copy()]
    temps_sortie = [0.0]

    diagnostics_somme = {
        "SW_absorbe_surface": np.zeros_like(temperature),
        "LW_down_absorbe_surface": np.zeros_like(temperature),
        "LW_up_surface": np.zeros_like(temperature),
        "flux_latent": np.zeros_like(temperature),
        "flux_convection": np.zeros_like(temperature),
        "flux_net_surface": np.zeros_like(temperature),
    }

    for pas in range(nombre_pas):
        t_sec = pas * config.dt_s
        jour_annee = int(t_sec // 86400.0) % 365 + 1
        prochaine_temperature = np.empty_like(temperature)

        for i, latitude in enumerate(latitudes):
            for j, longitude in enumerate(longitudes):
                jour_annee, heure_solaire = _jour_et_heure(t_sec, float(longitude))
                colonne = extraire_colonne(
                    paquet,
                    float(latitude),
                    float(longitude),
                    jour_annee=jour_annee,
                )
                nouvelle_temperature, diagnostic = _pas_implicite_cellule(
                    colonne,
                    temperature[i, j],
                    capacite[i, j],
                    jour_annee,
                    heure_solaire,
                    config,
                )
                prochaine_temperature[i, j] = nouvelle_temperature
                for nom, valeur in diagnostic.items():
                    diagnostics_somme[nom][i, j] += valeur

        temperature = prochaine_temperature
        if (pas + 1) % config.frequence_sortie_pas == 0 or pas == nombre_pas - 1:
            sorties_temperature.append(temperature.copy())
            temps_sortie.append((pas + 1) * config.dt_s)
        if config.afficher_progression:
            _afficher_progression(pas + 1, nombre_pas, unite="pas")

    diagnostics_moyens = {
        nom: valeurs / nombre_pas for nom, valeurs in diagnostics_somme.items()
    }

    return {
        "temps_s": np.array(temps_sortie, dtype=np.float64),
        "temperature_surface_k": np.stack(sorties_temperature).astype(np.float32),
        "lat_deg": np.array(latitudes, dtype=np.float64),
        "lon_deg": np.array(longitudes, dtype=np.float64),
        "capacite_surface_j_m2_k": capacite.astype(np.float32),
        "diagnostics_moyens": diagnostics_moyens,
        "metadata": {
            "modele": "modele4",
            "mode_sortie": "temporel",
            "jours": config.jours,
            "dt_s": config.dt_s,
            "co2_ppm": config.co2_ppm,
            "frequence_sortie_pas": config.frequence_sortie_pas,
            "iterations_implicites": config.iterations_implicites,
            "mode_convection": config.surface.mode_convection,
            "facteur_latent": config.surface.facteur_latent,
            "vent_m_s": config.surface.vent_m_s,
            "afficher_progression": config.afficher_progression,
            "source_paquet": str(paquet["npz_path"]),
            "source_capacite": surface.source_capacite_surface(config.rzsm_csv),
            "source_flux_latent": surface.source_flux_latent(),
            "lat_indices": list(lat_indices),
            "lon_indices": list(lon_indices),
        },
    }


def simuler_mensuel(paquet, config=None, mois=MOIS_DEFAUT):
    """Calcule 12 cartes mensuelles globales, sans integration annuelle lourde."""

    if config is None:
        config = ConfigurationModele4()
    if config.dt_s <= 0:
        raise ValueError("dt_s doit etre strictement positif.")

    donnees = paquet["donnees"]
    lat_indices = _indices_grille(
        len(donnees["lat_deg"]),
        indices=config.indices_lat,
        maximum=config.max_latitudes,
    )
    lon_indices = _indices_grille(
        len(donnees["lon_deg"]),
        indices=config.indices_lon,
        maximum=config.max_longitudes,
    )
    latitudes = donnees["lat_deg"][list(lat_indices)]
    longitudes = donnees["lon_deg"][list(lon_indices)]
    grille_rzsm = (
        surface.charger_grille_rzsm(config.rzsm_csv) if config.rzsm_csv is not None else None
    )

    mois = tuple(int(valeur) for valeur in mois)
    for valeur in mois:
        if not 1 <= valeur <= 12:
            raise ValueError("Les mois doivent etre compris entre 1 et 12.")

    shape = (len(mois), len(latitudes), len(longitudes))
    temperature = np.empty(shape, dtype=np.float64)
    capacite = np.empty(shape, dtype=np.float64)
    diagnostics = {
        "SW_absorbe_surface": np.empty(shape, dtype=np.float64),
        "LW_down_absorbe_surface": np.empty(shape, dtype=np.float64),
        "LW_up_surface": np.empty(shape, dtype=np.float64),
        "flux_latent": np.empty(shape, dtype=np.float64),
        "flux_convection": np.empty(shape, dtype=np.float64),
        "flux_net_surface": np.empty(shape, dtype=np.float64),
    }

    nombre_etapes = len(mois) * len(latitudes) * len(longitudes)
    etape = 0
    if config.afficher_progression:
        _afficher_progression(0, nombre_etapes, unite="cellules-mois")

    for indice_mois, mois_courant in enumerate(mois):
        jour_annee = physique.jour_milieu_mois(mois_courant)
        for i, latitude in enumerate(latitudes):
            for j, longitude in enumerate(longitudes):
                colonne = extraire_colonne(
                    paquet,
                    float(latitude),
                    float(longitude),
                    mois=mois_courant,
                )
                temperature_depart = _temperature_initiale(
                    colonne,
                    temperature_imposee=config.temperature_initiale_k,
                )
                rzsm = surface.rzsm_plus_proche(
                    grille_rzsm,
                    colonne["surface"]["latitude_deg"],
                    colonne["surface"]["longitude_deg"],
                )
                capacite_cellule = surface.capacite_surface(colonne["surface"], rzsm=rzsm)
                temperature_cellule, diagnostic = _pas_implicite_cellule(
                    colonne,
                    temperature_depart,
                    capacite_cellule,
                    jour_annee,
                    12.0,
                    config,
                    moyenne_journaliere_sw=True,
                )

                temperature[indice_mois, i, j] = temperature_cellule
                capacite[indice_mois, i, j] = capacite_cellule
                for nom, valeur in diagnostic.items():
                    diagnostics[nom][indice_mois, i, j] = valeur

                etape += 1
                if config.afficher_progression:
                    _afficher_progression(etape, nombre_etapes, unite="cellules-mois")

    temps_s = np.array(
        [(physique.jour_milieu_mois(mois_courant) - 1) * 86400.0 for mois_courant in mois],
        dtype=np.float64,
    )

    return {
        "mois": np.array(mois, dtype=np.int16),
        "temps_s": temps_s,
        "temperature_surface_k": temperature.astype(np.float32),
        "lat_deg": np.array(latitudes, dtype=np.float64),
        "lon_deg": np.array(longitudes, dtype=np.float64),
        "capacite_surface_j_m2_k": capacite.astype(np.float32),
        "diagnostics_moyens": diagnostics,
        "metadata": {
            "modele": "modele4",
            "mode_sortie": "mensuel",
            "mois": list(mois),
            "dt_s": config.dt_s,
            "co2_ppm": config.co2_ppm,
            "iterations_implicites": config.iterations_implicites,
            "moyenne_journaliere_sw": True,
            "mode_convection": config.surface.mode_convection,
            "facteur_latent": config.surface.facteur_latent,
            "vent_m_s": config.surface.vent_m_s,
            "afficher_progression": config.afficher_progression,
            "source_paquet": str(paquet["npz_path"]),
            "source_capacite": surface.source_capacite_surface(config.rzsm_csv),
            "source_flux_latent": surface.source_flux_latent(),
            "lat_indices": list(lat_indices),
            "lon_indices": list(lon_indices),
        },
    }


def enregistrer_resultat(resultat, chemin):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = resultat["diagnostics_moyens"]
    np.savez_compressed(
        chemin,
        temps_s=resultat["temps_s"],
        mois=resultat.get("mois", np.array([], dtype=np.int16)),
        temperature_surface_k=resultat["temperature_surface_k"],
        lat_deg=resultat["lat_deg"],
        lon_deg=resultat["lon_deg"],
        capacite_surface_j_m2_k=resultat["capacite_surface_j_m2_k"],
        sw_absorbe_surface_moyen_w_m2=diagnostics["SW_absorbe_surface"],
        lw_down_absorbe_surface_moyen_w_m2=diagnostics["LW_down_absorbe_surface"],
        lw_up_surface_moyen_w_m2=diagnostics["LW_up_surface"],
        flux_latent_moyen_w_m2=diagnostics["flux_latent"],
        flux_convection_moyen_w_m2=diagnostics["flux_convection"],
        flux_net_surface_moyen_w_m2=diagnostics["flux_net_surface"],
        metadata_json=json.dumps(resultat["metadata"], ensure_ascii=True, indent=2),
    )
    return chemin


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Modele 4 - grille de surface")
    parseur.add_argument(
        "--mode",
        choices=("mensuel", "temporel"),
        default="mensuel",
        help="mensuel par defaut: 12 cartes globales ; temporel: integration pas-a-pas.",
    )
    parseur.add_argument("--paquet", type=Path, default=DOSSIER_PAQUET_DEFAUT)
    parseur.add_argument("--output", type=Path, default=SORTIE_DEFAUT)
    parseur.add_argument("--jours", type=float, default=1.0)
    parseur.add_argument("--dt", type=float, default=1800.0)
    parseur.add_argument("--co2", type=float, default=physique.CO2_DEFAUT_PPM)
    parseur.add_argument("--temperature-initiale", type=float, default=None)
    parseur.add_argument("--frequence-sortie-pas", type=int, default=48)
    parseur.add_argument("--iterations-implicites", type=int, default=4)
    parseur.add_argument("--max-latitudes", type=int, default=None)
    parseur.add_argument("--max-longitudes", type=int, default=None)
    parseur.add_argument(
        "--rzsm-csv",
        type=Path,
        default=RZSM_MODELE0_DEFAUT,
        help=(
            "CSV RZSM du modele 0 pour la capacite surfacique. "
            f"Par defaut: {RZSM_MODELE0_DEFAUT}"
        ),
    )
    parseur.add_argument("--facteur-latent", type=float, default=1.0)
    parseur.add_argument(
        "--convection",
        choices=("aucune", "forcee", "naturelle", "toutes"),
        default="toutes",
    )
    parseur.add_argument("--vent", type=float, default=surface.VENT_DEFAUT_M_S)
    parseur.add_argument(
        "--no-progress",
        action="store_true",
        help="Desactive la barre de progression console.",
    )
    return parseur


def main():
    args = construire_parseur().parse_args()
    paquet = charger_paquet_grille(args.paquet)
    config = ConfigurationModele4(
        jours=args.jours,
        dt_s=args.dt,
        co2_ppm=args.co2,
        temperature_initiale_k=args.temperature_initiale,
        frequence_sortie_pas=args.frequence_sortie_pas,
        iterations_implicites=args.iterations_implicites,
        max_latitudes=args.max_latitudes,
        max_longitudes=args.max_longitudes,
        rzsm_csv=args.rzsm_csv,
        afficher_progression=not args.no_progress,
        surface=surface.ConfigurationSurface(
            facteur_latent=args.facteur_latent,
            mode_convection=args.convection,
            vent_m_s=args.vent,
        ),
    )
    if args.mode == "mensuel":
        resultat = simuler_mensuel(paquet, config)
    else:
        resultat = simuler(paquet, config)
    chemin = enregistrer_resultat(resultat, args.output)
    temperature = resultat["temperature_surface_k"]
    print("modele4_simulation_ok")
    print(f"sortie = {chemin}")
    print(f"mode_sortie = {resultat['metadata']['mode_sortie']}")
    print(f"shape_T_surface = {temperature.shape}")
    print(f"T_min_K = {float(np.nanmin(temperature)):.3f}")
    print(f"T_max_K = {float(np.nanmax(temperature)):.3f}")
    print(f"T_moyenne_finale_K = {float(np.nanmean(temperature[-1])):.3f}")


if __name__ == "__main__":
    main()
