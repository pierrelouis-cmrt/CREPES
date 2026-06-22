"""Modele 4 rapide : integration vectorisee de la grille de surface.

Ce script garde le modele 3 comme source radiative de reference pour les
precalculs mensuels, puis fait evoluer toute la grille avec des tableaux numpy.
La sortie par defaut est une carte toutes les 4 heures.
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
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.codes_python.modele3 import calculer_colonne_radiative
except ImportError:  # Permet aussi : python modele4/rapide.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modele3.codes_python import physique
    from modele3.codes_python.donnees import DOSSIER_PAQUET_DEFAUT, charger_paquet_grille, extraire_colonne
    from modele3.codes_python.modele3 import calculer_colonne_radiative

try:
    from . import surface
    from .modele4 import RZSM_MODELE0_DEFAUT
except ImportError:  # Permet aussi : python modele4/rapide.py
    from modele4.codes_python import surface
    from modele4.codes_python.modele4 import RZSM_MODELE0_DEFAUT


SORTIE_DEFAUT = Path(__file__).resolve().parents[1] / "sorties" / "simulation_modele4_rapide.npz"


@dataclass
class ConfigurationRapide:
    jours: float = 1.0
    dt_s: float = 1800.0
    sortie_heures: float = 4.0
    co2_ppm: float = physique.CO2_DEFAUT_PPM
    temperature_initiale_k: float | None = None
    indices_lat: tuple[int, ...] | None = None
    indices_lon: tuple[int, ...] | None = None
    max_latitudes: int | None = None
    max_longitudes: int | None = None
    rzsm_csv: Path | None = RZSM_MODELE0_DEFAUT
    facteur_latent: float = 1.0
    mode_convection: str = "toutes"
    vent_m_s: float = surface.VENT_DEFAUT_M_S
    temperature_air_defaut_k: float = surface.TEMPERATURE_AIR_DEFAUT_K
    afficher_progression: bool = True


def _indices_grille(taille, indices=None, maximum=None):
    if indices is not None:
        selection = tuple(int(indice) for indice in indices)
    elif maximum is None:
        selection = tuple(range(taille))
    else:
        maximum = max(1, min(int(maximum), taille))
        selection = tuple(range(maximum))

    for indice in selection:
        if indice < 0 or indice >= taille:
            raise ValueError(f"Indice de grille hors limites: {indice} pour taille {taille}")
    return selection


def _afficher_progression(etapes_effectuees, nombre_etapes, unite, largeur=32):
    fraction = etapes_effectuees / max(1, nombre_etapes)
    remplis = int(round(largeur * fraction))
    barre = "#" * remplis + "-" * (largeur - remplis)
    sys.stderr.write(
        f"\rmodele4 rapide [{barre}] {100.0 * fraction:6.2f}% "
        f"({etapes_effectuees}/{nombre_etapes} {unite})"
    )
    if etapes_effectuees >= nombre_etapes:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _mois_utiles(nombre_pas, dt_s):
    mois = set()
    for pas in range(nombre_pas):
        jour_annee = int((pas * dt_s) // 86400.0) % 365 + 1
        mois.add(physique.mois_depuis_jour_annee(jour_annee))
    return tuple(sorted(mois))


def _temperature_initiale(colonne, temperature_imposee=None):
    if temperature_imposee is not None:
        return float(temperature_imposee)
    surface_colonne = colonne["surface"]
    for nom in ("skin_temperature_k", "temperature_2m_k"):
        valeur = surface_colonne.get(nom)
        if valeur is not None and np.isfinite(valeur):
            return float(valeur)
    return physique.TEMPERATURE_SURFACE_DEFAUT_K


def _cosinus_solaire_grille(latitudes, longitudes, jour_annee, t_sec):
    lat_rad = np.radians(latitudes)[:, None]
    declinaison = physique.declinaison_solaire_rad(jour_annee)
    heure_solaire = ((t_sec / 3600.0) + longitudes[None, :] / 15.0) % 24.0
    angle_horaire = np.radians(15.0 * (heure_solaire - 12.0))
    cosinus = (
        np.sin(lat_rad) * np.sin(declinaison)
        + np.cos(lat_rad) * np.cos(declinaison) * np.cos(angle_horaire)
    )
    return np.maximum(cosinus, 0.0)


def _coefficient_convection_forcee(vent_m_s):
    return surface.coefficient_convection_forcee(vent_m_s)


def _flux_convection_vectoriel(temperature_surface, temperature_air, config):
    mode = config.mode_convection
    if mode == "aucune":
        zeros = np.zeros_like(temperature_surface)
        return zeros, zeros
    if mode not in {"forcee", "naturelle", "toutes"}:
        raise ValueError(f"Mode de convection inconnu: {mode}")

    delta_t = temperature_surface - temperature_air
    flux = np.zeros_like(temperature_surface)
    h_total = np.zeros_like(temperature_surface)

    if mode in {"forcee", "toutes"}:
        h_forcee = _coefficient_convection_forcee(config.vent_m_s)
        flux += h_forcee * delta_t
        h_total += h_forcee

    if mode in {"naturelle", "toutes"}:
        lam = 0.026
        longueur = 0.05
        g = 9.81
        nu = 1.5e-5
        alpha = 2e-5
        beta = 1.0 / np.maximum(temperature_air, 1.0)
        prandtl = nu / alpha
        coeff = np.where(delta_t >= 0.0, 0.54, 0.27)
        grashof = g * beta * delta_t * longueur**3 / nu**2
        rayleigh = grashof * prandtl
        nusselt = coeff * np.abs(rayleigh) ** 0.25
        h_naturelle = nusselt * lam / longueur
        flux += h_naturelle * delta_t
        h_total += h_naturelle

    return flux, h_total


def _precalculer_champs(paquet, config, latitudes, longitudes, mois_utiles):
    # Le rapide prepare les champs lents une fois pour eviter de rappeler le modele 3.
    grille_rzsm = (
        surface.charger_grille_rzsm(config.rzsm_csv) if config.rzsm_csv is not None else None
    )
    shape = (len(mois_utiles), len(latitudes), len(longitudes))
    champs = {
        "albedo": np.empty(shape, dtype=np.float64),
        "tau_sw": np.empty(shape, dtype=np.float64),
        "lw_down_absorbe": np.empty(shape, dtype=np.float64),
        "temperature_air": np.empty(shape, dtype=np.float64),
        "flux_latent": np.empty(shape, dtype=np.float64),
        "capacite": np.empty(shape, dtype=np.float64),
        "temperature_initiale": np.empty((len(latitudes), len(longitudes)), dtype=np.float64),
    }

    total = len(mois_utiles) * len(latitudes) * len(longitudes)
    etape = 0
    if config.afficher_progression:
        _afficher_progression(0, total, "precalculs")

    config_surface = surface.ConfigurationSurface(
        facteur_latent=config.facteur_latent,
        mode_convection=config.mode_convection,
        vent_m_s=config.vent_m_s,
        temperature_air_defaut_k=config.temperature_air_defaut_k,
    )

    for indice_mois, mois in enumerate(mois_utiles):
        for i, latitude in enumerate(latitudes):
            for j, longitude in enumerate(longitudes):
                colonne = extraire_colonne(paquet, float(latitude), float(longitude), mois=mois)
                surface_colonne = colonne["surface"]
                radiatif = calculer_colonne_radiative(
                    colonne,
                    temperature_surface_k=_temperature_initiale(
                        colonne,
                        config.temperature_initiale_k,
                    ),
                    co2_ppm=config.co2_ppm,
                    jour_annee=physique.jour_milieu_mois(mois),
                    heure_solaire=12.0,
                    moyenne_journaliere_sw=True,
                )
                rzsm = surface.rzsm_plus_proche(
                    grille_rzsm,
                    surface_colonne["latitude_deg"],
                    surface_colonne["longitude_deg"],
                )
                champs["albedo"][indice_mois, i, j] = surface.fraction(
                    surface_colonne.get("albedo_surface"),
                    defaut=0.30,
                )
                champs["tau_sw"][indice_mois, i, j] = surface.fraction(
                    surface_colonne.get("transmissivite_sw_mensuelle"),
                    defaut=0.0,
                )
                champs["lw_down_absorbe"][indice_mois, i, j] = radiatif[
                    "LW_down_absorbe_surface"
                ]
                champs["temperature_air"][indice_mois, i, j] = surface.temperature_air(
                    surface_colonne,
                    config_surface,
                )
                champs["flux_latent"][indice_mois, i, j] = surface.flux_latent_moyen(
                    surface_colonne,
                    facteur=config.facteur_latent,
                )
                champs["capacite"][indice_mois, i, j] = surface.capacite_surface(
                    surface_colonne,
                    rzsm=rzsm,
                )
                if indice_mois == 0:
                    champs["temperature_initiale"][i, j] = _temperature_initiale(
                        colonne,
                        config.temperature_initiale_k,
                    )

                etape += 1
                if config.afficher_progression:
                    _afficher_progression(etape, total, "precalculs")

    champs["mois_utiles"] = np.array(mois_utiles, dtype=np.int16)
    return champs


def simuler_rapide(paquet, config=None):
    if config is None:
        config = ConfigurationRapide()
    if config.jours <= 0:
        raise ValueError("jours doit etre strictement positif.")
    if config.dt_s <= 0:
        raise ValueError("dt_s doit etre strictement positif.")
    if config.sortie_heures <= 0:
        raise ValueError("sortie_heures doit etre strictement positif.")

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
    latitudes = np.asarray(donnees["lat_deg"][list(lat_indices)], dtype=np.float64)
    longitudes = np.asarray(donnees["lon_deg"][list(lon_indices)], dtype=np.float64)

    nombre_pas = max(1, int(round(config.jours * 86400.0 / config.dt_s)))
    sortie_pas = max(1, int(round(config.sortie_heures * 3600.0 / config.dt_s)))
    mois_utiles = _mois_utiles(nombre_pas, config.dt_s)
    index_mois = {mois: indice for indice, mois in enumerate(mois_utiles)}

    champs = _precalculer_champs(paquet, config, latitudes, longitudes, mois_utiles)
    temperature = champs["temperature_initiale"].copy()
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
        "flux_net_surface": np.zeros_like(temperature),
    }

    if config.afficher_progression:
        _afficher_progression(0, nombre_pas, "pas")

    # A partir d'ici, toute la grille avance ensemble avec des tableaux numpy.
    for pas in range(nombre_pas):
        t_sec = pas * config.dt_s
        jour_annee = int(t_sec // 86400.0) % 365 + 1
        mois = physique.mois_depuis_jour_annee(jour_annee)
        indice_mois = index_mois[mois]

        albedo = champs["albedo"][indice_mois]
        tau_sw = champs["tau_sw"][indice_mois]
        lw_down = champs["lw_down_absorbe"][indice_mois]
        temperature_air = champs["temperature_air"][indice_mois]
        flux_latent = champs["flux_latent"][indice_mois]
        capacite = champs["capacite"][indice_mois]

        cosinus = _cosinus_solaire_grille(latitudes, longitudes, jour_annee, t_sec)
        sw_absorbe = physique.CONSTANTE_SOLAIRE * cosinus * tau_sw * (1.0 - albedo)
        lw_up = physique.flux_lw_surface(temperature, physique.EMISSIVITE_SURFACE_CONSTANTE)
        flux_convection, h_convection = _flux_convection_vectoriel(
            temperature,
            temperature_air,
            config,
        )
        flux_net = sw_absorbe + lw_down - lw_up - flux_latent - flux_convection

        d_lw = (
            4.0
            * physique.EMISSIVITE_SURFACE_CONSTANTE
            * physique.SIGMA
            * np.maximum(temperature, 1.0) ** 3
        )
        # Le pas semi-implicite amortit LW montant et convection dans le meme calcul.
        delta_t = config.dt_s * flux_net / (capacite + config.dt_s * (d_lw + h_convection))
        temperature = temperature + delta_t
        if not np.isfinite(temperature).all():
            raise FloatingPointError("Temperature non finie dans le modele 4 rapide.")

        diagnostics_somme["SW_absorbe_surface"] += sw_absorbe
        diagnostics_somme["LW_down_absorbe_surface"] += lw_down
        diagnostics_somme["LW_up_surface"] += lw_up
        diagnostics_somme["flux_latent"] += flux_latent
        diagnostics_somme["flux_convection"] += flux_convection
        diagnostics_somme["flux_net_surface"] += flux_net

        if (pas + 1) % sortie_pas == 0 or pas == nombre_pas - 1:
            t_sortie = (pas + 1) * config.dt_s
            sorties_temperature.append(temperature.astype(np.float32))
            temps_sortie.append(t_sortie)
            jours_sortie.append(t_sortie / 86400.0 + 1.0)
            heures_sortie.append((t_sortie / 3600.0) % 24.0)

        if config.afficher_progression:
            _afficher_progression(pas + 1, nombre_pas, "pas")

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
        "mois_precalcules": champs["mois_utiles"],
        "capacite_surface_j_m2_k": champs["capacite"].astype(np.float32),
        "diagnostics_moyens": diagnostics_moyens,
        "metadata": {
            "modele": "modele4_rapide",
            "description": "integration vectorisee, precalculs mensuels via modele 3",
            "etat_provenance": "courant",
            "artefact_obsolete": False,
            "jours": config.jours,
            "dt_s": config.dt_s,
            "sortie_heures": config.sortie_heures,
            "co2_ppm": config.co2_ppm,
            "mode_convection": config.mode_convection,
            "facteur_latent": config.facteur_latent,
            "vent_m_s": config.vent_m_s,
            "temperature_air_defaut_k": config.temperature_air_defaut_k,
            "source_paquet": str(paquet["npz_path"]),
            "source_capacite": surface.source_capacite_surface(config.rzsm_csv),
            "source_flux_latent": surface.source_flux_latent(),
            "statut_flux_latent": surface.STATUT_FLUX_LATENT,
            "lat_indices": list(lat_indices),
            "lon_indices": list(lon_indices),
            "mois_precalcules": [int(mois) for mois in champs["mois_utiles"]],
            "schema_temperature": "semi-implicite lineaire LW_up + convection",
        },
    }


def enregistrer_resultat(resultat, chemin):
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
        flux_net_surface_moyen_w_m2=diagnostics["flux_net_surface"],
        metadata_json=json.dumps(resultat["metadata"], ensure_ascii=True, indent=2),
    )
    return chemin


def construire_parseur():
    parseur = argparse.ArgumentParser(description="Modele 4 rapide - sortie 4h par defaut")
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
    parseur.add_argument("--temperature-air", type=float, default=surface.TEMPERATURE_AIR_DEFAUT_K)
    parseur.add_argument("--no-progress", action="store_true")
    return parseur


def main():
    args = construire_parseur().parse_args()
    paquet = charger_paquet_grille(args.paquet)
    config = ConfigurationRapide(
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
        afficher_progression=not args.no_progress,
    )
    resultat = simuler_rapide(paquet, config)
    chemin = enregistrer_resultat(resultat, args.output)
    temperature = resultat["temperature_surface_k"]
    print("modele4_rapide_simulation_ok")
    print(f"sortie = {chemin}")
    print(f"shape_T_surface = {temperature.shape}")
    print(f"sortie_heures = {config.sortie_heures:.3f}")
    print(f"mois_precalcules = {resultat['mois_precalcules'].tolist()}")
    print(f"T_min_K = {float(np.nanmin(temperature)):.3f}")
    print(f"T_max_K = {float(np.nanmax(temperature)):.3f}")
    print(f"T_moyenne_finale_K = {float(np.nanmean(temperature[-1])):.3f}")


if __name__ == "__main__":
    main()
