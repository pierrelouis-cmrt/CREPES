"""Generateur global des donnees CREPES.

Usage courant:

    python3 outils_generation_donnees/generer_donnees.py
    python3 outils_generation_donnees/generer_donnees.py --status
    python3 outils_generation_donnees/generer_donnees.py --list
    python3 outils_generation_donnees/generer_donnees.py --run grille-lowres-rapide --force

Les grilles de temperature generees ici utilisent le moteur actuel
`modele_courbe.py`, donc la convection forcee et la convection naturelle sont
actives par defaut.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CODES_DIR = PROJECT_ROOT / "codes_python"
LOCAL_MPL_CACHE = Path(tempfile.gettempdir()) / "crepes_matplotlib"
LOCAL_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_MPL_CACHE))
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

import bibliotheque as lib  # noqa: E402
import fonctions as f  # noqa: E402
from chemins import (  # noqa: E402
    ALBEDO_DIR,
    CERES_FILE,
    HIRES_FAST_NPY,
    HIRES_NPY,
    HIRES_STABILIZED_NPY,
    LOWRES_FAST_NPY,
    LOWRES_NPY,
    LOWRES_STABILIZED_NPY,
    MONTHLY_TEMPERATURE_DIR,
    RESSOURCES_DIR,
    RZSM_CSV,
)
from modele_courbe import flux_convection, f_rhs  # noqa: E402
from physique import capacite_surface, chaleur_latente  # noqa: E402

try:
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter1d

    SCIPY_AVAILABLE = True
except ImportError:
    RegularGridInterpolator = None
    gaussian_filter1d = None
    SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


SECONDS_PER_DAY = 24 * 3600
STEPS_PER_YEAR = int(365 * SECONDS_PER_DAY / lib.dt)
DEFAULT_FAST_DAYS = 7
MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MONTH_FILES = (
    "Janvier.csv",
    "Février.csv",
    "Mars.csv",
    "Avril.csv",
    "Mai.csv",
    "Juin.csv",
    "Juillet.csv",
    "Août.csv",
    "Septembre.csv",
    "Octobre.csv",
    "Novembre.csv",
    "Décembre.csv",
)


@dataclass(frozen=True)
class GridTarget:
    identifiant: str
    titre: str
    resolution: str
    variante: str
    chemin_sortie: Path
    jours: int
    stabiliser: bool
    nlat: int | None
    nlon: int | None
    usage: str
    remarque: str


@dataclass(frozen=True)
class DataInventoryItem:
    nom: str
    statut: str
    fichiers: tuple[Path, ...]
    commande: str
    details: str


GRID_TARGETS = {
    "grille-lowres-rapide": GridTarget(
        "grille-lowres-rapide",
        "Grille rapide basse resolution",
        "lowres",
        "rapide",
        LOWRES_FAST_NPY,
        DEFAULT_FAST_DAYS,
        False,
        None,
        None,
        "planisphere basse resolution + sphere basse resolution",
        "Rapide: peu de jours, meme resolution spatiale que la grille lowres.",
    ),
    "grille-hires-rapide": GridTarget(
        "grille-hires-rapide",
        "Grille rapide haute resolution",
        "hires",
        "rapide",
        HIRES_FAST_NPY,
        DEFAULT_FAST_DAYS,
        False,
        70,
        140,
        "planisphere haute resolution + sphere haute resolution",
        "Rapide: evite le calcul haute resolution complet d'environ plusieurs dizaines de minutes.",
    ),
    "grille-lowres-1an": GridTarget(
        "grille-lowres-1an",
        "Grille annuelle basse resolution",
        "lowres",
        "1an",
        LOWRES_NPY,
        365,
        False,
        None,
        None,
        "planisphere basse resolution + sphere basse resolution",
        "Produit standard charge par defaut si present.",
    ),
    "grille-hires-1an": GridTarget(
        "grille-hires-1an",
        "Grille annuelle haute resolution",
        "hires",
        "1an",
        HIRES_NPY,
        365,
        False,
        70,
        140,
        "planisphere haute resolution + sphere haute resolution",
        "Produit standard haute resolution, potentiellement long.",
    ),
    "grille-lowres-stabilisee": GridTarget(
        "grille-lowres-stabilisee",
        "Grille stabilisee basse resolution",
        "lowres",
        "stabilisee",
        LOWRES_STABILIZED_NPY,
        365 * 2,
        True,
        None,
        None,
        "planisphere basse resolution + sphere basse resolution",
        "Calcule 2 ans puis conserve uniquement la deuxieme annee.",
    ),
    "grille-hires-stabilisee": GridTarget(
        "grille-hires-stabilisee",
        "Grille stabilisee haute resolution",
        "hires",
        "stabilisee",
        HIRES_STABILIZED_NPY,
        365 * 2,
        True,
        70,
        140,
        "planisphere haute resolution + sphere haute resolution",
        "Calcul le plus lourd: a lancer seulement quand necessaire.",
    ),
}


GROUPS = {
    "grilles-rapides": ("grille-lowres-rapide", "grille-hires-rapide"),
    "grilles-standard": ("grille-lowres-1an", "grille-hires-1an"),
    "grilles-stabilisees": (
        "grille-lowres-stabilisee",
        "grille-hires-stabilisee",
    ),
    "grilles-toutes": tuple(GRID_TARGETS),
    "tout-rapide": ("grille-lowres-rapide", "grille-hires-rapide"),
    "tout-standard": ("grille-lowres-1an", "grille-hires-1an"),
    "tout-complet": tuple(GRID_TARGETS),
    "donnees-derivees": ("temperatures-12mois",),
}


ALIASES = {
    "planisphere-lowres-rapide": "grille-lowres-rapide",
    "sphere-lowres-rapide": "grille-lowres-rapide",
    "planisphere-hires-rapide": "grille-hires-rapide",
    "sphere-hires-rapide": "grille-hires-rapide",
    "planisphere-lowres": "grille-lowres-1an",
    "sphere-lowres": "grille-lowres-1an",
    "planisphere-hires": "grille-hires-1an",
    "sphere-hires": "grille-hires-1an",
    "12-mois": "temperatures-12mois",
    "albedo-nasa": "albedo-surface-nasa",
}


DATA_TARGETS = {
    "temperatures-12mois": (
        "CSV mensuels pour affichage_3D_rapide.py",
        "Convertit une grille annuelle basse resolution generee par le moteur courant.",
    ),
    "albedo-surface-nasa": (
        "CSV mensuels d'albedo de surface",
        "Regeneration API NASA POWER sur le gabarit de coordonnees actif.",
    ),
}


DATA_INVENTORY = (
    DataInventoryItem(
        "Grilles temperature rapides",
        "GENERABLE PROPREMENT",
        (LOWRES_FAST_NPY, HIRES_FAST_NPY),
        "python3 outils_generation_donnees/generer_donnees.py --run tout-rapide --force --yes",
        "Utilise le moteur courant avec les deux convections actives par defaut.",
    ),
    DataInventoryItem(
        "Grilles temperature annuelles",
        "GENERABLE PROPREMENT",
        (LOWRES_NPY, HIRES_NPY),
        "python3 outils_generation_donnees/generer_donnees.py --run tout-standard --force --yes",
        "Calcul long, surtout en haute resolution. Les grilles existantes ne sont pas remplacees sans --force.",
    ),
    DataInventoryItem(
        "Grilles temperature stabilisees",
        "GENERABLE PROPREMENT",
        (LOWRES_STABILIZED_NPY, HIRES_STABILIZED_NPY),
        "python3 outils_generation_donnees/generer_donnees.py --run grilles-stabilisees --force --yes",
        "Calcule deux annees puis conserve la deuxieme. C'est le calcul le plus lourd.",
    ),
    DataInventoryItem(
        "Albedo surface mensuel",
        "GENERABLE PAR SCRIPT ACTUEL",
        tuple(sorted(ALBEDO_DIR.glob("albedo*.csv"))),
        "python3 outils_generation_donnees/generer_donnees.py --run albedo-surface-nasa --force --yes",
        "Appelle NASA POWER et ecrit directement dans ressources/albedo avec le format actif.",
    ),
    DataInventoryItem(
        "Temperatures mensuelles 12_mois",
        "GENERABLE PAR LE MOTEUR ACTUEL",
        tuple(sorted(MONTHLY_TEMPERATURE_DIR.glob("*.csv"))),
        "python3 outils_generation_donnees/generer_donnees.py --run temperatures-12mois --force --yes",
        "Convertit ressources/grilles/grid_lowres_1yr.npy en 12 CSV 1800x24 pour le viewer 3D rapide.",
    ),
    DataInventoryItem(
        "Humidite RZSM average_rzsm_tout.csv",
        "SOURCE EXTERNE LOCALE",
        (RZSM_CSV,),
        "Aucune commande locale fiable dans ce projet.",
        "Entree Carcajous lue par le moteur; les scripts d'affichage derives ont ete retires.",
    ),
    DataInventoryItem(
        "Albedo nuages CERES NetCDF",
        "SOURCE EXTERNE LOCALE",
        (CERES_FILE,),
        "Telecharger/remplacer manuellement le NetCDF source CERES.",
        "Le moteur lit ce fichier, mais aucun script local ne le reconstruit.",
    ),
    DataInventoryItem(
        "Shapefiles carte et cotes",
        "SOURCE EXTERNE LOCALE",
        (
            RESSOURCES_DIR / "carte" / "ne_110m_admin_0_countries.shp",
            RESSOURCES_DIR / "cotes" / "ne_10m_coastline.shp",
        ),
        "Remplacer manuellement les fichiers Natural Earth si besoin.",
        "Ils servent aux continents et aux contours. Aucun script local ne les telecharge ou reconstruit.",
    ),
)


def _progress(iterable, desc):
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc)
    print(desc)
    return iterable


def _taille_humaine(path: Path) -> str:
    if not path.exists():
        return "-"
    size = path.stat().st_size
    for unit in ("o", "Ko", "Mo", "Go"):
        if size < 1024 or unit == "Go":
            return f"{size:.1f} {unit}" if unit != "o" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} Go"


def _date_humaine(path: Path) -> str:
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _chemin_affiche(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _shape_npy(path: Path) -> str:
    if not path.exists():
        return "-"
    try:
        return "x".join(str(v) for v in np.load(path, mmap_mode="r").shape)
    except Exception as exc:
        return f"illisible ({exc})"


def _print_section(title: str):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _format_paths(paths: tuple[Path, ...]) -> str:
    if not paths:
        return "-"
    formatted = []
    for path in paths:
        if path.exists():
            formatted.append(f"{_chemin_affiche(path)} OK")
        else:
            formatted.append(f"{_chemin_affiche(path)} MANQUANT")
    return "; ".join(formatted)


def afficher_inventaire():
    _print_section("Inventaire des donnees")
    print("Statuts possibles:")
    print("- GENERABLE PROPREMENT: produit par ce script avec le moteur courant.")
    print("- GENERABLE PAR LE MOTEUR ACTUEL: derive d'une sortie moteur actuelle.")
    print("- GENERABLE PAR SCRIPT ACTUEL: produit par un script maintenu de ce dossier.")
    print("- SOURCE EXTERNE LOCALE: ressource d'entree a fournir/remplacer.")
    for item in DATA_INVENTORY:
        print(f"\n- {item.nom}")
        print(f"  statut: {item.statut}")
        print(f"  fichiers: {_format_paths(item.fichiers)}")
        print(f"  commande: {item.commande}")
        print(f"  details: {item.details}")


def afficher_liste():
    _print_section("Groupes de generation")
    for identifiant, membres in GROUPS.items():
        print(f"- {identifiant}: {', '.join(membres)}")

    _print_section("Grilles actives")
    for target in GRID_TARGETS.values():
        print(f"- {target.identifiant}")
        print(f"  {target.titre}")
        print(f"  sortie: {target.chemin_sortie.relative_to(PROJECT_ROOT)}")
        print(f"  usage: {target.usage}")
        print(f"  note: {target.remarque}")

    _print_section("Alias pratiques")
    for alias, cible in ALIASES.items():
        print(f"- {alias} -> {cible}")

    _print_section("Donnees actives hors grilles")
    for identifiant, (titre, note) in DATA_TARGETS.items():
        print(f"- {identifiant}: {titre}")
        print(f"  note: {note}")

    afficher_inventaire()


def afficher_statut():
    afficher_inventaire()

    _print_section("Ressources statiques")
    checks = [
        ("Albedo CERES NetCDF", CERES_FILE),
        ("Humidite RZSM", RZSM_CSV),
    ]
    for label, path in checks:
        print(
            f"- {label}: {'OK' if path.exists() else 'MANQUANT'} | "
            f"{path.relative_to(PROJECT_ROOT)} | {_taille_humaine(path)}"
        )

    albedo_csv = sorted(ALBEDO_DIR.glob("albedo*.csv"))
    mois_csv = sorted(MONTHLY_TEMPERATURE_DIR.glob("*.csv"))
    print(f"- CSV albedo mensuels: {len(albedo_csv)}/12")
    print(f"- CSV temperatures 12_mois: {len(mois_csv)}/12")

    _print_section("Grilles de temperature")
    for target in GRID_TARGETS.values():
        path = target.chemin_sortie
        status = "OK" if path.exists() else "MANQUANT"
        print(
            f"- {target.identifiant}: {status} | "
            f"{path.relative_to(PROJECT_ROOT)} | "
            f"shape={_shape_npy(path)} | {_taille_humaine(path)} | {_date_humaine(path)}"
        )
        meta = _metadata_path(path)
        if meta.exists():
            print(f"  metadata: {meta.relative_to(PROJECT_ROOT)}")


def _resolve_names(names: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_name in names:
        name = ALIASES.get(raw_name, raw_name)
        if name in GROUPS:
            resolved.extend(_resolve_names(list(GROUPS[name])))
        elif name in GRID_TARGETS or name in DATA_TARGETS:
            resolved.append(name)
        else:
            raise ValueError(f"Cible inconnue: {raw_name}")
    return list(dict.fromkeys(resolved))


def _normaliser_longitudes(longitudes: np.ndarray) -> np.ndarray:
    return ((longitudes + 180) % 360) - 180


def _interpoler_cube(
    cube_source: np.ndarray,
    lat_source: np.ndarray,
    lon_source: np.ndarray,
    lat_cible: np.ndarray,
    lon_cible: np.ndarray,
) -> np.ndarray:
    if not SCIPY_AVAILABLE:
        raise RuntimeError("scipy est requis pour generer les grilles haute resolution.")

    lat_order = np.argsort(lat_source)
    lon_order = np.argsort(lon_source)
    lat_sorted = lat_source[lat_order]
    lon_sorted = lon_source[lon_order]
    cube_sorted = cube_source[:, lat_order, :][:, :, lon_order]
    points = np.array(np.meshgrid(lat_cible, lon_cible, indexing="ij"))
    points = np.moveaxis(points, 0, -1)
    result = np.empty((cube_source.shape[0], len(lat_cible), len(lon_cible)))

    for month in range(cube_source.shape[0]):
        interpolateur = RegularGridInterpolator(
            (lat_sorted, lon_sorted),
            cube_sorted[month],
            bounds_error=False,
            fill_value=None,
        )
        result[month] = interpolateur(points)
    return result


def _charger_nuages_mensuels(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    cloud_map = f.load_monthly_cloud_albedo_from_ceres(
        lat_deg=None,
        lon_deg=None,
        return_full_map=True,
    )
    if hasattr(cloud_map, "sel"):
        selection = cloud_map.sel(lat=latitudes, lon=longitudes, method="nearest")
        try:
            selection = selection.transpose("month", "lat", "lon")
        except ValueError:
            pass
        values = selection.to_numpy()
    else:
        values = np.asarray(cloud_map, dtype=float)

    if values.ndim == 1:
        values = np.broadcast_to(values[:, None, None], (12, len(latitudes), len(longitudes)))
    if values.shape[0] != 12:
        values = np.resize(values, (12, len(latitudes), len(longitudes)))
    return values


def _preparer_champs(target: GridTarget):
    print("--- Chargement des champs geophysiques ---")
    monthly_albedo_low, lat_low, lon_low = f.load_albedo_series(ALBEDO_DIR)

    if target.resolution == "lowres":
        latitudes = lat_low
        longitudes = lon_low
        monthly_albedo = monthly_albedo_low
    else:
        nlat = target.nlat or len(lat_low)
        nlon = target.nlon or len(lon_low)
        latitudes = np.linspace(float(lat_low.min()), float(lat_low.max()), nlat)
        longitudes = np.linspace(float(lon_low.min()), float(lon_low.max()), nlon)
        print(f"Interpolation haute resolution: {nlat} x {nlon}")
        monthly_albedo = _interpoler_cube(
            monthly_albedo_low,
            lat_low,
            lon_low,
            latitudes,
            longitudes,
        )

    monthly_cloud = _charger_nuages_mensuels(latitudes, longitudes)
    albedo_sol_daily = f.lisser_donnees_annuelles(monthly_albedo, sigma=15.0)
    albedo_nuages_daily = f.lisser_donnees_annuelles(monthly_cloud, sigma=15.0)

    C_grid = np.empty((len(latitudes), len(longitudes)), dtype=float)
    q_grid = np.empty_like(C_grid)

    for i in _progress(range(len(latitudes)), "Capacite + flux latent"):
        lat = float(latitudes[i])
        for j, lon in enumerate(longitudes):
            lon_norm = float(_normaliser_longitudes(np.array([lon]))[0])
            C_grid[i, j], _ = capacite_surface.compute_surface_capacity(lat, lon_norm)
            q_grid[i, j] = chaleur_latente.P_em_surf_evap(lat, lon_norm, verbose=False)

    print("--- Champs prets ---")
    return latitudes, longitudes, albedo_sol_daily, albedo_nuages_daily, C_grid, q_grid


def _q_latent_smoothed(q_base, days, lat_rad, lon_deg):
    step_count = int(days * SECONDS_PER_DAY / lib.dt)
    sign_daynight = np.empty(step_count)
    for index in range(step_count):
        t_sec = index * lib.dt
        jour, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        sign_daynight[index] = (
            1.0 if f.cos_incidence(lat_rad, jour + 1, heure_solaire) > 0 else -1.0
        )
    return gaussian_filter1d(q_base * sign_daynight, sigma=3.0, mode="wrap")


def _integrer_point(
    days,
    lat_deg,
    lon_deg,
    alb_sol_daily,
    alb_nuages_daily,
    C_const,
    q_base,
    mode_convection,
    temperature_air,
    vent,
    vent_api,
):
    lat_rad = np.radians(lat_deg)
    step_count = int(days * SECONDS_PER_DAY / lib.dt)
    T = np.empty(step_count + 1)
    T[0] = 288.15 - 30 * np.sin(lat_rad) ** 2
    q_latent_smoothed = _q_latent_smoothed(q_base, days, lat_rad, lon_deg)

    sim_params = {"convection": {"mode": mode_convection}}
    if mode_convection != "aucune":
        sim_params["convection"] = {
            "mode": mode_convection,
            "temperature_air": temperature_air,
            "vent": None if vent_api else vent,
        }
        if vent_api and mode_convection in ("forcee", "toutes"):
            from physique import convection

            sim_params["convection"]["vents_journaliers"] = convection.get_daily_wind_speed(
                lat_deg,
                lon_deg,
            )

    for index in range(step_count):
        t_sec = index * lib.dt
        day_of_year, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        phi_n = lib.P_inc_solar(
            lat_rad,
            day_of_year + 1,
            heure_solaire,
            alb_sol_daily[day_of_year],
            alb_nuages_daily[day_of_year],
        )

        X = T[index]
        for _ in range(8):
            q_conv = flux_convection(X, index, sim_params)
            F = X - T[index] - lib.dt * f_rhs(
                X,
                phi_n,
                C_const,
                q_latent_smoothed[index],
                q_conv,
            )
            if mode_convection == "aucune":
                dF = 1.0 - lib.dt * (-4.0 * lib.sigma * X**3 / C_const)
            else:
                eps = max(1e-4, abs(X) * 1e-6)
                q_plus = flux_convection(X + eps, index, sim_params)
                q_minus = flux_convection(X - eps, index, sim_params)
                rhs_plus = f_rhs(
                    X + eps,
                    phi_n,
                    C_const,
                    q_latent_smoothed[index],
                    q_plus,
                )
                rhs_minus = f_rhs(
                    X - eps,
                    phi_n,
                    C_const,
                    q_latent_smoothed[index],
                    q_minus,
                )
                dF = 1.0 - lib.dt * ((rhs_plus - rhs_minus) / (2 * eps))
            if abs(dF) < 1e-12:
                break
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[index + 1] = X
    return T


def _ecrire_metadata(path: Path, target: GridTarget, shape, args, secondes):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "outils_generation_donnees/generer_donnees.py",
        "target": target.identifiant,
        "title": target.titre,
        "output": _chemin_affiche(path),
        "shape": list(shape),
        "dtype": str(np.dtype(args.dtype)),
        "dt_seconds": lib.dt,
        "days_computed": target.jours,
        "stabilized_second_year_only": target.stabiliser,
        "convection": {
            "mode": "aucune" if args.sans_convection else args.convection,
            "temperature_air_K": args.temperature_air,
            "wind_m_s": None if args.vent_api else args.vent,
            "wind_api": args.vent_api,
        },
        "engine": "modele_courbe.f_rhs + modele_courbe.flux_convection",
        "usage": target.usage,
        "elapsed_seconds": round(secondes, 2),
    }
    with _metadata_path(path).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def _target_with_fast_days(target: GridTarget, fast_days: int) -> GridTarget:
    if target.variante != "rapide" or target.jours == fast_days:
        return target
    return GridTarget(
        target.identifiant,
        target.titre,
        target.resolution,
        target.variante,
        target.chemin_sortie,
        fast_days,
        target.stabiliser,
        target.nlat,
        target.nlon,
        target.usage,
        target.remarque,
    )


def generer_grille(target: GridTarget, args):
    target = _target_with_fast_days(target, args.fast_days)
    output_dir = Path(args.output_dir) if args.output_dir else target.chemin_sortie.parent
    output_path = output_dir / target.chemin_sortie.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force and not args.dry_run:
        if not args.yes:
            reponse = input(
                f"{output_path} existe deja. Le remplacer ? [o/N] "
            ).strip().lower()
            if reponse not in ("o", "oui", "y", "yes"):
                print("Generation annulee pour cette cible.")
                return
        else:
            raise FileExistsError(
                f"{output_path} existe deja. Ajouter --force pour remplacer."
            )

    mode_convection = "aucune" if args.sans_convection else args.convection
    if args.vent_api and mode_convection in ("forcee", "toutes") and not args.yes:
        reponse = input(
            "Le mode --vent-api peut lancer beaucoup d'appels reseau. Continuer ? [o/N] "
        ).strip().lower()
        if reponse not in ("o", "oui", "y", "yes"):
            print("Generation annulee.")
            return

    print(f"\nCible: {target.identifiant}")
    print(f"Sortie: {output_path}")
    print(f"Convection: {mode_convection}")
    print(f"Duree calculee: {target.jours} jours")
    if args.dry_run:
        print("DRY-RUN: aucun fichier ne sera ecrit.")
        return

    started = time.time()
    (
        latitudes,
        longitudes,
        albedo_sol_daily,
        albedo_nuages_daily,
        C_grid,
        q_grid,
    ) = _preparer_champs(target)

    total_steps = int(target.jours * SECONDS_PER_DAY / lib.dt) + 1
    save_start = STEPS_PER_YEAR if target.stabiliser else 0
    save_steps = total_steps - save_start
    shape = (save_steps, len(latitudes), len(longitudes))
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    grid = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.dtype(args.dtype),
        shape=shape,
    )

    for i in _progress(range(len(latitudes)), "Simulation grille"):
        lat = float(latitudes[i])
        for j, lon in enumerate(longitudes):
            lon_norm = float(_normaliser_longitudes(np.array([lon]))[0])
            serie = _integrer_point(
                target.jours,
                lat,
                lon_norm,
                albedo_sol_daily[:, i, j],
                albedo_nuages_daily[:, i, j],
                C_grid[i, j],
                q_grid[i, j],
                mode_convection,
                args.temperature_air,
                args.vent,
                args.vent_api,
            )
            grid[:, i, j] = serie[save_start:]

    grid.flush()
    del grid
    tmp_path.replace(output_path)
    elapsed = time.time() - started
    _ecrire_metadata(output_path, target, shape, args, elapsed)
    print(f"Termine: {output_path} ({_taille_humaine(output_path)})")
    print(f"Temps ecoule: {elapsed / 60:.1f} min")


def _month_midday_indices():
    start_day = 0
    indices = []
    steps_per_day = int(SECONDS_PER_DAY / lib.dt)
    steps_per_hour = int(3600 / lib.dt)
    for month_length in MONTH_LENGTHS:
        mid_day = start_day + month_length // 2
        hour_indices = [
            mid_day * steps_per_day + hour * steps_per_hour
            for hour in range(24)
        ]
        indices.append(hour_indices)
        start_day += month_length
    return indices


def _dedupe_longitudes(longitudes):
    rounded = np.round(np.asarray(longitudes, dtype=float), 8)
    order = np.argsort(rounded)
    sorted_lons = rounded[order]
    unique_lons, first_indices = np.unique(sorted_lons, return_index=True)
    selected = order[first_indices]
    return unique_lons, selected


def _resample_frame_for_12mois(frame, source_lats, source_lons):
    target_lons = np.linspace(-180.0, 180.0, 60)
    unique_lons, lon_indices = _dedupe_longitudes(source_lons)
    frame_unique = frame[:, lon_indices]
    by_lon = np.empty((frame_unique.shape[0], len(target_lons)), dtype=float)
    for row_index, row in enumerate(frame_unique):
        by_lon[row_index] = np.interp(target_lons, unique_lons, row)

    source_lats = np.asarray(source_lats, dtype=float)
    if source_lats[0] < source_lats[-1]:
        by_lon = by_lon[::-1]
    return by_lon


def generer_temperatures_12mois(args):
    source_grid = Path(args.monthly_source_grid) if args.monthly_source_grid else LOWRES_NPY
    output_dir = Path(args.output_dir) if args.output_dir else MONTHLY_TEMPERATURE_DIR
    output_files = [output_dir / filename for filename in MONTH_FILES]

    if not source_grid.exists():
        raise FileNotFoundError(
            f"Grille source manquante: {_chemin_affiche(source_grid)}. "
            "Generer d'abord --run grille-lowres-1an."
        )
    existing = [path for path in output_files if path.exists()]
    if existing and not args.force and not args.dry_run:
        if args.yes:
            raise FileExistsError(
                "Des CSV 12_mois existent deja. Ajouter --force pour remplacer."
            )
        reponse = input(f"{len(existing)} CSV 12_mois existent deja. Remplacer ? [o/N] ")
        if reponse.strip().lower() not in ("o", "oui", "y", "yes"):
            print("Generation annulee.")
            return

    _, source_lats, source_lons = f.load_albedo_series(ALBEDO_DIR)
    grid = np.load(source_grid, mmap_mode="r")
    if grid.ndim != 3:
        raise ValueError(f"Grille source invalide: shape={grid.shape}")
    if grid.shape[1] != len(source_lats) or grid.shape[2] != len(source_lons):
        raise ValueError(
            "La grille source doit avoir la resolution basse active "
            f"({len(source_lats)}x{len(source_lons)}), pas {grid.shape[1]}x{grid.shape[2]}."
        )

    indices_by_month = _month_midday_indices()
    max_index = max(max(indices) for indices in indices_by_month)
    if grid.shape[0] <= max_index:
        raise ValueError(
            f"La grille source ne couvre pas un an complet: {grid.shape[0]} pas, "
            f"il en faut au moins {max_index + 1}."
        )

    print("\nCible: temperatures-12mois")
    print(f"Source: {_chemin_affiche(source_grid)}")
    print(f"Sortie: {_chemin_affiche(output_dir)}")
    if args.dry_run:
        for path in output_files:
            print(f"DRY-RUN: ecriture {_chemin_affiche(path)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, hour_indices in zip(MONTH_FILES, indices_by_month):
        matrix = np.empty((30 * 60, 24), dtype=float)
        for hour, grid_index in enumerate(hour_indices):
            frame = _resample_frame_for_12mois(
                np.asarray(grid[grid_index], dtype=float),
                source_lats,
                source_lons,
            )
            matrix[:, hour] = frame.reshape(-1)
        path = output_dir / filename
        np.savetxt(path, matrix, delimiter=",", fmt="%.6f")
        print(f"Ecrit: {_chemin_affiche(path)}")


def generer_albedo_surface_nasa(args):
    script = SCRIPT_DIR / "albedo" / "generer_albedo_surface.py"
    output_dir = Path(args.output_dir) if args.output_dir else ALBEDO_DIR
    command = [
        sys.executable,
        str(script),
        "--year",
        str(args.albedo_year),
        "--output-dir",
        str(output_dir),
        "--sleep",
        str(args.api_sleep),
        "--timeout",
        str(args.api_timeout),
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if args.yes:
        command.append("--yes")
    print("\nCible: albedo-surface-nasa", flush=True)
    print("Commande: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def executer_cibles(names: list[str], args):
    for name in _resolve_names(names):
        if name in GRID_TARGETS:
            generer_grille(GRID_TARGETS[name], args)
        elif name == "temperatures-12mois":
            generer_temperatures_12mois(args)
        elif name == "albedo-surface-nasa":
            generer_albedo_surface_nasa(args)


def _choisir_cibles_interactif():
    print("\nChoix disponibles:")
    print("  1. Tout rapide: lowres + hires rapides")
    print("  2. Tout standard: lowres + hires 1 an")
    print("  3. Tout complet actif: rapides + 1 an + stabilisees")
    print("  4. Regenerer les CSV 12_mois depuis la grille annuelle")
    print("  5. Regenerer l'albedo de surface via NASA POWER")
    print("  6. Une cible precise")
    choice = input("Votre choix: ").strip()
    if choice == "1":
        return ["tout-rapide"]
    if choice == "2":
        return ["tout-standard"]
    if choice == "3":
        return ["tout-complet"]
    if choice == "4":
        return ["temperatures-12mois"]
    if choice == "5":
        return ["albedo-surface-nasa"]
    if choice == "6":
        afficher_liste()
        raw = input("Entrer un ou plusieurs identifiants separes par des espaces: ")
        return raw.split()
    return []


def menu_interactif(args):
    while True:
        print("\nCREPES - generation globale des donnees")
        print("1. Voir l'etat des ressources")
        print("2. Voir toutes les cibles")
        print("3. Generer / regenerer")
        print("4. Quitter")
        choice = input("Choix: ").strip()
        if choice == "1":
            afficher_statut()
        elif choice == "2":
            afficher_liste()
        elif choice == "3":
            names = _choisir_cibles_interactif()
            if names:
                executer_cibles(names, args)
        elif choice == "4":
            return
        else:
            print("Choix inconnu.")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Generation globale des donnees CREPES."
    )
    parser.add_argument("--list", action="store_true", help="Affiche les cibles.")
    parser.add_argument("--status", action="store_true", help="Affiche l'etat des ressources.")
    parser.add_argument(
        "--run",
        nargs="+",
        default=None,
        help="Lance une cible ou un groupe, ex: --run tout-rapide.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Affiche sans ecrire.")
    parser.add_argument("--force", action="store_true", help="Remplace les fichiers existants.")
    parser.add_argument("--yes", action="store_true", help="Mode non interactif.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie alternatif pour tester sans toucher ressources/.",
    )
    parser.add_argument(
        "--monthly-source-grid",
        default=None,
        help="Grille annuelle basse resolution source pour temperatures-12mois.",
    )
    parser.add_argument(
        "--fast-days",
        type=int,
        default=DEFAULT_FAST_DAYS,
        help=f"Nombre de jours pour les grilles rapides. Defaut: {DEFAULT_FAST_DAYS}.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float64", "float32"],
        default="float64",
        help="Type numerique des grilles sauvegardees. Defaut: float64.",
    )
    convection_group = parser.add_mutually_exclusive_group()
    convection_group.add_argument(
        "--convection",
        choices=["toutes", "aucune", "forcee", "naturelle"],
        default="toutes",
        help="Convections actives pendant la generation. Defaut: toutes.",
    )
    convection_group.add_argument(
        "--sans-convection",
        action="store_true",
        help="Desactive les deux convections.",
    )
    parser.add_argument(
        "--temperature-air",
        type=float,
        default=288.0,
        help="Temperature d'air en K pour la convection.",
    )
    wind_group = parser.add_mutually_exclusive_group()
    wind_group.add_argument(
        "--vent",
        type=float,
        default=2.5,
        help="Vent constant en m/s pour la convection forcee. Defaut: 2.5.",
    )
    wind_group.add_argument(
        "--vent-api",
        action="store_true",
        help="Utilise le vent journalier NASA/cache. A eviter pour une grille complete.",
    )
    parser.add_argument(
        "--albedo-year",
        type=int,
        default=2023,
        help="Annee NASA POWER pour albedo-surface-nasa. Defaut: 2023.",
    )
    parser.add_argument(
        "--api-sleep",
        type=float,
        default=0.1,
        help="Pause entre appels API albedo en secondes. Defaut: 0.1.",
    )
    parser.add_argument(
        "--api-timeout",
        type=float,
        default=30.0,
        help="Timeout des appels API albedo en secondes. Defaut: 30.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    if args.fast_days <= 0:
        raise ValueError("--fast-days doit etre positif.")

    if args.list:
        afficher_liste()
    if args.status:
        afficher_statut()
    if args.run:
        executer_cibles(args.run, args)
    if not (args.list or args.status or args.run):
        menu_interactif(args)


if __name__ == "__main__":
    main()
