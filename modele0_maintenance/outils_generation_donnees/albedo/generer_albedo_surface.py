"""Genere les CSV mensuels d'albedo de surface depuis NASA POWER.

Le script ecrit directement le format actif attendu par `ressources/albedo`:
un fichier `albedo01.csv` ... `albedo12.csv`, avec la grille de coordonnees
prise depuis les CSV actifs existants.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_ALBEDO_DIR = PROJECT_ROOT / "ressources" / "albedo"
MONTH_COUNT = 12
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def _load_grid_template(template_dir: Path):
    template_path = template_dir / "albedo01.csv"
    if not template_path.exists():
        raise FileNotFoundError(
            f"Gabarit introuvable: {template_path}. Fournir un dossier avec albedo01.csv."
        )
    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    return header[1:], [row[0] for row in rows]


def _valid_number(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= -900:
        return None
    return number


def _fetch_monthly_albedo(lat, lon, year, timeout):
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_UP",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }
    response = requests.get(NASA_POWER_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    params_data = payload["properties"]["parameter"]
    down = params_data["ALLSKY_SFC_SW_DWN"]
    up = params_data["ALLSKY_SFC_SW_UP"]

    month_down = {month: [] for month in range(1, MONTH_COUNT + 1)}
    month_up = {month: [] for month in range(1, MONTH_COUNT + 1)}
    for date_key, down_value in down.items():
        month = int(date_key[4:6])
        down_number = _valid_number(down_value)
        up_number = _valid_number(up.get(date_key))
        if down_number is None or up_number is None or down_number <= 0:
            continue
        month_down[month].append(down_number)
        month_up[month].append(up_number)

    values = []
    for month in range(1, MONTH_COUNT + 1):
        if not month_down[month]:
            values.append("")
            continue
        albedo = sum(month_up[month]) / sum(month_down[month])
        values.append(f"{max(0.0, min(1.0, albedo)):.6f}")
    return values


def generate_albedo_csvs(
    year: int,
    output_dir: Path,
    template_dir: Path,
    force: bool,
    dry_run: bool,
    yes: bool,
    sleep_seconds: float,
    timeout: float,
):
    longitudes, latitudes = _load_grid_template(template_dir)
    output_files = [output_dir / f"albedo{month:02d}.csv" for month in range(1, 13)]
    existing = [path for path in output_files if path.exists()]
    if existing and not force and not dry_run:
        if yes:
            raise FileExistsError("Des CSV albedo existent deja. Ajouter --force.")
        answer = input(f"{len(existing)} CSV albedo existent deja. Remplacer ? [o/N] ")
        if answer.strip().lower() not in ("o", "oui", "y", "yes"):
            print("Generation annulee.")
            return

    cell_count = len(latitudes) * len(longitudes)
    print(f"Grille albedo: {len(latitudes)} latitudes x {len(longitudes)} longitudes")
    print(f"Annee NASA POWER: {year}")
    print(f"Sortie: {output_dir}")
    if dry_run:
        print(f"DRY-RUN: {cell_count} appels NASA POWER seraient effectues.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        [["Latitude/Longitude"] + longitudes] + [[lat] + [""] * len(longitudes) for lat in latitudes]
        for _ in range(MONTH_COUNT)
    ]

    total = cell_count
    done = 0
    for row_index, lat_text in enumerate(latitudes, start=1):
        lat = float(lat_text)
        for col_index, lon_text in enumerate(longitudes, start=1):
            lon = float(lon_text)
            monthly_values = _fetch_monthly_albedo(lat, lon, year, timeout)
            for month_index, value in enumerate(monthly_values):
                tables[month_index][row_index][col_index] = value
            done += 1
            if done % 25 == 0 or done == total:
                print(f"Progression albedo: {done}/{total}")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    for month, path in enumerate(output_files, start=1):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerows(tables[month - 1])
        print(f"Ecrit: {path.relative_to(PROJECT_ROOT)}")


def _build_parser():
    parser = argparse.ArgumentParser(description="Genere les CSV albedo depuis NASA POWER.")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--output-dir", type=Path, default=ACTIVE_ALBEDO_DIR)
    parser.add_argument("--template-dir", type=Path, default=ACTIVE_ALBEDO_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser


def main():
    args = _build_parser().parse_args()
    generate_albedo_csvs(
        args.year,
        args.output_dir,
        args.template_dir,
        args.force,
        args.dry_run,
        args.yes,
        args.sleep,
        args.timeout,
    )


if __name__ == "__main__":
    main()
