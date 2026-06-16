"""Cree un extrait JSON compact depuis les gros fichiers de /ressources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .donnees import charger_colonne_locale, sauvegarder_donnees_extraites
except ImportError:  # Permet aussi: python modele3/preparer_point.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modele3.donnees import charger_colonne_locale, sauvegarder_donnees_extraites


def construire_parseur() -> argparse.ArgumentParser:
    parseur = argparse.ArgumentParser(description="Prepare un point local pour Git")
    parseur.add_argument("--lat", type=float, default=48.8566)
    parseur.add_argument("--lon", type=float, default=2.3522)
    parseur.add_argument("--mois", type=int, default=7)
    parseur.add_argument(
        "--output",
        type=Path,
        default=Path("modele3/donnees_exemple/paris_2024_m07.json"),
    )
    return parseur


def main() -> None:
    args = construire_parseur().parse_args()
    donnees = charger_colonne_locale(
        lat=args.lat,
        lon=args.lon,
        mois=args.mois,
        utiliser_extrait_defaut=False,
    )
    sauvegarder_donnees_extraites(donnees, args.output)
    print(f"extrait_ecrit = {args.output}")
    print(f"source = {donnees.get('source', 'inconnue')}")


if __name__ == "__main__":
    main()
