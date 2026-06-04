"""Planisphere haute resolution CREPES.

Affichage compatible avec les resultats Carcajous haute resolution et contours
des continents traces depuis le shapefile local.
"""

import argparse
import sys
from pathlib import Path

CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from visualisation.visualisation_commune import creer_planisphere  # noqa: E402


def _build_parser():
    parser = argparse.ArgumentParser(description="Planisphère haute résolution CREPES")
    parser.add_argument("--jour", type=int, default=0)
    parser.add_argument("--heure", type=int, default=0)
    parser.add_argument(
        "--grille",
        choices=["auto", "rapide", "1an", "stabilisee"],
        default="auto",
        help="Grille a charger: auto, rapide, 1an ou stabilisee.",
    )
    parser.add_argument("--save", default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    creer_planisphere(
        preferer_haute_resolution=True,
        variante_grille=args.grille,
        jour=args.jour,
        heure=args.heure,
        afficher=not args.no_show,
        sauvegarde=args.save,
    )
