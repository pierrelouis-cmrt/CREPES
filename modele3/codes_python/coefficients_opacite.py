"""Chargement des coefficients d'opacite effectifs du modele 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DOSSIER_RESSOURCES_MODELE3 = Path(__file__).resolve().parents[1] / "ressources"
CHEMIN_COEFFICIENTS_H2O = (
    DOSSIER_RESSOURCES_MODELE3
    / "calibrage_opacite_h2o"
    / "coefficients_h2o_modele3.json"
)


def _coefficient_h2o_depuis_entree(entree: dict[str, Any]) -> tuple[str, float]:
    nom = str(entree["nom"])
    for cle in ("a_h2o_modele3", "a_h2o_hitran", "a_h2o"):
        if cle in entree:
            return nom, float(entree[cle])
    raise ValueError(f"Coefficient H2O absent pour la bande {nom!r}.")


def _normaliser_coefficients_h2o(payload: Any) -> dict[str, float]:
    if isinstance(payload, dict) and isinstance(payload.get("coefficients"), dict):
        coefficients = {
            str(nom): float(valeur)
            for nom, valeur in payload["coefficients"].items()
        }
    elif isinstance(payload, dict) and isinstance(payload.get("coefficients"), list):
        coefficients = dict(
            _coefficient_h2o_depuis_entree(entree)
            for entree in payload["coefficients"]
        )
    elif isinstance(payload, dict):
        coefficients = {str(nom): float(valeur) for nom, valeur in payload.items()}
    else:
        raise ValueError("Format de coefficients H2O invalide.")

    for nom, valeur in coefficients.items():
        if not valeur >= 0.0:
            raise ValueError(f"Coefficient H2O invalide pour {nom}: {valeur!r}")
    return coefficients


def charger_coefficients_h2o(chemin: Path = CHEMIN_COEFFICIENTS_H2O) -> dict[str, float]:
    """Charge les coefficients H2O utilises par le runtime du modele."""

    payload = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return _normaliser_coefficients_h2o(payload)


COEFFICIENTS_H2O_MODELE3 = charger_coefficients_h2o()
