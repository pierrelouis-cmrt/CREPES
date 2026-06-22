"""Chargement des coefficients d'opacite effectifs du modele 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DOSSIER_RESSOURCES_MODELE3 = Path(__file__).resolve().parents[1] / "ressources"
CHEMIN_COEFFICIENTS_CO2 = (
    DOSSIER_RESSOURCES_MODELE3
    / "calibrage_opacite_co2"
    / "coefficients_co2_modele3.json"
)
CHEMIN_COEFFICIENTS_H2O = (
    DOSSIER_RESSOURCES_MODELE3
    / "calibrage_opacite_h2o"
    / "coefficients_h2o_modele3.json"
)
CHEMIN_COEFFICIENTS_NUAGES = (
    DOSSIER_RESSOURCES_MODELE3
    / "calibrage_opacite_nuages"
    / "coefficients_nuages_modele3.json"
)


def _valider_coefficients(nom_famille: str, coefficients: dict[str, float]) -> dict[str, float]:
    for nom, valeur in coefficients.items():
        if not valeur >= 0.0:
            raise ValueError(f"Coefficient {nom_famille} invalide pour {nom}: {valeur!r}")
    return coefficients


def _coefficient_depuis_entree(
    entree: dict[str, Any],
    cles: tuple[str, ...],
    nom_famille: str,
) -> tuple[str, float]:
    nom = str(entree["nom"])
    for cle in cles:
        if cle in entree:
            return nom, float(entree[cle])
    raise ValueError(f"Coefficient {nom_famille} absent pour la bande {nom!r}.")


def _normaliser_coefficients_bandes(
    payload: Any,
    cles: tuple[str, ...],
    nom_famille: str,
) -> dict[str, float]:
    if isinstance(payload, dict) and isinstance(payload.get("coefficients"), dict):
        coefficients = {
            str(nom): float(valeur)
            for nom, valeur in payload["coefficients"].items()
        }
    elif isinstance(payload, dict) and isinstance(payload.get("coefficients"), list):
        coefficients = dict(
            _coefficient_depuis_entree(entree, cles, nom_famille)
            for entree in payload["coefficients"]
        )
    elif isinstance(payload, dict):
        coefficients = {str(nom): float(valeur) for nom, valeur in payload.items()}
    else:
        raise ValueError(f"Format de coefficients {nom_famille} invalide.")

    return _valider_coefficients(nom_famille, coefficients)


def _coefficient_h2o_depuis_entree(entree: dict[str, Any]) -> tuple[str, float]:
    nom = str(entree["nom"])
    for cle in ("a_h2o_modele3", "a_h2o_hitran", "a_h2o"):
        if cle in entree:
            return nom, float(entree[cle])
    raise ValueError(f"Coefficient H2O absent pour la bande {nom!r}.")


def _normaliser_coefficients_h2o(payload: Any) -> dict[str, float]:
    # Garde le contrat historique: les rapports de calibrage peuvent exposer
    # plusieurs valeurs, le runtime retient explicitement a_h2o_modele3.
    if isinstance(payload, dict) and isinstance(payload.get("coefficients"), list):
        return _valider_coefficients(
            "H2O",
            dict(_coefficient_h2o_depuis_entree(entree) for entree in payload["coefficients"]),
        )
    return _normaliser_coefficients_bandes(
        payload,
        ("a_h2o_modele3", "a_h2o_hitran", "a_h2o"),
        "H2O",
    )


def charger_coefficients_co2(chemin: Path = CHEMIN_COEFFICIENTS_CO2) -> dict[str, float]:
    """Charge les coefficients CO2 calibres utilises par le runtime."""

    payload = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return _normaliser_coefficients_bandes(
        payload,
        ("a_co2_modele3", "a_co2_final", "a_co2_hitran", "a_co2"),
        "CO2",
    )


def charger_coefficients_h2o(chemin: Path = CHEMIN_COEFFICIENTS_H2O) -> dict[str, float]:
    """Charge les coefficients H2O utilises par le runtime du modele."""

    payload = json.loads(Path(chemin).read_text(encoding="utf-8"))
    return _normaliser_coefficients_h2o(payload)


def charger_parametres_nuages(chemin: Path = CHEMIN_COEFFICIENTS_NUAGES) -> dict[str, float]:
    """Charge les parametres simples de l'opacite nuageuse long-onde."""

    payload = json.loads(Path(chemin).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("parametres"), dict):
        parametres = payload["parametres"]
    elif isinstance(payload, dict):
        parametres = payload
    else:
        raise ValueError("Format de coefficients nuages invalide.")

    tau = float(parametres.get("tau_lw_par_fraction_nuage", 0.0))
    if tau < 0.0:
        raise ValueError("tau_lw_par_fraction_nuage doit etre positif ou nul.")
    return {"tau_lw_par_fraction_nuage": tau}


COEFFICIENTS_CO2_MODELE3 = charger_coefficients_co2()
COEFFICIENTS_H2O_MODELE3 = charger_coefficients_h2o()
PARAMETRES_NUAGES_MODELE3 = charger_parametres_nuages()
