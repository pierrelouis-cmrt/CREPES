"""Chargement du paquet unique de coefficients d'opacite du modele 3."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


DOSSIER_RESSOURCES_MODELE3 = Path(__file__).resolve().parents[1] / "ressources"
CHEMIN_COEFFICIENTS_OPACITE = DOSSIER_RESSOURCES_MODELE3 / "coefficients_opacite_modele3.npz"
TAU_LW_NUAGE_DEFAUT = 1.0


def _valider_coefficients(nom_famille: str, coefficients: dict[str, float]) -> dict[str, float]:
    normalises: dict[str, float] = {}
    for nom, valeur in coefficients.items():
        coefficient = float(valeur)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(f"Coefficient {nom_famille} invalide pour {nom}: {valeur!r}")
        normalises[str(nom)] = coefficient
    return normalises


def _lire_coefficients(
    npz,
    cle_noms: str,
    cle_valeurs: str,
    nom_famille: str,
) -> dict[str, float]:
    if cle_noms not in npz.files or cle_valeurs not in npz.files:
        raise KeyError(f"Coefficients {nom_famille} absents de {CHEMIN_COEFFICIENTS_OPACITE}.")

    noms = np.asarray(npz[cle_noms]).astype(str)
    valeurs = np.asarray(npz[cle_valeurs], dtype=np.float64)
    if noms.ndim != 1 or valeurs.ndim != 1 or noms.shape != valeurs.shape:
        raise ValueError(f"Tableaux {nom_famille} incoherents dans le paquet coefficients.")

    return _valider_coefficients(
        nom_famille,
        {str(nom): float(valeur) for nom, valeur in zip(noms, valeurs)},
    )


def charger_coefficients_opacite(
    chemin: Path = CHEMIN_COEFFICIENTS_OPACITE,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Charge CO2, H2O et nuages depuis le meme fichier `.npz`."""

    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Paquet coefficients introuvable: {chemin}")

    with np.load(chemin, allow_pickle=False) as npz:
        coefficients_co2 = _lire_coefficients(npz, "noms_co2", "a_co2", "CO2")
        coefficients_h2o = _lire_coefficients(npz, "noms_h2o", "a_h2o", "H2O")
        tau = float(np.asarray(npz["tau_lw_par_fraction_nuage"], dtype=np.float64).item())

    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau_lw_par_fraction_nuage doit etre positif ou nul.")

    return (
        coefficients_co2,
        coefficients_h2o,
        {"tau_lw_par_fraction_nuage": tau},
    )


def _tableaux_coefficients(prefixe: str, coefficients: dict[str, float]) -> dict[str, np.ndarray]:
    coefficients = _valider_coefficients(prefixe.upper(), coefficients)
    noms = np.array(sorted(coefficients), dtype="U64")
    valeurs = np.array([coefficients[nom] for nom in noms], dtype=np.float64)
    return {f"noms_{prefixe}": noms, f"a_{prefixe}": valeurs}


def ecrire_coefficients_opacite(
    chemin: Path = CHEMIN_COEFFICIENTS_OPACITE,
    *,
    coefficients_co2: dict[str, float] | None = None,
    coefficients_h2o: dict[str, float] | None = None,
    parametres_nuages: dict[str, float] | None = None,
) -> Path:
    """Ecrit un paquet `.npz` en conservant les familles non remplacees."""

    chemin = Path(chemin)
    if chemin.exists():
        co2_actuel, h2o_actuel, nuages_actuels = charger_coefficients_opacite(chemin)
    else:
        co2_actuel, h2o_actuel = {}, {}
        nuages_actuels = {"tau_lw_par_fraction_nuage": TAU_LW_NUAGE_DEFAUT}

    co2 = co2_actuel if coefficients_co2 is None else coefficients_co2
    h2o = h2o_actuel if coefficients_h2o is None else coefficients_h2o
    nuages = nuages_actuels if parametres_nuages is None else parametres_nuages
    tau = float(nuages.get("tau_lw_par_fraction_nuage", TAU_LW_NUAGE_DEFAUT))
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError("tau_lw_par_fraction_nuage doit etre positif ou nul.")

    chemin.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chemin,
        **_tableaux_coefficients("co2", co2),
        **_tableaux_coefficients("h2o", h2o),
        tau_lw_par_fraction_nuage=np.array(tau, dtype=np.float64),
    )
    return chemin


def charger_coefficients_co2(chemin: Path = CHEMIN_COEFFICIENTS_OPACITE) -> dict[str, float]:
    return charger_coefficients_opacite(chemin)[0]


def charger_coefficients_h2o(chemin: Path = CHEMIN_COEFFICIENTS_OPACITE) -> dict[str, float]:
    return charger_coefficients_opacite(chemin)[1]


def charger_parametres_nuages(chemin: Path = CHEMIN_COEFFICIENTS_OPACITE) -> dict[str, float]:
    return charger_coefficients_opacite(chemin)[2]


(
    COEFFICIENTS_CO2_MODELE3,
    COEFFICIENTS_H2O_MODELE3,
    PARAMETRES_NUAGES_MODELE3,
) = charger_coefficients_opacite()
