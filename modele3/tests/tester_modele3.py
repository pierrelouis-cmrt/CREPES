"""Tests numeriques minimaux du modele 3 final."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3 import physique
from modele3.donnees import charger_paquet_grille, extraire_colonne, iterer_colonnes
from modele3.modele3 import calculer_colonne_radiative, construire_couches
from modele3.ressources.generer_donnees import _nearest_matrix, normaliser_longitudes_180


def _donnees_test(transmissivite=0.60):
    return {
        "surface": {
            "latitude_deg": 0.0,
            "longitude_deg": 0.0,
            "mois": 7,
            "pression_surface_pa": 100_000.0,
            "albedo_surface": 0.30,
            "transmissivite_sw_mensuelle": transmissivite,
        },
        "couches": [
            {
                "pression_bas_hpa": 1000.0,
                "pression_haut_hpa": 850.0,
                "temperature_k": 285.0,
                "humidite_specifique_kgkg": 0.006,
                "masse_air_kg_m2": 1529.57,
                "masse_h2o_kg_m2": 9.18,
            },
            {
                "pression_bas_hpa": 850.0,
                "pression_haut_hpa": 700.0,
                "temperature_k": 275.0,
                "humidite_specifique_kgkg": 0.003,
                "masse_air_kg_m2": 1529.57,
                "masse_h2o_kg_m2": 4.59,
            },
        ],
        "validation_flux": {
            "era5_sw_down_surface_w_m2": 240.0,
            "era5_sw_net_surface_w_m2": 168.0,
        },
        "source": "test",
    }


def tester_shortwave_unique_transmissivite():
    donnees = _donnees_test()
    resultat = calculer_colonne_radiative(donnees, moyenne_journaliere_sw=True)
    attendu = (
        resultat["SW_TOA_local"]
        * donnees["surface"]["transmissivite_sw_mensuelle"]
        * (1.0 - donnees["surface"]["albedo_surface"])
    )
    assert abs(resultat["SW_absorbe_surface"] - attendu) < 1e-9


def tester_shortwave_mensuel_utilise_moyenne_paquet():
    donnees = _donnees_test()
    donnees["surface"]["sw_toa_moyen_mensuel_w_m2"] = 321.5
    resultat = calculer_colonne_radiative(donnees, moyenne_journaliere_sw=True)
    assert resultat["mode_shortwave"] == "moyenne_mensuelle_paquet"
    assert resultat["SW_TOA_local"] == 321.5
    assert resultat["SW_down_surface"] == 321.5 * donnees["surface"]["transmissivite_sw_mensuelle"]


def tester_emissivite_constante():
    resultat = calculer_colonne_radiative(_donnees_test())
    assert resultat["emissivite_surface"] == 0.98
    assert resultat["sources"]["emissivite_surface"] == "constante_0.98"


def tester_nuages_lw_absents_des_opacites():
    couches = construire_couches(_donnees_test())
    diagnostic = physique.opacites_couche_bande(couches[0], physique.BANDES_INFRAROUGES[0])
    assert set(diagnostic) == {
        "couche",
        "bande",
        "tau_co2",
        "tau_h2o",
        "tau_total",
        "transmission",
        "emissivite",
    }
    assert diagnostic["tau_total"] == diagnostic["tau_co2"] + diagnostic["tau_h2o"]


def tester_couche_non_positive_refusee_au_calcul():
    donnees = _donnees_test()
    donnees["couches"][0]["pression_haut_hpa"] = donnees["couches"][0]["pression_bas_hpa"]
    try:
        construire_couches(donnees)
    except ValueError as exc:
        assert "pression_bas_pa" in str(exc)
    else:
        raise AssertionError("Une couche a epaisseur nulle doit etre refusee.")


def tester_extraction_trace_les_couches_ignorees():
    donnees = {
        "lat_deg": np.array([0.0], dtype=np.float32),
        "lon_deg": np.array([0.0], dtype=np.float32),
        "pression_surface_hpa": np.full((12, 1, 1), 1000.0, dtype=np.float32),
        "albedo_surface": np.full((12, 1, 1), 0.3, dtype=np.float32),
        "transmissivite_sw_mensuelle": np.full((12, 1, 1), 0.6, dtype=np.float32),
        "sw_toa_moyen_mensuel_w_m2": np.full((12, 1), 300.0, dtype=np.float32),
        "pression_bas_couche_hpa": np.full((12, 2, 1, 1), np.nan, dtype=np.float32),
        "pression_haut_couche_hpa": np.full((12, 2, 1, 1), np.nan, dtype=np.float32),
        "temperature_couche_k": np.full((12, 2, 1, 1), 280.0, dtype=np.float32),
        "humidite_specifique_couche_kgkg": np.full((12, 2, 1, 1), 0.004, dtype=np.float32),
        "masse_air_couche_kg_m2": np.full((12, 2, 1, 1), 1000.0, dtype=np.float32),
        "masse_h2o_couche_kg_m2": np.full((12, 2, 1, 1), 4.0, dtype=np.float32),
    }
    donnees["pression_bas_couche_hpa"][:, 0, 0, 0] = 1000.0
    donnees["pression_haut_couche_hpa"][:, 0, 0, 0] = 850.0
    donnees["pression_bas_couche_hpa"][:, 1, 0, 0] = 850.0
    donnees["pression_haut_couche_hpa"][:, 1, 0, 0] = 850.0
    paquet = {
        "donnees": donnees,
        "metadata": {"variables": {}, "conventions": {"longitude_deg": "-180..180"}},
        "npz_path": Path("synthetique.npz"),
    }
    colonne = extraire_colonne(paquet, 0.0, 0.0, mois=1)
    assert len(colonne["couches"]) == 1
    assert colonne["diagnostics_donnees"]["couches_ignorees_non_positives"] == 1
    assert colonne["diagnostics_donnees"]["couches_non_positives_exemples"]


def tester_longitudes_albedo_normalisees_antimeridien():
    source_lat = np.array([0.0])
    source_lon = np.array([0.0, 90.0, 180.0, 182.5, 270.0])
    valeurs = np.array([[0.1, 0.2, 0.3, 0.7, 0.4]])
    cible = _nearest_matrix(
        source_lat,
        source_lon,
        valeurs,
        np.array([0.0]),
        np.array([-177.5]),
        allow_fallbacks=False,
        fallback=0.3,
    )
    assert normaliser_longitudes_180(np.array([182.5]))[0] == -177.5
    assert abs(float(cible[0, 0]) - 0.7) < 1e-6


def tester_paquet_grille_chargeable_et_final():
    paquet = charger_paquet_grille()
    metadata = paquet["metadata"]
    assert metadata["shape"]["lat"] == 36
    assert metadata["shape"]["lon"] == 72
    assert metadata["conventions"]["longitude_deg"] == "-180..180"
    assert "couches_verticales" in metadata["diagnostics_generation"]
    assert "transmissivite_sw_mensuelle" in metadata["variables"]
    assert "sw_toa_moyen_mensuel_w_m2" in metadata["variables"]
    assert abs(float(paquet["donnees"]["poids_surface"].sum()) - 1.0) < 1e-6
    albedo = paquet["donnees"]["albedo_surface"]
    assert 0.0 <= float(np.nanmin(albedo)) <= float(np.nanmax(albedo)) <= 1.0
    transmissivite = paquet["donnees"]["transmissivite_sw_mensuelle"]
    assert 0.0 <= float(np.nanmin(transmissivite)) <= float(np.nanmax(transmissivite)) <= 1.0
    p_bas = paquet["donnees"]["pression_bas_couche_hpa"]
    p_haut = paquet["donnees"]["pression_haut_couche_hpa"]
    valides = np.isfinite(p_bas) & np.isfinite(p_haut)
    assert np.count_nonzero(valides & (p_bas <= p_haut)) == 0


def tester_colonne_depuis_paquet():
    paquet = charger_paquet_grille()
    colonne = extraire_colonne(paquet, 0.0, 0.0, mois=7)
    resultat = calculer_colonne_radiative(
        colonne,
        temperature_surface_k=293.0,
        moyenne_journaliere_sw=True,
    )
    assert colonne["source"].startswith("paquet ")
    assert colonne["surface"]["emissivite_surface"] == 0.98
    assert 0.0 <= colonne["surface"]["albedo_surface"] <= 1.0
    assert 0.0 <= colonne["surface"]["transmissivite_sw_mensuelle"] <= 1.0
    assert len(colonne["couches"]) > 0
    assert colonne["diagnostics_donnees"]["couches_ignorees_non_positives"] == 0
    for nom in (
        "SW_TOA_local",
        "SW_down_surface",
        "SW_absorbe_surface",
        "LW_up_surface",
        "LW_down_surface",
        "LW_down_absorbe_surface",
        "OLR",
        "flux_net_radiatif_surface",
    ):
        assert math.isfinite(resultat[nom])
    assert 0.0 <= resultat["SW_absorbe_surface"] <= 500.0
    assert 250.0 <= resultat["LW_up_surface"] <= 600.0
    assert 0.0 < resultat["OLR"] <= 600.0


def tester_appel_en_boucle_sur_plusieurs_colonnes():
    paquet = charger_paquet_grille()
    colonnes = iterer_colonnes(paquet, mois=1)
    resultats = []
    for _ in range(3):
        resultats.append(calculer_colonne_radiative(next(colonnes)))
    assert len(resultats) == 3
    assert all(resultat["LW_up_surface"] > 0.0 for resultat in resultats)


def main():
    tester_shortwave_unique_transmissivite()
    tester_shortwave_mensuel_utilise_moyenne_paquet()
    tester_emissivite_constante()
    tester_nuages_lw_absents_des_opacites()
    tester_couche_non_positive_refusee_au_calcul()
    tester_extraction_trace_les_couches_ignorees()
    tester_longitudes_albedo_normalisees_antimeridien()
    tester_paquet_grille_chargeable_et_final()
    tester_colonne_depuis_paquet()
    tester_appel_en_boucle_sur_plusieurs_colonnes()
    print("tests_modele3_ok")


if __name__ == "__main__":
    main()
