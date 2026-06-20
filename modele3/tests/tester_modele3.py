"""Tests numeriques minimaux du modele 3 final."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3 import physique
from modele3.donnees import charger_paquet_grille, extraire_colonne, iterer_colonnes
from modele3.modele3 import calculer_colonne_radiative, construire_couches


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


def tester_court_onde_unique_transmissivite():
    donnees = _donnees_test()
    resultat = calculer_colonne_radiative(donnees, moyenne_journaliere_sw=True)
    attendu = (
        resultat["SW_TOA_local"]
        * donnees["surface"]["transmissivite_sw_mensuelle"]
        * (1.0 - donnees["surface"]["albedo_surface"])
    )
    assert abs(resultat["SW_absorbe_surface"] - attendu) < 1e-9


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


def tester_paquet_grille_chargeable_et_final():
    paquet = charger_paquet_grille()
    metadata = paquet["metadata"]
    assert metadata["shape"]["lat"] == 36
    assert metadata["shape"]["lon"] == 72
    assert "transmissivite_sw_mensuelle" in metadata["variables"]
    assert "sw_toa_moyen_mensuel_w_m2" in metadata["variables"]
    assert abs(float(paquet["donnees"]["poids_surface"].sum()) - 1.0) < 1e-6
    transmissivite = paquet["donnees"]["transmissivite_sw_mensuelle"]
    assert 0.0 <= float(transmissivite.min()) <= float(transmissivite.max()) <= 1.0


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
    assert resultat["SW_absorbe_surface"] >= 0.0
    assert resultat["LW_up_surface"] > 0.0
    assert resultat["OLR"] > 0.0


def tester_appel_en_boucle_sur_plusieurs_colonnes():
    paquet = charger_paquet_grille()
    colonnes = iterer_colonnes(paquet, mois=1)
    resultats = []
    for _ in range(3):
        resultats.append(calculer_colonne_radiative(next(colonnes)))
    assert len(resultats) == 3
    assert all(resultat["LW_up_surface"] > 0.0 for resultat in resultats)


def main():
    tester_court_onde_unique_transmissivite()
    tester_emissivite_constante()
    tester_nuages_lw_absents_des_opacites()
    tester_paquet_grille_chargeable_et_final()
    tester_colonne_depuis_paquet()
    tester_appel_en_boucle_sur_plusieurs_colonnes()
    print("tests_modele3_ok")


if __name__ == "__main__":
    main()
