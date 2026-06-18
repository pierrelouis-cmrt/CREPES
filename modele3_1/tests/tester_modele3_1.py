"""Tests numeriques minimaux du modele 3.1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3_1 import physique
from modele3_1.donnees import charger_paquet_grille, extraire_colonne, iterer_colonnes
from modele3_1.modele3_1 import (
    TEMPERATURE_SURFACE_DEFAUT_K,
    calculer_colonne_radiative,
    construire_couches,
)


def _donnees_test(albedo_nuages=0.20, cloud_total=1.0, emissivite_surface=0.10):
    return {
        "surface": {
            "latitude_deg": 48.8566,
            "longitude_deg": 2.3522,
            "mois": 7,
            "pression_surface_pa": 100_000.0,
            "albedo_surface": 0.30,
            "albedo_nuages_effectif": albedo_nuages,
            "transmissivite_sw_mensuelle": 0.60,
            "emissivite_surface": emissivite_surface,
            "cloud_total": cloud_total,
            "low_cloud": cloud_total,
            "medium_cloud": cloud_total,
            "high_cloud": cloud_total,
        },
        "profil": {
            "pressions_hpa": [1000.0, 850.0, 700.0, 500.0, 300.0, 100.0, 10.0, 1.0],
            "temperatures_k": [290.0, 280.0, 270.0, 255.0, 235.0, 220.0, 230.0, 245.0],
            "humidites_specifiques_kgkg": [0.006, 0.004, 0.002, 0.0008, 0.0001, 1e-5, 1e-5, 1e-5],
            "fractions_nuageuses": None,
        },
        "validation_flux": {
            "era5_sw_down_surface_w_m2": 240.0,
            "era5_sw_net_surface_w_m2": 168.0,
        },
        "source": "test",
    }


def tester_emissivite_constante_ignore_branches_surface():
    resultat = calculer_colonne_radiative(_donnees_test(emissivite_surface=0.10))
    assert resultat["emissivite_surface"] == 0.98
    assert resultat["sources"]["emissivite_surface"] == "constante_0.98"


def tester_albedo_nuages_fourni_pas_cloud_total():
    donnees = _donnees_test(albedo_nuages=0.20, cloud_total=1.0)
    resultat = calculer_colonne_radiative(
        donnees,
        temperature_surface_k=TEMPERATURE_SURFACE_DEFAUT_K,
        moyenne_journaliere_sw=True,
        mode_court_onde="toa_nuages_ceres",
    )
    attendu = physique.flux_sw_absorbe_surface(
        resultat["SW_incident_surface"],
        0.30,
        0.20,
    )
    assert abs(resultat["SW_absorbe_surface"] - attendu) < 1e-9
    assert resultat["albedo_nuages_effectif"] == 0.20


def tester_mode_transmissivite_sw_utilise_transmissivite():
    donnees = _donnees_test()
    resultat = calculer_colonne_radiative(
        donnees,
        temperature_surface_k=TEMPERATURE_SURFACE_DEFAUT_K,
        moyenne_journaliere_sw=True,
        mode_court_onde="transmissivite_sw",
    )
    attendu = (
        resultat["SW_TOA_local"]
        * donnees["surface"]["transmissivite_sw_mensuelle"]
        * (1.0 - donnees["surface"]["albedo_surface"])
    )
    assert abs(resultat["SW_absorbe_surface"] - attendu) < 1e-9
    assert resultat["mode_court_onde"] == "transmissivite_sw"


def tester_mode_era5_net_renvoie_flux_net():
    resultat = calculer_colonne_radiative(_donnees_test(), mode_court_onde="era5_net")
    assert resultat["SW_absorbe_surface"] == 168.0


def tester_nuages_lw_absents_des_opacites():
    couches = construire_couches(_donnees_test(cloud_total=1.0))
    diagnostic = physique.opacites_couche_bande(couches[0], physique.BANDES_INFRAROUGES[0])
    assert "tau_nuage" not in diagnostic
    assert diagnostic["tau_total"] == diagnostic["tau_co2"] + diagnostic["tau_h2o"]


def tester_paquet_grille_chargeable_et_sources_racine():
    paquet = charger_paquet_grille()
    metadata = paquet["metadata"]
    assert metadata["shape"]["lat"] == 36
    assert metadata["shape"]["lon"] == 72
    assert "transmissivite_sw_mensuelle" in metadata["variables"]
    assert "sw_toa_moyen_mensuel_w_m2" in metadata["variables"]
    assert abs(float(paquet["donnees"]["poids_surface"].sum()) - 1.0) < 1e-6
    transmissivite = paquet["donnees"]["transmissivite_sw_mensuelle"]
    assert 0.0 <= float(transmissivite.min()) <= float(transmissivite.max()) <= 1.0
    sources = "\n".join(str(v) for v in metadata["sources_fichiers"].values())
    assert "ressources/albedo" in sources
    assert "modele0_maintenance" not in sources


def tester_paris_depuis_paquet_et_diagnostics_legers():
    paquet = charger_paquet_grille()
    colonne = extraire_colonne(paquet, 48.8566, 2.3522, mois=7)
    resultat = calculer_colonne_radiative(
        colonne,
        temperature_surface_k=293.0,
        moyenne_journaliere_sw=True,
        diagnostics_lourds=False,
    )
    assert colonne["source"].startswith("paquet ")
    assert colonne["surface"]["emissivite_surface"] == 0.98
    assert 0.0 <= colonne["surface"]["albedo_surface"] <= 1.0
    assert 0.0 <= colonne["surface"]["albedo_nuages_effectif"] <= 0.95
    assert 0.0 <= colonne["surface"]["transmissivite_sw_mensuelle"] <= 1.0
    assert len(colonne["couches"]) > 0
    assert resultat["couches"] == []
    assert resultat["LW_up_surface"] > 0.0
    assert resultat["OLR"] > 0.0
    attendu_sw = colonne["validation_flux"]["era5_sw_down_surface_w_m2"] * (
        1.0 - colonne["surface"]["albedo_surface"]
    )
    assert abs(resultat["SW_absorbe_surface"] - attendu_sw) < 5.0
    resultat_era5 = calculer_colonne_radiative(
        colonne,
        temperature_surface_k=293.0,
        mode_court_onde="era5_down_albedo",
    )
    assert abs(resultat_era5["SW_absorbe_surface"] - attendu_sw) < 1e-9


def tester_appel_en_boucle_sur_plusieurs_colonnes():
    paquet = charger_paquet_grille()
    colonnes = iterer_colonnes(paquet, mois=1)
    resultats = []
    for _ in range(3):
        colonne = next(colonnes)
        resultats.append(calculer_colonne_radiative(colonne))
    assert len(resultats) == 3
    assert all(resultat["LW_up_surface"] > 0.0 for resultat in resultats)


def main():
    tester_emissivite_constante_ignore_branches_surface()
    tester_albedo_nuages_fourni_pas_cloud_total()
    tester_mode_transmissivite_sw_utilise_transmissivite()
    tester_mode_era5_net_renvoie_flux_net()
    tester_nuages_lw_absents_des_opacites()
    tester_paquet_grille_chargeable_et_sources_racine()
    tester_paris_depuis_paquet_et_diagnostics_legers()
    tester_appel_en_boucle_sur_plusieurs_colonnes()
    print("tests_modele3_1_ok")


if __name__ == "__main__":
    main()
