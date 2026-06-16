"""Tests numeriques minimaux du modele 3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3.donnees import charger_colonne_locale
from modele3.modele3 import (
    TEMPERATURE_SURFACE_DEFAUT_K,
    calculer_colonne_radiative,
    construire_bords_pression_hpa,
    construire_couches,
)
from modele3.physique import calculs as physique


def _donnees_test(q_bas, q_haut=1e-5):
    return {
        "surface": {
            "latitude_deg": 48.8566,
            "longitude_deg": 2.3522,
            "mois": 7,
            "pression_surface_pa": 100_000.0,
            "albedo_surface": 0.2,
            "emissivite_surface": 0.98,
            "cloud_total": 0.0,
            "low_cloud": 0.0,
            "medium_cloud": 0.0,
            "high_cloud": 0.0,
        },
        "profil": {
            "pressions_hpa": [1000.0, 850.0, 700.0, 500.0, 300.0, 100.0, 10.0, 1.0],
            "temperatures_k": [290.0, 280.0, 270.0, 255.0, 235.0, 220.0, 230.0, 245.0],
            "humidites_specifiques_kgkg": [
                q_bas,
                0.7 * q_bas,
                0.4 * q_bas,
                0.1 * q_bas,
                0.01 * q_bas,
                q_haut,
                q_haut,
                q_haut,
            ],
            "fractions_nuageuses": None,
        },
        "validation_flux": {},
        "source": "test",
    }


def tester_bords_pression_surface_locale():
    bords = construire_bords_pression_hpa(750.0)
    assert bords == [750.0, 700.0, 500.0, 300.0, 200.0, 100.0, 50.0, 20.0, 10.0, 1.0]


def tester_couches_descendantes_et_delta_positif():
    couches = construire_couches(_donnees_test(0.006))
    assert len(couches) == 10
    assert all(couche["pression_bas_pa"] - couche["pression_haut_pa"] > 0.0 for couche in couches)
    assert all(
        bas["pression_haut_pa"] == haut["pression_bas_pa"]
        for bas, haut in zip(couches[:-1], couches[1:])
    )


def tester_humidite_augmente_lw_down_et_reduit_olr():
    sec = calculer_colonne_radiative(_donnees_test(0.0002), TEMPERATURE_SURFACE_DEFAUT_K)
    humide = calculer_colonne_radiative(_donnees_test(0.0100), TEMPERATURE_SURFACE_DEFAUT_K)
    assert humide["LW_down_surface"] > sec["LW_down_surface"]
    assert humide["OLR"] < sec["OLR"]


def tester_calculs_physiques_separes():
    albedo_cloud = physique.albedo_nuage_effectif(0.5)
    sw_absorbe = physique.flux_sw_absorbe_surface(1000.0, 0.2, albedo_cloud)
    assert albedo_cloud == 0.25
    assert sw_absorbe == 600.0


def tester_seuils_execution_secours():
    donnees = charger_colonne_locale(
        lat=0.0,
        lon=0.0,
        mois=1,
        ressources_dir=Path("/chemin/inexistant"),
        utiliser_extrait_defaut=False,
    )
    resultat = calculer_colonne_radiative(donnees)
    assert 0.0 <= resultat["albedo_cloud"] <= 0.95
    assert resultat["LW_up_surface"] > 0.0
    assert resultat["OLR"] > 0.0


def main():
    tester_bords_pression_surface_locale()
    tester_couches_descendantes_et_delta_positif()
    tester_humidite_augmente_lw_down_et_reduit_olr()
    tester_calculs_physiques_separes()
    tester_seuils_execution_secours()
    print("tests_modele3_ok")


if __name__ == "__main__":
    main()
