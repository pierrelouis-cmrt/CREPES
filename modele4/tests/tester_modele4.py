"""Tests numeriques minimaux du modele 4."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3.codes_python.donnees import charger_paquet_grille
from modele4.codes_python import surface
from modele4.codes_python.modele4 import ConfigurationModele4, enregistrer_resultat, simuler, simuler_mensuel


def tester_capacite_surface_finie():
    cellule = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 0.0,
    }
    capacite = surface.capacite_surface(cellule)
    attendu = surface.capacite_sol_sec()
    assert abs(capacite - attendu) < 1e-6
    assert 5e5 < capacite < 2e6


def tester_capacite_depuis_rzsm_modifie_le_sol():
    cellule = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 0.0,
    }
    capacite_seche = surface.capacite_surface(cellule)
    capacite_humide = surface.capacite_surface(cellule, rzsm=0.35)
    assert capacite_humide > capacite_seche


def tester_capacite_rzsm_nan_retombe_sur_sol_sec():
    cellule = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 0.0,
    }
    capacite = surface.capacite_surface(cellule, rzsm=np.nan)
    assert abs(capacite - surface.capacite_sol_sec()) < 1e-6


def tester_capacite_ocean_ignore_rzsm_et_reste_plausible():
    cellule_ocean = {
        "land_fraction": 0.0,
        "snow_ice_fraction": 0.0,
    }
    capacite = surface.capacite_surface(cellule_ocean, rzsm=0.35)
    assert abs(capacite - surface.capacite_ocean_surface()) < 1e-6
    assert capacite > 3.0 * surface.capacite_sol_sec()
    assert 3e6 < capacite < 1e7


def tester_capacite_glace_neige_prioritaire_sur_rzsm():
    cellule_glace = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 1.0,
    }
    capacite = surface.capacite_surface(cellule_glace, rzsm=0.35)
    assert abs(capacite - surface.capacite_glace_neige_surface()) < 1e-6
    assert surface.capacite_sol_sec() < capacite < surface.capacite_ocean_surface()


def tester_capacite_surface_positive_plausible():
    capacites = surface.capacite_depuis_rzsm(np.array([0.05, 0.35, 0.9]))
    assert np.isfinite(capacites).all()
    assert (capacites > 5.0e5).all()
    assert (capacites < 5.0e6).all()


def tester_grille_rzsm_modele0_bins_1_degre():
    contenu = "\n".join(
        (
            "lat,lon,RZSM",
            "0.0,0.0,0.2",
            "0.25,0.25,0.4",
            "1.0,1.0,0.8",
        )
    )
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "rzsm.csv"
        chemin.write_text(contenu, encoding="utf-8")
        grille = surface.charger_grille_rzsm(chemin)

    valeur = surface.rzsm_plus_proche(grille, 0.2, 0.2)
    assert abs(valeur - 0.3) < 1e-12


def tester_flux_latent_par_continent_sans_moyenne():
    cellule = {
        "latitude_deg": 10.0,
        "longitude_deg": 20.0,
        "land_fraction": 1.0,
    }
    flux = surface.flux_latent_moyen(
        cellule,
        detecteur_continent=lambda _lat, _lon: "Africa",
    )
    assert abs(flux - surface.Q_LATENT_CONTINENT_W_M2["Africa"]) < 1e-12

    flux_polaire = surface.flux_latent_moyen(
        {"latitude_deg": 80.0, "longitude_deg": 0.0},
        detecteur_continent=lambda _lat, _lon: "Europe",
    )
    assert flux_polaire == 0.0


def tester_flux_convection_signe():
    config = surface.ConfigurationSurface(mode_convection="toutes", vent_m_s=2.5)
    flux_chaud = surface.flux_convection(300.0, 290.0, config)
    flux_froid = surface.flux_convection(280.0, 290.0, config)
    assert flux_chaud > 0.0
    assert flux_froid < 0.0


def tester_simulation_courte_point():
    paquet = charger_paquet_grille()
    config = ConfigurationModele4(
        jours=1.0 / 48.0,
        dt_s=1800.0,
        frequence_sortie_pas=1,
        iterations_implicites=2,
        indices_lat=(18,),
        indices_lon=(36,),
        surface=surface.ConfigurationSurface(facteur_latent=0.0, mode_convection="aucune"),
    )
    resultat = simuler(paquet, config)
    temperatures = resultat["temperature_surface_k"]
    assert temperatures.shape == (2, 1, 1)
    assert np.isfinite(temperatures).all()
    assert temperatures.min() > 150.0
    assert temperatures.max() < 350.0
    assert resultat["capacite_surface_j_m2_k"].shape == (1, 1)
    assert np.isfinite(resultat["capacite_surface_j_m2_k"]).all()
    assert (resultat["capacite_surface_j_m2_k"] > 0.0).all()
    assert "flux_net_surface" in resultat["diagnostics_moyens"]

    diagnostics = resultat["diagnostics_moyens"]
    assert diagnostics["SW_absorbe_surface"][0, 0] >= 0.0
    assert diagnostics["LW_down_absorbe_surface"][0, 0] > 0.0
    assert diagnostics["LW_up_surface"][0, 0] > 0.0
    assert diagnostics["flux_latent"][0, 0] == 0.0
    assert diagnostics["flux_convection"][0, 0] == 0.0

    delta_temperature = float(temperatures[-1, 0, 0] - temperatures[0, 0, 0])
    flux_net = float(diagnostics["flux_net_surface"][0, 0])
    capacite = float(resultat["capacite_surface_j_m2_k"][0, 0])
    attendu = config.dt_s * flux_net / capacite
    assert abs(delta_temperature - attendu) < 5e-4


def tester_ecriture_npz():
    paquet = charger_paquet_grille()
    config = ConfigurationModele4(
        jours=1.0 / 48.0,
        dt_s=1800.0,
        frequence_sortie_pas=1,
        iterations_implicites=1,
        indices_lat=(18,),
        indices_lon=(36,),
        surface=surface.ConfigurationSurface(facteur_latent=0.0, mode_convection="aucune"),
    )
    resultat = simuler(paquet, config)
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "modele4_test.npz"
        enregistrer_resultat(resultat, chemin)
        with np.load(chemin) as npz:
            assert "temperature_surface_k" in npz.files
            assert "metadata_json" in npz.files


def tester_simulation_mensuelle_point():
    paquet = charger_paquet_grille()
    config = ConfigurationModele4(
        iterations_implicites=1,
        indices_lat=(18,),
        indices_lon=(36,),
        surface=surface.ConfigurationSurface(facteur_latent=0.0, mode_convection="aucune"),
    )
    resultat = simuler_mensuel(paquet, config)
    temperatures = resultat["temperature_surface_k"]
    assert temperatures.shape == (12, 1, 1)
    assert resultat["mois"].tolist() == list(range(1, 13))
    assert np.isfinite(temperatures).all()
    assert resultat["metadata"]["mode_sortie"] == "diagnostic_mensuel_un_pas"


def main():
    tester_capacite_surface_finie()
    tester_capacite_depuis_rzsm_modifie_le_sol()
    tester_capacite_rzsm_nan_retombe_sur_sol_sec()
    tester_capacite_ocean_ignore_rzsm_et_reste_plausible()
    tester_capacite_glace_neige_prioritaire_sur_rzsm()
    tester_capacite_surface_positive_plausible()
    tester_grille_rzsm_modele0_bins_1_degre()
    tester_flux_latent_par_continent_sans_moyenne()
    tester_flux_convection_signe()
    tester_simulation_courte_point()
    tester_ecriture_npz()
    tester_simulation_mensuelle_point()
    print("tests_modele4_ok")


if __name__ == "__main__":
    main()
