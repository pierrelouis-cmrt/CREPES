"""Tests numeriques minimaux du modele 4."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3.donnees import charger_paquet_grille
from modele4 import surface
from modele4.modele4 import ConfigurationModele4, enregistrer_resultat, simuler, simuler_mensuel


def tester_capacite_surface_finie():
    cellule = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 0.0,
    }
    capacite = surface.capacite_surface(cellule)
    attendu = surface.CP_SEC * 1000.0 * surface.RHO_BULK * surface.EPAISSEUR_ACTIVE_M
    assert abs(capacite - attendu) < 1e-6


def tester_capacite_depuis_rzsm_modifie_le_sol():
    cellule = {
        "land_fraction": 1.0,
        "snow_ice_fraction": 0.0,
    }
    capacite_seche = surface.capacite_surface(cellule)
    capacite_humide = surface.capacite_surface(cellule, rzsm=0.35)
    assert capacite_humide > capacite_seche


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
    assert resultat["capacite_surface_j_m2_k"].shape == (1, 1)
    assert "flux_net_surface" in resultat["diagnostics_moyens"]


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
    assert resultat["metadata"]["mode_sortie"] == "mensuel"


def main():
    tester_capacite_surface_finie()
    tester_capacite_depuis_rzsm_modifie_le_sol()
    tester_flux_convection_signe()
    tester_simulation_courte_point()
    tester_ecriture_npz()
    tester_simulation_mensuelle_point()
    print("tests_modele4_ok")


if __name__ == "__main__":
    main()
