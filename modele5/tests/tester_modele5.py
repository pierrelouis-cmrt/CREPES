"""Tests numeriques du modele 5."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modele3.codes_python.donnees import charger_paquet_grille
from modele4.codes_python.rapide import ConfigurationRapide, simuler_rapide
from modele5.codes_python.modele5 import (
    ConfigurationModele5,
    _geometrie_grille,
    calculer_convergence_horizontale,
    enregistrer_resultat,
    simuler,
)


def tester_echange_nul_si_emission_uniforme():
    latitudes = np.array([-2.5, 2.5])
    longitudes = np.array([-5.0, 0.0, 5.0])
    emission = np.full((2, 3, 2, 3), 42.0)
    epaisseur = np.full((2, 2, 3), 3_000.0)
    convergence = calculer_convergence_horizontale(
        emission, epaisseur, _geometrie_grille(latitudes, longitudes), periodique_longitude=True
    )
    assert np.allclose(convergence, 0.0)


def tester_echange_conserve_la_puissance_globale():
    latitudes = np.array([-7.5, -2.5, 2.5])
    longitudes = np.array([-5.0, 0.0, 5.0])
    rng = np.random.default_rng(7)
    emission = rng.uniform(1.0, 20.0, size=(2, 3, 3, 3))
    epaisseur = rng.uniform(1_000.0, 8_000.0, size=(2, 3, 3))
    geometrie = _geometrie_grille(latitudes, longitudes)
    convergence = calculer_convergence_horizontale(
        emission, epaisseur, geometrie, periodique_longitude=True
    )
    puissance_nette = np.sum(convergence * geometrie["aire_m2"][None, None, :, None])
    assert abs(puissance_nette) < 1e-4


def tester_simulation_courte_et_sortie_npz():
    paquet = charger_paquet_grille()
    config = ConfigurationModele5(
        jours=1.0 / 24.0,
        dt_s=1800.0,
        sortie_heures=0.5,
        max_latitudes=2,
        max_longitudes=3,
        facteur_latent=0.0,
        mode_convection="aucune",
        afficher_progression=False,
    )
    resultat = simuler(paquet, config)
    assert resultat["temperature_surface_k"].shape == (3, 2, 3)
    assert np.isfinite(resultat["temperature_surface_k"]).all()
    assert "flux_horizontal_net_surface" in resultat["diagnostics_moyens"]
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "modele5_test.npz"
        enregistrer_resultat(resultat, chemin)
        with np.load(chemin) as npz:
            assert "flux_horizontal_net_surface_moyen_w_m2" in npz.files
            assert "temperature_surface_k" in npz.files


def tester_facteur_horizontal_nul_reproduit_modele4_et_echange_agit():
    paquet = charger_paquet_grille()
    commun = dict(
        jours=1.0 / 24.0,
        dt_s=1800.0,
        sortie_heures=0.5,
        max_latitudes=2,
        max_longitudes=3,
        facteur_latent=0.0,
        mode_convection="aucune",
        afficher_progression=False,
    )
    resultat_4 = simuler_rapide(paquet, ConfigurationRapide(**commun))
    resultat_sans_horizontal = simuler(
        paquet, ConfigurationModele5(**commun, facteur_horizontal=0.0)
    )
    resultat_avec_horizontal = simuler(paquet, ConfigurationModele5(**commun))
    assert np.array_equal(
        resultat_4["temperature_surface_k"],
        resultat_sans_horizontal["temperature_surface_k"],
    )
    ecart = np.max(
        np.abs(
            resultat_avec_horizontal["temperature_surface_k"]
            - resultat_sans_horizontal["temperature_surface_k"]
        )
    )
    assert ecart > 0.0


def main():
    tester_echange_nul_si_emission_uniforme()
    tester_echange_conserve_la_puissance_globale()
    tester_simulation_courte_et_sortie_npz()
    tester_facteur_horizontal_nul_reproduit_modele4_et_echange_agit()
    print("tests_modele5_ok")


if __name__ == "__main__":
    main()
