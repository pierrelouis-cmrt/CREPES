"""Tests numeriques minimaux du modele 4 rapide."""

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
from modele4.modele4 import ConfigurationModele4, simuler
from modele4.rapide import ConfigurationRapide, enregistrer_resultat, simuler_rapide


BORNE_T_MIN_K = 150.0
BORNE_T_MAX_K = 350.0


def tester_sortie_4h_par_defaut_sur_point():
    paquet = charger_paquet_grille()
    config = ConfigurationRapide(
        max_latitudes=1,
        max_longitudes=1,
        facteur_latent=0.0,
        mode_convection="aucune",
        afficher_progression=False,
    )
    resultat = simuler_rapide(paquet, config)
    temperatures = resultat["temperature_surface_k"]
    assert temperatures.shape == (7, 1, 1)
    assert np.isfinite(temperatures).all()
    assert temperatures.min() > BORNE_T_MIN_K
    assert temperatures.max() < BORNE_T_MAX_K
    assert resultat["heures"].tolist() == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 0.0]
    assert resultat["mois_precalcules"].tolist() == [1]
    assert resultat["metadata"]["modele"] == "modele4_rapide"
    assert resultat["metadata"]["statut_flux_latent"] == surface.STATUT_FLUX_LATENT
    assert (resultat["capacite_surface_j_m2_k"] > 5.0e5).all()


def tester_sortie_npz_rapide():
    paquet = charger_paquet_grille()
    config = ConfigurationRapide(
        jours=1.0 / 6.0,
        max_latitudes=1,
        max_longitudes=1,
        facteur_latent=0.0,
        mode_convection="aucune",
        afficher_progression=False,
    )
    resultat = simuler_rapide(paquet, config)
    with tempfile.TemporaryDirectory() as dossier:
        chemin = Path(dossier) / "modele4_rapide_test.npz"
        enregistrer_resultat(resultat, chemin)
        with np.load(chemin) as npz:
            assert "temperature_surface_k" in npz.files
            assert "jours" in npz.files
            assert "heures" in npz.files
            assert "mois_precalcules" in npz.files
            assert npz["temperature_surface_k"].shape == (2, 1, 1)


def tester_coherence_complet_rapide_un_pas():
    paquet = charger_paquet_grille()
    config_surface = surface.ConfigurationSurface(facteur_latent=0.0, mode_convection="aucune")
    config_complet = ConfigurationModele4(
        jours=1.0 / 48.0,
        dt_s=1800.0,
        temperature_initiale_k=300.0,
        frequence_sortie_pas=1,
        iterations_implicites=4,
        indices_lat=(18,),
        indices_lon=(36,),
        surface=config_surface,
    )
    config_rapide = ConfigurationRapide(
        jours=1.0 / 48.0,
        dt_s=1800.0,
        sortie_heures=0.5,
        temperature_initiale_k=300.0,
        indices_lat=(18,),
        indices_lon=(36,),
        facteur_latent=0.0,
        mode_convection="aucune",
        afficher_progression=False,
    )

    complet = simuler(paquet, config_complet)
    rapide = simuler_rapide(paquet, config_rapide)
    temperature_complete = complet["temperature_surface_k"]
    temperature_rapide = rapide["temperature_surface_k"]

    assert temperature_complete.shape == temperature_rapide.shape == (2, 1, 1)
    assert abs(float(temperature_complete[0, 0, 0] - temperature_rapide[0, 0, 0])) < 1e-6
    assert abs(float(temperature_complete[-1, 0, 0] - temperature_rapide[-1, 0, 0])) < 2e-2


def main():
    tester_sortie_4h_par_defaut_sur_point()
    tester_sortie_npz_rapide()
    tester_coherence_complet_rapide_un_pas()
    print("tests_modele4_rapide_ok")


if __name__ == "__main__":
    main()
