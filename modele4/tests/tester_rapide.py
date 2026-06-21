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
from modele4.rapide import ConfigurationRapide, enregistrer_resultat, simuler_rapide


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
    assert np.isfinite(resultat["capacite_surface_j_m2_k"]).all()
    assert (resultat["capacite_surface_j_m2_k"] > 0.0).all()
    assert resultat["heures"].tolist() == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 0.0]
    assert resultat["mois_precalcules"].tolist() == [1]
    assert resultat["metadata"]["modele"] == "modele4_rapide"


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


def main():
    tester_sortie_4h_par_defaut_sur_point()
    tester_sortie_npz_rapide()
    print("tests_modele4_rapide_ok")


if __name__ == "__main__":
    main()
