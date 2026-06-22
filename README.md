# CREPES

Projet Climat, Groupe D, 2026

## Organisation

| Dossier                | Rôle                                                                                            |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| `annexe/` | Pistes pour l'année prochaine                                        |
| `modele0_maintenance/` | Ancien modèle combiné, conservé comme référence stable.                                         |
| `modele1/`             | Colonne radiative CO2 simplifiée à 3 couches.                                                   |
| `modele2/`             | Colonne atmosphérique CO2 à 6 couches avec noyau radiatif infrarouge simplifié.                 |
| `modele2_5/`           | Itération du modèle 2 : 10 couches en pression, profil standard, bandes CO2 découpées et tests. |
| `modele3/`             | Colonne radiative finale pour le modèle 4, avec paquet `.npz` compact et provenances explicites. |
| `modele4/`             | Grille de température de surface couplée au modèle 3 et aux termes de surface du modèle 0.       |
| `modele5/`             | Grille modèle 4 rapide avec échanges radiatifs horizontaux entre colonnes voisines.              |
| `visualisation/`       | Visualisation des modèles 4 et 5.                                      |


## Lancement des codes 
Bien effectuer la commande suivante :
```bash 
python -m pip install -r requirements.txt
```


## Modèle 0

Lancer une simulation courte depuis la racine :

```bash
python3 modele0_maintenance/codes_python/modele_courbe.py --lat 48.5 --lon 2.3 --days 2 --no-plot
```

Inventorier les données du modèle 0 :

```bash
python3 modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

La documentation complète du modèle 0 est dans
`modele0_maintenance/README.md`.

## Modèle 2

Lancer le noyau radiatif du modèle 2 :

```bash
./.venv/bin/python modele2/codes_python/modele2.py
```

Régénérer le profil vertical de pression et de CO2 :

```bash
./.venv/bin/python modele2/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --no-plot
```

La documentation détaillée du modèle 2 est dans `modele2/README.md`.

## Modèle 2.5

Lancer le noyau radiatif du modèle 2.5 :

```bash
./.venv/bin/python modele2_5/codes_python/modele2_5.py
```

Lancer les tests numériques séparés :

```bash
./.venv/bin/python modele2_5/ressources/tester_modele2_5.py
```

Régénérer les profils standard et CO2 :

```bash
./.venv/bin/python modele2_5/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 84 --step-m 100 --surface-co2-ppm 420 --output modele2_5/ressources/profil_vertical_atmosphere_co2.png --csv modele2_5/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

La documentation détaillée du modèle 2.5 est dans `modele2_5/README.md`.

## Modèle 3

Régénérer le paquet compact :

```bash
./.venv/bin/python -m modele3.ressources.generer_donnees --overwrite
```

Lancer une colonne depuis le paquet global :

```bash
cd modele3
./modele3.py
```

Avec des options depuis la racine :

```bash
./.venv/bin/python -m modele3 --lat 0 --lon 0 --mois 7 --temperature-surface 293.0 --moyenne-journaliere-sw
```

Lancer les tests :

```bash
./.venv/bin/python modele3/tests/tester_modele3.py
```

Documentation détaillée :

- `modele3/README.md`
- `modele3/documentation/THEORIE.md`
- `modele3/documentation/PROVENANCE_DONNEES.md`

## Modèle 4

Lancer le diagnostic mensuel global par défaut :

```bash
./.venv/bin/python -m modele4.codes_python.modele4
```

Lancer le moteur rapide, sortie toutes les 4 heures par défaut :

```bash
./.venv/bin/python -m modele4.codes_python.rapide
```

Lancer un test temporel court sur une cellule :

```bash
./.venv/bin/python -m modele4.modele4 --mode temporel --jours 0.020833333333333332 --max-latitudes 1 --max-longitudes 1 --frequence-sortie-pas 1 --output /tmp/modele4_test.npz
```

Lancer les tests :

```bash
./.venv/bin/python modele4/tests/tester_modele4.py
./.venv/bin/python modele4/tests/tester_rapide.py
```

Documentation détaillée :

- `modele4/README.md`
- `modele4/THEORIE.md`

## Modèle 5

Lancer le modèle couplé horizontal, sortie toutes les 4 heures par défaut :

```bash
./.venv/bin/python -m modele5.codes_python.modele5
```

Lancer une petite grille de développement :

```bash
./.venv/bin/python -m modele5.modele5 --jours 1 --max-latitudes 4 --max-longitudes 8 --output modele5/sorties/simulation_dev.npz
```

Comparer au modèle 4 rapide sans échange horizontal :

```bash
./.venv/bin/python -m modele5.modele5 --facteur-horizontal 0 --output modele5/sorties/simulation_sans_horizontal.npz
```

Lancer les tests :

```bash
./.venv/bin/python modele5/tests/tester_modele5.py
```

Documentation détaillée :

- `modele5/README.md`

## Annexes

Les scripts annexes sont indépendants des modèles principaux.

Lancer la colonne radiative simplifiée au CH₄ :

```bash
python annexe/codes_python/modele_ch4.py
```

Générer un profil vertical de CH₄ :

```bash
python annexe/codes_python/profil_atmosphere_ch4.py --no-plot --csv annexe/codes_python/profil_ch4.csv
```

Calculer et enregistrer le spectre CH₄ :

```bash
python annexe/codes_python/spectre_absorbance_ch4.py --no-plot --output annexe/sorties/absorbance_ch4.png
```

L'outil CAMS/CO₂ requiert un compte Copernicus ADS et une clé API renseignée
dans le script avant son lancement :

```bash
python annexe/codes_python/Fraction_massique_CO2.py
```

Documentation détaillée : `annexe/README.md`.

## Visualisation

Afficher ou choisir une sortie `.npz` des modèles 4 ou 5 depuis la racine :

```bash
python visualisation/planisphere.py --sorties modele4/sorties modele5/sorties
```

Pour ouvrir directement un fichier et exporter un PNG sans fenêtre :

```bash
python visualisation/planisphere.py --fichier modele5/sorties/simulation_modele5.npz --save visualisation/planisphere.png --no-show
```

Documentation détaillée : `visualisation/README.md`.
