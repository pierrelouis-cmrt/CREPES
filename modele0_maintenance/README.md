# Modèle 0 — maintenance

Le modèle 0 est la version historique maintenue du modèle de surface CREPES.
Il simule la température de surface en un point ou sur une grille, à partir du
rayonnement solaire, des albédos, de la capacité thermique, de la chaleur
latente et de la convection.

## Parcours de la documentation

| Dossier ou document | Contenu |
| --- | --- |
| [`codes_python/`](codes_python/README.md) | Moteur ponctuel, modules physiques, installation et visualisations. |
| [`outils_generation_donnees/`](outils_generation_donnees/README.md) | Création et inventaire des grilles et données dérivées. |
| [`ressources/`](ressources/README.md) | Données lues par le modèle et formats attendus. |
| [`documents_sources/`](documents_sources/README.md) | Documents PDF historiques provenant des groupes sources. |
| [`PROVENANCE.md`](PROVENANCE.md) | Origine des briques de code et des données. |

## Démarrage

La simulation ponctuelle et les prérequis sont décrits dans le
[README des codes Python](codes_python/README.md). Pour contrôler les données
présentes sans rien modifier :

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

## Organisation

```text
modele0_maintenance/
├── codes_python/                 moteur, physique et visualisations
├── outils_generation_donnees/    génération contrôlée des sorties
├── ressources/                   entrées, sorties et caches locaux
├── documents_sources/            PDF de référence historiques
└── PROVENANCE.md                 traçabilité
```

Les README des sous-dossiers sont complémentaires : les instructions générales
ne sont données qu'à leur emplacement le plus pertinent.
