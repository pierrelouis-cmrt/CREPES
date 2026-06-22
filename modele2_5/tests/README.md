# Tests — modèle 2.5

Ce dossier contient les vérifications numériques manuelles du noyau radiatif à
dix couches. Elles n'écrivent ni CSV ni image : elles affichent les résultats
et lèvent une erreur au premier invariant non respecté.

## Lancer

Depuis la racine du dépôt :

```bash
python modele2_5/tests/tester_modele2_5.py
```

## Vérifications effectuées

- limite transparente : l'OLR vaut `σT_surface⁴` et le flux descendant est nul ;
- grille verticale : dix couches, pression décroissante et altitude croissante ;
- monotonie : l'OLR diminue lorsque la concentration de CO₂ augmente ;
- forçage du doublement `280 → 560 ppm` : compris entre `3,70` et `4,10 W m⁻²` ;
- comportement proche d'une loi logarithmique sur deux doublements successifs.

En fin d'exécution, le script affiche aussi des cas de référence pour `280`,
`420`, `560` et `1120 ppm` afin de comparer les résultats entre deux versions.

## Structure

| Fichier | Rôle |
| --- | --- |
| `tester_modele2_5.py` | Lance les cinq vérifications et affiche les cas de référence. |

Le moteur testé est documenté dans le [README parent](../README.md). Le profil
vertical et les sorties associées sont documentés dans
[`../ressources/`](../ressources/README.md) et
[`../sorties/`](../sorties/README.md).
