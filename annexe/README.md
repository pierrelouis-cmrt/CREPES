# Annexes

Ce dossier rassemble des scripts d'exploration et de préparation de données.
Ils sont indépendants des modèles exécutés normalement : aucun des fichiers de
ce dossier n'est importé automatiquement par les modèles 0 à 5.

## Structure

| Élément | Rôle |
| --- | --- |
| `codes_python/` | Scripts d'étude du CO₂ et du CH₄. |
| `sorties/` | Images générées par le script d'absorbance. |

## Contenu

Les annexes couvrent deux sujets :

- le téléchargement et le sous-échantillonnage de concentrations de CO₂ CAMS
  sur une grille mondiale de `5° × 5°` ;
- des prototypes de profil, spectre et colonne radiative simplifiée pour le
  méthane CH₄.

Les commandes, dépendances, paramètres et formats de sortie sont documentés
dans le [README des codes Python](codes_python/README.md).

> Ces outils sont exploratoires. Le CSV CO₂ produit et les diagnostics CH₄ ne
> sont pas lus automatiquement par le modèle 3 ni par les modèles de surface.
