# Sorties — modèle 1

Ce dossier conserve les graphiques de diagnostic produits autour de l'absorption
infrarouge du CO₂. Le moteur radiatif `modele1.py` ne lit pas ces images : il
utilise les absorbances moyennes définies dans son code.

## Structure

| Fichier | Rôle |
| --- | --- |
| `absorbance_CO2.png` | Spectre d'absorbance du CO₂ calculé avec RADIS/HITRAN. |

## Régénérer le spectre CO₂

Depuis la racine du dépôt :

```bash
python "modele1/ressources/absorbance CO2.py" --no-plot --output modele1/sorties/absorbance_CO2.png
```

RADIS peut devoir télécharger les données HITRAN au premier lancement ; une
connexion Internet est alors nécessaire. La commande remplace l'image existante.
Pour les paramètres et les options du script, voir le
[README des ressources](../ressources/README.md).
