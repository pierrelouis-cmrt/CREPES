# Sorties — annexes

Ce dossier rassemble les figures et diagnostics produits par les scripts
exploratoires de l'annexe. Aucun modèle du dépôt ne lit ces fichiers.

## Structure

| Fichier | Rôle |
| --- | --- |
| `absorbance_ch4.png` | Spectre infrarouge du CH₄ calculé avec RADIS/HITRAN. |

## Régénérer le spectre CH₄

Depuis la racine du dépôt :

```bash
python annexe/codes_python/spectre_absorbance_ch4.py --no-plot --output annexe/sorties/absorbance_ch4.png
```

Le premier lancement requiert une connexion Internet pour que RADIS télécharge
les données HITRAN du méthane. La commande remplace l'image si elle existe déjà.
Les paramètres physiques du calcul sont détaillés dans le
[README des codes Python](../codes_python/README.md).
