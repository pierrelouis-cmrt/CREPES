# Ressources — modèle 0

Ce dossier est le point d'accès aux données lues par le modèle 0, à ses
résultats précalculés et aux caches locaux. Les chemins sont résolus dans
`codes_python/chemins.py`.

## Vérifier les ressources

Depuis la racine du dépôt :

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

## Structure

| Dossier | Rôle |
| --- | --- |
| `12_mois/` | Températures mensuelles utilisées par le viewer 3D. |
| `albedo/` | Albédo de surface mensuel et source CERES. |
| `caches/` | Réponses NASA POWER mémorisées localement. |
| `capacite_humidite/` | Données RZSM pour la capacité de surface. |
| `carte/` | Données cartographiques de pays. |
| `cotes/` | Données de traits de côte. |
| `grilles/` | Sorties de température pour les visualisations. |

Ces sous-dossiers contiennent des données détaillées ; ils sont conservés tels
quels et ne sont pas développés dans cette documentation.
