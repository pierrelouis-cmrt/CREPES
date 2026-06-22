# Ressources du modèle 0

Ce dossier rassemble les données lues par le moteur, les résultats précalculés et les caches locaux. Les chemins sont définis dans `codes_python/chemins.py`.

| Dossier | Contenu | Statut |
| --- | --- | --- |
| [`albedo/`](albedo/README.md) | Albédo de surface mensuel et fichier CERES. | Entrée du moteur. |
| [`capacite_humidite/`](capacite_humidite/README.md) | Humidité RZSM du sol. | Entrée externe locale. |
| [`carte/`](carte/README.md) | Pays Natural Earth. | Entrée externe locale. |
| [`cotes/`](cotes/README.md) | Lignes de côte Natural Earth. | Entrée de visualisation. |
| [`grilles/`](grilles/README.md) | Sorties `.npy` de température. | Générable. |
| [`12_mois/`](12_mois/README.md) | Extraits mensuels pour le globe 3D. | Générable. |
| [`caches/`](caches/README.md) | Réponses NASA POWER mémorisées. | Créé à la demande. |

Pour vérifier les ressources sans les modifier :

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --status
```

Les données générables sont détaillées dans le [README des générateurs](../outils_generation_donnees/README.md).
