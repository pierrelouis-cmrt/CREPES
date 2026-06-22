# Caches locaux

Ce dossier est rempli automatiquement lorsqu'un script consulte NASA POWER. Il évite de répéter les appels réseau pour les mêmes coordonnées et périodes.

| Fichier | Contenu |
| --- | --- |
| `wind_*.csv` | Série journalière de vitesse du vent. |
| `nasa_albedo_cache.csv` | Albédo moyen utilisé par le repli NASA POWER. |

Ces fichiers ne sont pas des données de référence : ils peuvent être supprimés pour forcer une nouvelle récupération. Le moteur garde un repli stable si le réseau est inaccessible.
