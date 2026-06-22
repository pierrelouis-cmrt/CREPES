# Carte des continents

Ce shapefile Natural Earth des pays est utilisé par `physique/chaleur_latente.py`. Il associe un point latitude/longitude à un continent et permet d'appliquer le flux latent moyen correspondant.

Conserver ensemble les fichiers `ne_110m_admin_0_countries` : `.shp`, `.shx`, `.dbf`, `.prj` et, si présent, `.cpg`. La table attributaire doit contenir le champ `CONTINENT`.

Sans ce shapefile, ou sans `geopandas` et `shapely`, les points sont traités comme océaniques. Le moteur reste exécutable mais les flux latents continentaux ne sont plus différenciés.
