**ERA5 Pressure Levels**
Fichier : `ressources/db7d35d0a9c6110c5f6d54212de24b21.nc`

Variables cochées/téléchargées :

- `t` : Température, en K
- `q` : Specific humidity, en `kg kg**-1`
- `cc` : Fraction of cloud cover, entre 0 et 1
- `z` : Geopotential, en `m**2 s**-2`

Niveaux de pression téléchargés :  
`1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1 hPa`.

**ERA5 Single / Surface Levels**
Fichiers : `ressources/4d43b9edb397c8d4595fc350432d5ac4/*.nc`

Variables surface classiques :

- `t2m` : 2 metre temperature, K
- `sst` : Sea surface temperature, K
- `sp` : Surface pressure, Pa
- `skt` : Skin temperature, K
- `hcc` : High cloud cover, 0-1
- `lcc` : Low cloud cover, 0-1
- `mcc` : Medium cloud cover, 0-1
- `tcc` : Total cloud cover, 0-1
- `fal` : Forecast albedo, 0-1
- `z` : Geopotential, `m**2 s**-2`
- `lsm` : Land-sea mask, 0-1
- `siconc` : Sea ice area fraction, 0-1
- `asn` : Snow albedo, 0-1
- `sd` : Snow depth, m water equivalent
- `cbh` : Cloud base height, m

Flux moyens téléchargés :

- `avg_sdlwrf`   : flux radiatif infrarouge descendant moyen à la surface
- `avg_sdlwrfcs` : flux radiatif infrarouge descendant moyen à la surface en ciel clair

- `avg_sdswrf`   : flux radiatif solaire descendant moyen à la surface
- `avg_sdswrfcs` : flux radiatif solaire descendant moyen à la surface en ciel clair

- `avg_slhtf`    : flux moyen de chaleur latente à la surface

- `avg_snlwrf`   : flux radiatif infrarouge net moyen à la surface
- `avg_snlwrfcs` : flux radiatif infrarouge net moyen à la surface en ciel clair

- `avg_snswrf`   : flux radiatif solaire net moyen à la surface
- `avg_snswrfcs` : flux radiatif solaire net moyen à la surface en ciel clair

- `avg_ishf`     : flux moyen de chaleur sensible à la surface

- `avg_tdswrf`   : flux radiatif solaire descendant moyen au sommet de l’atmosphère

- `avg_tnlwrf`   : flux radiatif infrarouge net moyen au sommet de l’atmosphère
- `avg_tnlwrfcs` : flux radiatif infrarouge net moyen au sommet de l’atmosphère en ciel clair

- `avg_tnswrfcs` : flux radiatif solaire net moyen au sommet de l’atmosphère en ciel clair

Contexte vérifié : ce sont des moyennes mensuelles ERA5 pour 2024, 12 mois, grille `0.25°`. Le modèle 3 final n’utilise plus tout : il garde surtout `t/q`, `sp/t2m/skt/lsm/siconc/sd`, l’albédo CSV, et les flux `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf`.
