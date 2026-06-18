**ERA5 Pressure Levels**
Fichier : `ressources/db7d35d0a9c6110c5f6d54212de24b21.nc`

Variables cochées/téléchargées :

- `t` : Temperature, en K
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

- `avg_sdlwrf` : mean surface downward long-wave radiation flux
- `avg_sdlwrfcs` : same, clear-sky
- `avg_sdswrf` : mean surface downward short-wave radiation flux
- `avg_sdswrfcs` : same, clear-sky
- `avg_slhtf` : mean surface latent heat flux
- `avg_snlwrf` : mean surface net long-wave radiation flux
- `avg_snlwrfcs` : same, clear-sky
- `avg_snswrf` : mean surface net short-wave radiation flux
- `avg_snswrfcs` : same, clear-sky
- `avg_ishf` : mean surface sensible heat flux
- `avg_tdswrf` : mean top downward short-wave radiation flux
- `avg_tnlwrf` : mean top net long-wave radiation flux
- `avg_tnlwrfcs` : same, clear-sky
- `avg_tnswrfcs` : mean top net short-wave radiation flux, clear-sky

Contexte vérifié : ce sont des moyennes mensuelles ERA5 pour 2024, 12 mois, grille `0.25°`. Le modèle 3 final n’utilise plus tout : il garde surtout `t/q`, `sp/t2m/skt/lsm/siconc/sd`, l’albédo CSV, et les flux `avg_sdlwrf`, `avg_snswrf`, `avg_tnlwrf`, `avg_sdswrf`.
