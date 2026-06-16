# Theorie et provenance du modele 3.1

Ce document separe clairement les lois physiques standard, les coefficients
effectifs du projet, les donnees externes et les choix volontairement ignores.

## 1. Objet du modele

Le modele 3.1 est une colonne radiative locale :

```text
(donnees_colonne, T_surface, CO2_ppm) -> flux radiatifs
```

La temperature de surface est une entree imposee. Le modele 3.1 ne resout pas
le bilan thermique dans le temps ; ce sera le role du modele 4.

## 2. Long-onde infrarouge

La surface emet :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
epsilon_surface = 0.98
```

Le choix `0.98` est volontaire. Dans le modele 3, l'ocean pouvait valoir
`0.985`, la terre et neige/glace `0.98`. Cet ecart change peu les flux devant
les incertitudes du modele. 3.1 supprime donc cette branche pour rendre le
contrat plus lisible.

Pour chaque bande infrarouge :

```text
tau_CO2 = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
tau_H2O = a_H2O_bande * (masse_H2O / 10 kg m-2)
tau_total = tau_CO2 + tau_H2O
transmission = exp(-D * tau_total)
D = 1.66
```

Les coefficients `a_CO2_bande` et `a_H2O_bande` sont effectifs. Ils ne
remplacent pas HITRAN, RADIS ou une methode correlated-k. Ils servent a garder
un noyau pedagogique CO2 + H2O coherent avec le modele 3.

## 3. Vapeur d'eau

ERA5 fournit `q`, humidite specifique en `kg/kg`. Pour une couche :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

La masse de reference `10 kg m-2` est un choix de normalisation effectif,
visible dans le code (`MASSE_H2O_REFERENCE_KG_M2`). Elle donne une echelle
simple aux bandes H2O, sans pretendre resoudre le spectre raie par raie.

## 4. Nuages

Deux usages sont volontairement separes.

Court-onde :

```text
SW_absorbe_surface =
    SW_incident_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

`albedo_nuages_effectif` vient de CERES :

```text
(toa_sw_all_mon - toa_sw_clr_c_mon) / solar_mon
```

Interpretation : part supplementaire de solaire reflechi en ciel tout temps par
rapport au ciel clair, normalisee par le solaire entrant. Ce n'est pas un
albedo microphysique local du nuage.

Long-onde :

```text
tau_total = tau_CO2 + tau_H2O
```

Le modele 3 ajoutait `tau_nuage = 0.10 * fraction_nuageuse`. 3.1 le retire du
chemin par defaut, car ce coefficient pouvait ameliorer artificiellement l'OLR
sans etre suffisamment source. Les fractions nuageuses ERA5 restent conservees
comme diagnostics et donnees futures.

## 5. Solaire

La geometrie solaire suit la formule simple deja utilisee dans le projet :

```text
cos(theta) =
    sin(latitude) * sin(declinaison)
  + cos(latitude) * cos(declinaison) * cos(angle_horaire)
SW_incident_TOA_local = S0 * max(cos(theta), 0)
S0 = 1361 W m-2
```

L'option `--moyenne-journaliere-sw` moyenne cette formule sur 96 pas horaires.
Il n'y a pas d'ozone, aerosols, absorption solaire H2O ou diffusion Rayleigh
explicite dans 3.1.

## 6. Donnees et transformations

Le generateur `modele3_1/generer_donnees.py` effectue les transformations une
seule fois :

| Etape | Transformation | Sortie dans le paquet |
| --- | --- | --- |
| ERA5 profils | Selection grille 5 degres, moyennes en pression de `t`, `q`, `cc`. | `temperature_couche_k`, `humidite_specifique_couche_kgkg`, `fraction_nuageuse_couche`. |
| ERA5 surface | Selection grille 5 degres. | `pression_surface_hpa`, `temperature_2m_k`, `skin_temperature_k`, `cloud_total`, `low_cloud`, `medium_cloud`, `high_cloud`. |
| ERA5 flux | Selection grille 5 degres. | Flux de validation : LW bas, SW net surface, OLR, SW descendant. |
| CSV albedo | Selection au plus proche sur la grille 5 degres. | `albedo_surface`. |
| CERES | Formule effective nuageuse puis selection au plus proche. | `albedo_nuages_effectif`. |
| Quantification | Echelles documentees dans `metadata.json`. | `.npz` compact de 2,1 Mo. |

Le paquet genere ne stocke pas les 37 niveaux ERA5 bruts, ni les fichiers MODIS,
ni de diagnostics bande-par-couche. Ces donnees sont soit inutiles au calcul
normal, soit regenerables.

## 7. Validation a ne pas surinterpreter

Pour Paris, le point de grille 5 degres le plus proche est `47.5 N, 2.5 E`.
Avec `T_surface = 293 K`, juillet et moyenne journaliere SW, la commande actuelle
donne environ :

```text
LW_down_surface        = 349.91 W m-2
ERA5 LW_down_surface   = 364.20 W m-2
OLR modele             = 286.15 W m-2
ERA5 OLR               = 252.90 W m-2
SW_absorbe_surface     = 334.62 W m-2
ERA5 SW_net_surface    = 188.80 W m-2
```

Le long-onde reste dans un ordre de grandeur utile. L'OLR s'eloigne du modele 3
apres retrait du nuage long-onde arbitraire, ce qui est attendu. Le court-onde
reste trop eleve car 3.1 ne represente pas le transfert solaire atmospherique
complet. Cette limite est documentee au lieu d'etre cachee par calibration.

## Sources externes

- Copernicus ERA5 monthly averaged pressure levels : profils `t`, `q`, `cc`.
- Copernicus ERA5 monthly averaged single levels : `sp`, nuages, surface, flux.
- NASA POWER : provenance des CSV mensuels d'albedo de surface historiques.
- NASA CERES EBAF-TOA Ed4.2.1 : flux TOA tout temps, ciel clair et solaire.
- HITRAN/RADIS : references spectroscopiques non utilisees directement ici.
