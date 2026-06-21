# Modèle 2.5 - itération propre du modèle 2

Mini-glossaire : OLR = flux infrarouge sortant vers l'espace ; LW = infrarouge thermique à grande longueur d'onde.

Le modèle 2.5 garde le même objectif que le modèle 2 : une seule colonne
atmosphérique, températures imposées, flux infrarouges CO2 calculés par couches.
Il modifie seulement quatre points : grille verticale, profil de température,
découpage spectral CO2 et calibration optique.

## Fichiers

| Fichier                                        | Rôle                                                           |
| ---------------------------------------------- | -------------------------------------------------------------- |
| `modele2_5.py`                                 | Noyau radiatif CO2 autonome.                                   |
| `ressources/profil_temperature_standard.py`    | Profil $T(z)$ de l'atmosphère standard 1976 + CSV + graphique. |
| `ressources/profil_vertical_atmosphere_co2.py` | Profil pression-température-CO2 + CSV + graphique.             |
| `ressources/calibrer_coefficients_optiques.py` | Recalibre l'échelle des opacités sur le doublement CO2.        |
| `ressources/tester_modele2_5.py`               | Tests numériques séparés du modèle principal.                  |
| `requirements.txt`                             | Dépendances locales du modèle 2.5.                             |

## Lancer

Depuis la racine du dépôt :

```bash
./.venv/bin/python modele2_5/modele2_5.py
./.venv/bin/python modele2_5/ressources/tester_modele2_5.py
./.venv/bin/python modele2_5/ressources/calibrer_coefficients_optiques.py
```

Régénérer les profils :

```bash
./.venv/bin/python modele2_5/ressources/profil_temperature_standard.py --max-altitude-km 84 --step-m 100 --output modele2_5/ressources/profil_temperature_standard.png --csv modele2_5/ressources/profil_temperature_standard.csv --no-plot
./.venv/bin/python modele2_5/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 84 --step-m 100 --surface-co2-ppm 420 --output modele2_5/ressources/profil_vertical_atmosphere_co2.png --csv modele2_5/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

## Grille verticale

Le modèle 2 utilisait 6 couches définies par altitude. Le 2.5 utilise 10 couches
définies par pressions d'interface :

```text
1013.25, 850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa
```

Ce n'est pas un découpage régulier en hPa : les pressions choisies sont un
sous-ensemble de la liste de niveaux de pression ERA5/ECMWF citée dans les
sources. Comme la pression diminue presque exponentiellement avec l'altitude,
les écarts de pression deviennent plus petits en altitude, ce qui donne des
couches plus comparables en épaisseur verticale et garde une résolution minimale
dans la stratosphère. Ce choix est acceptable ici car le CO2 est bien mélangé ;
pour ajouter, par exemple, la vapeur d'eau plus tard, il faudra raffiner la
basse troposphère avec davantage d'interfaces entre environ $1000$ et
$700\ \mathrm{hPa}$.

La température de la couche $k$ est alors la moyenne du profil $T(z)$ entre ces
deux altitudes :

$$
\overline{T}_k =
\frac{1}{z_{haut,k}-z_{bas,k}}
\int_{z_{bas,k}}^{z_{haut,k}} T(z)\,dz
$$

Sortie actuelle :

| Couche | Pression hPa |   Altitude km | $\overline{T}$ K |
| ------ | -----------: | ------------: | ---------------: |
| 1      |  1013.25-850 |   0.000-1.458 |          283.413 |
| 2      |      850-700 |   1.458-3.014 |          273.624 |
| 3      |      700-500 |   3.014-5.579 |          260.242 |
| 4      |      500-300 |   5.579-9.177 |          240.248 |
| 5      |      300-200 |  9.177-11.806 |          220.831 |
| 6      |      200-100 | 11.806-16.221 |          216.650 |
| 7      |       100-50 | 16.221-20.643 |          216.688 |
| 8      |        50-20 | 20.643-26.592 |          220.180 |
| 9      |        20-10 | 26.592-31.207 |          225.418 |
| 10     |         10-1 | 31.207-48.183 |          249.487 |

## Profil standard

Le profil suit l'atmosphère standard 1976 en altitude géopotentielle $h$ : permet d'obtenir la courbe de la température en fonction de l'altitude 

h est l'altitude gepotentielle en m, une autre manière de définir l'altitude afin que g(z) (qu'on approxime à 9.81 et qui n'est pas constant en fct de l'altitude) soit constant. 
Donc g(h) = cst et g(z) pas cst
$$
h=\frac{r_0 z}{r_0+z}
$$

Dans chaque couche standard :
Lb correspond à la pente de chaque petit morceau de la courbe 
$$
T(h)=T_b+L_b(h-h_b)
$$

Si $L_b\ne0$ :

$$
p(h)=p_b\left(\frac{T(h)}{T_b}\right)^{-g_0/(R_{air}L_b)}
$$

Si $L_b=0$ :

$$
p(h)=p_b\exp\left[-\frac{g_0(h-h_b)}{R_{air}T_b}\right]
$$

Le graphique `ressources/profil_temperature_standard.png` montre la courbe en zigzag
attendue : troposphère refroidissante, tropopause, stratosphère réchauffante,
puis mésosphère refroidissante.

## CO2 moyen par couche

Le CO2 reste bien mélangé par défaut :

$$
C(z)=420\ \mathrm{ppm}
$$

La moyenne par couche reste pondérée par la masse d'air, donc par $\Delta p$ :

$$
\overline{C}_k =
\frac{\int C(p)\,dp}{\int dp}
$$

## Bandes CO2

Les bandes retenues sont :

- $15\ \mu m$, principale bande thermique terrestre ;
- $4.3\ \mu m$, bande CO2 forte sur le spectre d'absorption du CO2 mais faible dans les longueurs d'onde émise par la terre à 288 K.

La bande $2.7\ \mu m$ est négligée : son diagnostic était proche de $0.001\ \mathrm{W\,m^{-2}}$ pour $280\to560\ ppm$.

Le découpage cœur/ailes est conservé : le diagnostic montre qu'il change la
réponse marginale au doublement CO2. Coefficients effectifs utilisés dans
$\Delta\tau$, après calibration :

| Sous-bande                   | $\lambda$ µm | Rôle  |    $a_b$ |
| ---------------------------- | -----------: | ----- | -------: |
| CO2_15um_aile_gauche_externe |  13.00-14.00 | aile  | 0.010471 |
| CO2_15um_aile_gauche_interne |  14.00-14.60 | aile  | 0.114530 |
| CO2_15um_coeur_sature        |  14.60-15.40 | cœur | 1.308912 |
| CO2_15um_aile_droite_interne |  15.40-16.20 | aile  | 0.130891 |
| CO2_15um_aile_droite_externe |  16.20-18.00 | aile  | 0.015707 |
| CO2_4_3um_aile_gauche        |    4.00-4.20 | aile  | 0.006545 |
| CO2_4_3um_coeur_sature       |    4.20-4.40 | cœur | 0.490842 |
| CO2_4_3um_aile_droite        |    4.40-4.60 | aile  | 0.006545 |

## Opacité et flux

La profondeur optique d'une couche reste :

$$
\Delta\tau_{k,b} =
a_b
\frac{\overline{C}_k}{C_0}
\frac{\Delta p_k}{p_s}
$$

avec :

$$
C_0=280\ \mathrm{ppm}, \qquad p_s=101325\ \mathrm{Pa}
$$

Le facteur diffusif n'est plus $D=1$. On utilise :

$$
D=1.66
$$

Donc :

$$
\mathcal{T}_{k,b}=\exp(-D\Delta\tau_{k,b}),
\qquad
\varepsilon_{k,b}=1-\mathcal{T}_{k,b}
$$

Les flux par bande restent ceux du modèle 2 :

$$
F^\uparrow_{k+1,b}
=
\mathcal{T}_{k,b}F^\uparrow_{k,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k)
$$

$$
F^\downarrow_{k,b}
=
\mathcal{T}_{k,b}F^\downarrow_{k+1,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k)
$$

## Calibration

Les coefficients $a_b$ sont effectifs : ils ne remplacent pas HITRAN ligne par
ligne. Ils sont calibrés pour que, à températures fixées :

$$
OLR(280\ ppm)-OLR(560\ ppm)=3.93\ \mathrm{W\,m^{-2}}
$$

Résultat du script :

```text
echelle_opacite_co2 = 0.0327228010
forcage_280_560_ppm_W_m2 = 3.930000
```

Diagnostic de contribution :

| Sélection          | Forçage 280->560 ppm W/m² |
| ------------------ | ------------------------: |
| Toutes sous-bandes |                  3.930000 |
| 15 µm total        |                  3.816900 |
| 15 µm cœur        |                  1.818766 |
| 15 µm ailes        |                  1.998134 |
| 4.3 µm total       |                  0.113100 |

## Résultats de référence

Sortie du test séparé :

| CO2 ppm |   OLR W/m² | LW descendant surface W/m² |
| ------: | ---------: | -------------------------: |
|     280 | 381.901210 |                  13.322620 |
|     420 | 379.639619 |                  16.503798 |
|     560 | 377.971210 |                  18.900362 |
|    1120 | 373.687516 |                  25.437590 |

Ces flux ne sont pas le budget infrarouge terrestre complet : vapeur d'eau,
nuages, ozone et convection restent absents comme dans le modèle 2.

## Tests

`ressources/tester_modele2_5.py` vérifie :

- opacité nulle : $OLR=\sigma T_s^4$ et $LW_{\downarrow,surface}=0$ ;
- 10 couches, pression descendante, altitude montante ;
- $OLR$ diminue quand le CO2 augmente à températures fixées ;
- $280\to560\ ppm$ reste entre $3.70$ et $4.10\ \mathrm{W\,m^{-2}}$ ;
- deux doublements successifs restent proches d'une réponse logarithmique.

## Sources principales

- U.S. Standard Atmosphere 1976, NOAA/NASA/USAF :
  https://ntrs.nasa.gov/citations/19770009539
- IPCC AR6 WGI, chapitre 7, supplément : $F_{2xCO2}=3.93\ \mathrm{W\,m^{-2}}$ :
  https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf
- Myhre et al. (1998), formule logarithmique classique du CO2 :
  https://doi.org/10.1029/98GL01908
- HITRAN, base spectroscopique de référence pour les raies moléculaires :
  https://hitran.org/
- Gordon et al. (2022), HITRAN2020 :
  https://doi.org/10.1016/j.jqsrt.2021.107949
- Amundsen et al. (2014), approximation two-stream/correlated-k avec $D=1.66$ :
  https://arxiv.org/abs/1402.0814
- ECMWF/Copernicus ERA5, liste des niveaux de pression disponibles :
  https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Levellistings
