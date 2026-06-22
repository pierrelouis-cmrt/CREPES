# Calibrage des coefficients CO2 du modele 3

Ce document decrit la methode rigoureuse a utiliser pour remplacer les
coefficients `a_co2` pedagogiques du modele 3 par des coefficients effectifs
calibres depuis un spectre de reference HITRAN.

Le point important est le suivant : les coefficients du modele 3 ne deviennent
pas des sections efficaces spectroscopiques. Ils restent des coefficients
effectifs, mais ils sont derives proprement d'un calcul ligne par ligne puis
recales sur une cible de forcage documentee.

## Formule du modele 3

Dans `modele3/physique.py`, la profondeur optique CO2 d'une couche et d'une
bande vaut :

```text
tau_CO2[k,b] =
    a_CO2[b] * (CO2_ppm / 280) * (delta_p[k] / 101325)
```

La transmission utilisee par le transfert long-onde est ensuite :

```text
T[k,b] = exp(-1.66 * (tau_CO2[k,b] + tau_H2O[k,b]))
```

Le calibrage cherche donc des `a_CO2[b]` compatibles avec cette forme simple.

## Script dedie

Le script est :

```text
modele3/calibrer_coefficients_co2.py
```

Dependances optionnelles :

```bash
./.venv/bin/python -m pip install -r modele3/requirements-calibrage.txt
```

Selon la configuration RADIS/HITRAN utilisee, un compte HITRAN et une cle API
peuvent etre necessaires pour telecharger les donnees. La page HAPI officielle
explique l'installation `hitran-api` et l'usage des cles API :
https://hitran.org/hapi/

Estimer le volume de calcul avant de lancer HITRAN/RADIS :

```bash
./.venv/bin/python -m modele3.calibrer_coefficients_co2 --dry-run
```

Lancer un calibrage raisonnable :

```bash
./.venv/bin/python -m modele3.calibrer_coefficients_co2 \
  --latitudes=-60,-30,0,30,60 \
  --longitudes=0 \
  --mois=1,4,7,10 \
  --co2-values=280,420,560,1120 \
  --wstep 0.02
```

Le script produit :

```text
modele3/ressources/calibrage_opacite_co2/calibrage_coefficients_co2.json
modele3/ressources/calibrage_opacite_co2/coefficients_co2_calibres.py
```

Le JSON contient les mesures spectrales compressees, les coefficients avant et
apres calibrage, le facteur global de forcage et les impacts par bande.

## Methode exacte

### 1. Choisir les bandes CO2

Le script reprend les bandes CO2 deja definies dans `physique.py` :

```text
CO2_15um_aile_gauche_externe
CO2_15um_aile_gauche_interne
CO2_15um_coeur_sature
CO2_15um_aile_droite_interne
CO2_15um_aile_droite_externe
CO2_4_3um_aile_gauche
CO2_4_3um_coeur_sature
CO2_4_3um_aile_droite
```

Chaque bande garde son role physique : `aile` ou `coeur sature`.

### 2. Calculer un spectre de reference HITRAN

Pour chaque couche ERA5 echantillonnee et chaque bande, le script appelle RADIS
avec la databank HITRAN :

```text
calc_spectrum(
    wmin, wmax,
    molecule="CO2",
    pressure=p_milieu,
    Tgas=T_couche,
    mole_fraction=CO2_ppm * 1e-6,
    path_length=epaisseur_couche_cm,
    databank=("hitran", "range")
)
```

La couche est traitee comme homogene. Son epaisseur geometrique est reconstruite
depuis l'hydrostatique et le gaz parfait :

```text
masse_air = delta_p / g
rho_air = p_milieu * M_air / (R * T_couche)
epaisseur = masse_air / rho_air
```

Cette etape donne une transmission haute resolution :

```text
T_ref[k,b,nu,CO2]
```

### 3. Compresser le spectre en une transmission de bande

On ne moyenne pas directement les absorptions. On moyenne la transmission avec
un poids de Planck, car les flux thermiques du modele sont des integrales de
Planck par bande :

```text
Tbar_ref[k,b,C] =
    integral_bande B_nu(T_k) * T_ref[k,b,nu,C] dnu
    ------------------------------------------------
    integral_bande B_nu(T_k) dnu
```

Le script travaille en nombre d'onde `nu` en `cm-1`. Les constantes
multiplicatives de Planck s'annulent dans le rapport.

### 4. Convertir la transmission en profondeur optique equivalente

Le modele 3 utilise :

```text
T = exp(-D * tau)
D = 1.66
```

Donc la profondeur optique equivalente de reference est :

```text
tau_eq_ref[k,b,C] = -ln(Tbar_ref[k,b,C]) / 1.66
```

Le facteur `1.66` est retire ici parce qu'il sera remis par le modele 3 au
moment du calcul de transmission.

### 5. Ajuster chaque coefficient `a_CO2[b]`

Pour chaque bande, on pose :

```text
X[k,C] = (C / 280) * (delta_p[k] / 101325)
```

Le modele impose :

```text
tau_eq_ref[k,b,C] ~= a_CO2[b] * X[k,C]
```

Le coefficient brut HITRAN est obtenu par moindres carres ponderes :

```text
a_CO2[b] =
    sum(w * X * tau_eq_ref)
    -----------------------
    sum(w * X^2)
```

Par defaut, le poids est :

```text
w = poids_aire_colonne * flux_Planck_bande(T_couche)
```

Cela donne plus de poids aux colonnes qui representent plus de surface et aux
couches/bandes qui contribuent davantage au flux thermique.

### 6. Recaler globalement sur le forcage CO2

Apres le fit spectral, le script calcule le forcage du modele 3 a temperature
fixee :

```text
DeltaF_2x = OLR(280 ppm) - OLR(560 ppm)
```

Puis il cherche par bissection un facteur global `s` tel que :

```text
DeltaF_2x(s * a_CO2) ~= cible
```

La cible par defaut est :

```text
5.35 * ln(2) = 3.708 W m-2
```

C'est la cible la plus coherente pour un test a temperature fixee, car elle
vient de la formule de Myhre et al. :

```text
DeltaF = 5.35 ln(C / C0)
```

On peut aussi utiliser la valeur AR6 :

```bash
--cible-forcage 3.93
```

Mais il faut alors preciser dans le rapport que `3.93 W m-2` est un ERF AR6
avec ajustements, alors que le modele 3 calcule plutot une variation OLR
instantanee a profil fixe.

### 7. Quantifier coeur contre ailes

Le script ne juge pas l'importance d'une bande avec la taille de `a_CO2`.
Il calcule son impact radiatif direct :

```text
impact_b = OLR_b(280 ppm) - OLR_b(560 ppm)
```

Puis :

```text
part_b = impact_b / sum_b(impact_b)
```

Il agrège aussi :

```text
impact_coeur = somme des bandes dont role contient "coeur"
impact_ailes = somme des bandes dont role contient "aile"
```

C'est la bonne metrique pour expliquer la saturation :

- le coeur peut avoir un `a_CO2` tres grand ;
- mais si sa transmission est deja presque nulle, augmenter le CO2 change peu
  l'OLR ;
- les ailes ont souvent un `a_CO2` plus faible, mais une sensibilite marginale
  plus forte.

## Ce qui est source

| Element | Source |
| --- | --- |
| Donnees spectroscopiques ligne par ligne | HITRAN / HAPI : https://hitran.org/hapi/ |
| Interface Python de spectres HITRAN | RADIS `calc_spectrum` : https://radis.readthedocs.io/en/latest/source/radis.lbl.calc.html |
| Formule logarithmique CO2 | Myhre et al. 1998 : https://doi.org/10.1029/98GL01908 |
| Valeur AR6 `2xCO2 = 3.93 W m-2` | IPCC AR6 WGI Chapter 7 SM : https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf |
| Profils locaux T(p), q(p) | ERA5 pressure levels : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels |

## Limites explicites

- Les `a_CO2` calibres restent des coefficients effectifs de bande.
- Le script ne transforme pas le modele 3 en modele ligne par ligne.
- La couche est supposee homogene en pression, temperature et CO2.
- La methode ne calibre pas encore les coefficients H2O.
- Le recalage global force le bon ordre de grandeur du doublement CO2, mais ne
  garantit pas que chaque sous-bande soit parfaite.
- Les resultats dependent de l'echantillon choisi (`latitudes`, `mois`,
  `longitudes`) et du pas spectral `wstep`.

## Phrase a mettre dans le rapport

Les coefficients CO2 du modele 3 sont calibres en deux temps. D'abord, un calcul
HITRAN/RADIS ligne par ligne fournit une transmission spectrale de reference
pour chaque couche et sous-bande CO2. Cette transmission est moyennee avec un
poids de Planck, convertie en profondeur optique equivalente, puis ajustee par
moindres carres sur la forme simplifiee du modele. Ensuite, tous les
coefficients CO2 sont multiplies par un facteur global afin que le modele
retrouve le forcage de reference du doublement du CO2. Les contributions du
coeur et des ailes sont enfin mesurees par leur variation d'OLR, et non par la
taille brute de leurs coefficients.
