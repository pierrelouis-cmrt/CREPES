# Recherche court-onde et optimisation pour le modele 4

Objectif : corriger ou contourner le biais court-onde du modele 3.1 sans
ajouter un coefficient arbitraire. La solution doit rester compatible avec le
paquet compact 3.1 et assez simple pour un projet de fin d'annee.

## Constat local

Le paquet 3.1 contient deja :

- `era5_sw_down_surface_w_m2` : flux court-onde descendant moyen a la surface ;
- `era5_sw_net_surface_w_m2` : flux court-onde net moyen a la surface ;
- `albedo_surface` : albedo mensuel de surface ;
- `albedo_nuages_effectif` : effet court-onde nuageux effectif CERES au sommet
  de l'atmosphere.

La formule actuelle du modele 3.1 est :

```text
SW_absorbe_surface =
    SW_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Elle surestime fortement le flux absorbe a la surface parce qu'elle applique
une correction nuageuse TOA comme si elle representait toute la transmission
atmospherique de surface. Elle ignore aussi l'absorption et la diffusion par
l'atmosphere claire.

Controle numerique sur le paquet actuel, compare a `era5_sw_net_surface_w_m2`.
La colonne "ERA5 down * (1 - albedo)" correspond au forcage mensuel de
reference. La recommandation finale ci-dessous garde plutot `S0*cos(i)` au pas
de temps et utilise ERA5 pour construire une transmissivite mensuelle.

| Mois | Formule actuelle, biais moyen | ERA5 down * (1 - albedo), biais moyen |
| --- | ---: | ---: |
| Janvier | +75.63 W/m2 | +5.12 W/m2 |
| Avril | +78.61 W/m2 | +2.63 W/m2 |
| Juillet | +98.66 W/m2 | +1.52 W/m2 |
| Octobre | +74.02 W/m2 | +2.42 W/m2 |

Pour Paris en juillet :

```text
ERA5 SW_down_surface = 228.4 W/m2
albedo_surface       = 0.1726
ERA5 down*(1-alpha)  = 188.98 W/m2
ERA5 SW_net_surface  = 188.80 W/m2
modele 3.1 actuel    = 334.62 W/m2
```

## Sources scientifiques utilisees

### ERA5 / ECMWF

La documentation ERA5 indique que les flux moyens sont des moyennes temporelles
en W/m2, et que les moyennes mensuelles de moyennes journalieres couvrent le
mois complet. Elle liste explicitement :

- `mean_surface_downward_short_wave_radiation_flux`, W m**-2 ;
- `mean_surface_net_short_wave_radiation_flux`, W m**-2 ;
- `mean_surface_downward_short_wave_radiation_flux_clear_sky`, W m**-2.

Source :

```text
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
```

Conclusion pour le projet : les champs ERA5 du paquet sont des forcages
court-onde de surface directement utilisables, pas seulement des diagnostics.

### Constante solaire S0

Dans le code actuel, `S0` vaut `1361 W/m2` :

- modele 0 : `modele0_maintenance/codes_python/physique/solaire.py`,
  `constante_solaire = 1361.0` ;
- modele 3 et 3.1 : `CONSTANTE_SOLAIRE = 1361.0`.

Cette valeur represente l'irradiance solaire totale au sommet de l'atmosphere,
sur une surface perpendiculaire aux rayons solaires, a environ 1 unite
astronomique. Elle correspond aussi a la valeur nominale adoptee par l'IAU en
2015 pour l'irradiance solaire totale nominale.

Source :

```text
https://arxiv.org/abs/1605.09788
```

### FAO-56 / bilan radiatif de surface

FAO-56 rappelle que le rayonnement solaire atteignant la surface depend de la
position du soleil, de la turbidite atmospherique et des nuages. Il donne le
bilan court-onde net de surface sous la forme :

```text
R_ns = (1 - alpha) * R_s
```

avec `R_s` le rayonnement solaire atteignant la surface et `alpha` l'albedo. La
meme source donne aussi une estimation de ciel clair :

```text
R_so = (0.75 + 2e-5 * z) * R_a
```

et indique qu'une fraction `R_s / R_a` typique varie approximativement de 0.25
par ciel tres couvert a 0.75 par ciel clair.

Source :

```text
https://www.fao.org/4/x0490e/x0490e07.htm
```

Conclusion pour le projet : utiliser `SW_down_surface * (1 - albedo_surface)`
est une formule standard et defendable. Si on veut garder une geometrie solaire
interne, une transmission effective `tau_SW = SW_down_surface / SW_TOA_local`
est aussi defendable, car elle correspond au rapport `R_s / R_a`.

### NASA CERES EBAF-TOA

CERES EBAF-TOA Edition 4.2.1 fournit des moyennes mensuelles des flux au sommet
de l'atmosphere, avec flux tout temps et ciel clair. C'est une excellente source
pour un effet radiatif nuageux TOA, mais ce n'est pas une mesure directe de la
transmission solaire jusqu'a la surface.

Source :

```text
https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1
```

Conclusion pour le projet : `albedo_nuages_effectif` doit rester un diagnostic
ou une source de comparaison TOA. Il ne doit pas etre le facteur principal du
court-onde surface dans le modele 4.

### NASA POWER

NASA POWER documente que ses donnees solaires sont derivees d'observations
satellitaires et de modeles d'assimilation, avec des produits globaux continus.
Les CSV d'albedo utilises dans le projet viennent historiquement de rapports
de flux court-onde surface.

Sources :

```text
https://power.larc.nasa.gov/docs/tutorials/parameters/
https://power.larc.nasa.gov/docs/methodology/
```

Conclusion pour le projet : l'albedo mensuel de surface est une entree valable
pour `R_ns = (1 - alpha) * R_s`, mais il ne suffit pas a lui seul a representer
la transmission atmospherique.

## Recommandation court-onde

### Choix par defaut pour le modele 4

Garder la geometrie solaire du projet, puis corriger par une transmissivite
mensuelle derivee d'ERA5 :

```text
SW_TOA_local(t) = S0 * max(cos(i(t)), 0)

tau_SW_mensuel =
    era5_sw_down_surface_w_m2
  / moyenne_mensuelle(S0 * max(cos(i), 0))

SW_down_surface(t) =
    tau_SW_mensuel * SW_TOA_local(t)

SW_absorbe_surface(t) =
    SW_down_surface(t) * (1 - albedo_surface)
```

Raisons :

- formule physique standard du bilan court-onde net ;
- conserve le calcul solaire historique du projet ;
- garde le cycle jour/nuit et la saisonnalite venant de `S0*cos(i)` ;
- utilise ERA5 seulement pour representer une transmission atmospherique
  moyenne mensuelle ;
- conserve l'albedo de surface explicite du projet ;
- evite d'appliquer directement un effet nuageux TOA a la surface ;
- force la moyenne mensuelle de `SW_down_surface` a rester proche d'ERA5, sans
  remplacer tout le signal temporel par ERA5.

### Mode encore plus simple

Pour une premiere version tres robuste, mais moins pedagogique :

```text
SW_absorbe_surface =
    era5_sw_down_surface_w_m2 * (1 - albedo_surface)
```

Pour un mode de validation direct :

```text
SW_absorbe_surface = colonne.validation_flux["era5_sw_net_surface_w_m2"]
```

Ces deux modes sont utiles pour verifier les ordres de grandeur, mais ils
affaiblissent l'enjeu du projet si on les utilise comme seul court-onde dans la
boucle temporelle.

### Mode a eviter par defaut

Eviter :

```text
SW_absorbe_surface =
    SW_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Cette formule peut rester comme mode pedagogique "TOA simplifie", mais elle ne
doit pas piloter la temperature du modele 4.

## Implementation simple conseillee

Le modele 3.1 doit fournir un helper court-onde propre et le modele 4 doit
l'utiliser dans sa boucle. Principe :

1. Precalculer une moyenne mensuelle de `S0 * max(cos(i), 0)` sur la grille.
2. Construire `tau_SW_mensuel` depuis `era5_sw_down_surface_w_m2`.
3. A chaque pas du modele 4, recalculer `S0 * max(cos(i(t)), 0)`.
4. Appliquer `tau_SW_mensuel` puis l'albedo de surface.
5. Appeler `calculer_colonne_radiative` pour obtenir le long-onde :
   `LW_down_absorbe_surface`, `LW_up_surface`, diagnostics CO2/H2O.

```python
def court_onde_surface_modele4(colonne, jour_annee, heure_solaire):
    surface = colonne["surface"]
    sw_toa = flux_solaire_incident(
        surface["latitude_deg"],
        jour_annee,
        heure_solaire,
    )
    tau_sw = surface["transmissivite_sw_mensuelle"]
    albedo = surface["albedo_surface"]
    return sw_toa * tau_sw * (1.0 - albedo)
```

Le bilan du modele 4 devient alors :

```text
C_surface dT/dt =
    SW_absorbe_surface_corrige
  + LW_down_absorbe_surface_3_1
  - LW_up_surface_3_1
  - autres termes de surface du modele 0
```

## Optimisation du temps par colonne

Le cout principal du modele 3.1 vient de `flux_corps_noir_dans_bande`, qui
integre numeriquement Planck avec 2000 pas pour chaque bande et beaucoup de
temperatures. Ce calcul est repete alors que les temperatures de couches
mensuelles changent peu ou pas pendant la boucle du modele 4.

Mesure locale :

```text
baseline 100 colonnes : 6.56 s, soit environ 65 ms/colonne
cache simple Planck   : 0.87 s au premier passage, soit gain x7.6
cache rechauffe       : environ 0.2 ms/colonne sur un meme mois
```

Optimisation recommandee, simple et lisible :

1. Ajouter un cache `functools.lru_cache` autour de l'integrale de Planck par
   bande et temperature.
2. Commencer par un cache exact pour ne pas changer les resultats.
3. Si besoin, arrondir la temperature a 0.1 K ou 0.05 K pour augmenter les
   repetitions, en documentant l'erreur numerique.
4. Garder une option de test qui compare le mode cache et le mode direct sur
   quelques colonnes.

Exemple de principe :

```python
from functools import lru_cache

@lru_cache(maxsize=50000)
def flux_corps_noir_dans_bande_cache(temperature_k, lambda_min_um, lambda_max_um):
    return flux_corps_noir_dans_bande_direct(
        temperature_k,
        lambda_min_um,
        lambda_max_um,
    )
```

Ce n'est pas de l'informatique avancee : c'est seulement eviter de refaire une
integrale numerique identique des milliers de fois.

## Decision finale

Pour le modele 4, le meilleur compromis rigueur/simplicite est :

```text
court-onde par defaut =
    S0*cos(i,t) * transmissivite_SW_mensuelle_ERA5 * (1 - albedo_surface)

mode forcage simple =
    ERA5 SW_down_surface * (1 - albedo_surface)

mode validation =
    ERA5 SW_net_surface direct

mode ancien TOA =
    diagnostic seulement
```

Cette decision est mieux sourcee que l'ancien coefficient nuageux, s'appuie sur
des flux de surface deja presents dans le paquet, garde le coeur solaire du
projet, reduit fortement le biais mensuel, et reste facile a expliquer a
l'oral.
