# Recherche shortwave et optimisation pour le modèle 4

Objectif : documenter la correction du biais du flux shortwave du modèle 3 sans
ajouter un coefficient arbitraire. La solution reste compatible avec le paquet
compact du modèle 3 et assez simple pour un projet de fin d'année.

## Constat local

Le paquet compact du modèle 3 contient maintenant :

- `era5_sw_down_surface_w_m2` : flux shortwave descendant moyen à la surface ;
- `era5_sw_net_surface_w_m2` : flux shortwave net moyen à la surface ;
- `albedo_surface` : albédo mensuel de surface ;

Le biais corrigé venait de l'ancienne formule de travail :

```text
SW_absorbe_surface =
    SW_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Elle surestime fortement le flux absorbé à la surface parce qu'elle applique
une correction nuageuse TOA comme si elle représentait toute la transmission
atmosphérique de surface. Elle ignore aussi l'absorption et la diffusion par
l'atmosphère claire.

Un contrôle numérique interne a motivé cette correction, mais le tableau de
biais historique n'est pas conservé comme benchmark reproductible. La validation
versionnée est limitée aux invariants simples de `modele3/tests/tester_modele3.py`
et aux diagnostics intégrés dans le paquet
`modele3/ressources/donnees_precalculees/grille_5deg_2024/donnees_colonnes_5deg_2024.npz`.

## Sources scientifiques utilisées

### ERA5 / ECMWF

La documentation ERA5 indique que les flux moyens sont des moyennes temporelles
en W/m2, et que les moyennes mensuelles de moyennes journalières couvrent le
mois complet. Elle liste explicitement :

- `mean_surface_downward_short_wave_radiation_flux`, W m**-2 ;
- `mean_surface_net_short_wave_radiation_flux`, W m**-2 ;
- `mean_surface_downward_short_wave_radiation_flux_clear_sky`, W m**-2.

Source :

```text
https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation
```

Conclusion pour le projet : les champs ERA5 du paquet sont des entrees imposees
de flux shortwave de surface directement utilisables, pas seulement des diagnostics.

### Constante solaire S0

Dans le code actuel, `S0` vaut `1361 W/m2` :

- modèle 0 : `modele0_maintenance/codes_python/physique/solaire.py`,
  `constante_solaire = 1361.0` ;
- modèle 0 et modèle 3 : `CONSTANTE_SOLAIRE = 1361.0`.

Cette valeur représente l'irradiance solaire totale au sommet de l'atmosphère,
sur une surface perpendiculaire aux rayons solaires, à environ 1 unité
astronomique. Elle correspond aussi à la valeur nominale adoptée par l'IAU en
2015 pour l'irradiance solaire totale nominale.

Source :

```text
https://arxiv.org/abs/1605.09788
```

### FAO-56 / bilan radiatif de surface

FAO-56 rappelle que le rayonnement solaire atteignant la surface dépend de la
position du soleil, de la turbidité atmosphérique et des nuages. Il donne le
bilan shortwave net de surface sous la forme :

```text
R_ns = (1 - alpha) * R_s
```

avec `R_s` le rayonnement solaire atteignant la surface et `alpha` l'albédo. La
même source donne aussi une estimation de ciel clair :

```text
R_so = (0.75 + 2e-5 * z) * R_a
```

et indique qu'une fraction `R_s / R_a` typique varie approximativement de 0.25
par ciel très couvert à 0.75 par ciel clair.

Source :

```text
https://www.fao.org/4/x0490e/x0490e07.htm
```

Conclusion pour le projet : utiliser `SW_down_surface * (1 - albedo_surface)`
est une formule standard et défendable. Si on veut garder une géométrie solaire
interne, une transmission effective `tau_SW = SW_down_surface / SW_TOA_local`
est aussi défendable, car elle correspond au rapport `R_s / R_a`.

### NASA CERES EBAF-TOA

CERES EBAF-TOA Edition 4.2.1 fournit des moyennes mensuelles des flux au sommet
de l'atmosphère, avec flux tout temps et ciel clair. C'est une excellente source
pour un effet radiatif nuageux TOA, mais ce n'est pas une mesure directe de la
transmission solaire jusqu'à la surface.

Source :

```text
https://asdc.larc.nasa.gov/project/CERES/CERES_EBAF-TOA_Edition4.2.1
```

Conclusion pour le projet : `albedo_nuages_effectif` doit rester un diagnostic
ou une source de comparaison TOA. Il ne doit pas être le facteur principal du
flux shortwave de surface dans le modèle 4.

### NASA POWER

NASA POWER documente que ses données solaires sont dérivées d'observations
satellitaires et de modèles d'assimilation, avec des produits globaux continus.
Les CSV d'albédo utilisés dans le projet viennent historiquement de rapports
de flux shortwave de surface.

Sources :

```text
https://power.larc.nasa.gov/docs/tutorials/parameters/
https://power.larc.nasa.gov/docs/methodology/
```

Conclusion pour le projet : l'albédo mensuel de surface est une entrée valable
pour `R_ns = (1 - alpha) * R_s`, mais il ne suffit pas à lui seul à représenter
la transmission atmosphérique.

## Recommandation shortwave

### Choix par défaut pour le modèle 4

Garder la géométrie solaire du projet, puis corriger par une transmissivité
mensuelle dérivée d'ERA5 :

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

- formule physique standard du bilan shortwave net ;
- conserve le calcul solaire historique du projet ;
- garde le cycle jour/nuit et la saisonnalité venant de `S0*cos(i)` ;
- utilise ERA5 seulement pour représenter une transmission atmosphérique
  moyenne mensuelle ;
- conserve l'albédo de surface explicite du projet ;
- évite d'appliquer directement un effet nuageux TOA à la surface ;
- force la moyenne mensuelle de `SW_down_surface` à rester proche d'ERA5, sans
  remplacer tout le signal temporel par ERA5.

### Mode encore plus simple

Pour une première version très robuste, mais moins pédagogique :

```text
SW_absorbe_surface =
    era5_sw_down_surface_w_m2 * (1 - albedo_surface)
```

Pour un mode de validation direct :

```text
SW_absorbe_surface = colonne.validation_flux["era5_sw_net_surface_w_m2"]
```

Ces deux modes sont utiles pour vérifier les ordres de grandeur, mais ils
affaiblissent l'enjeu du projet si on les utilise comme seul flux shortwave
dans la boucle temporelle.

### Mode à éviter par défaut

Éviter :

```text
SW_absorbe_surface =
    SW_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Cette formule peut rester comme mode pédagogique "TOA simplifié", mais elle ne
doit pas piloter la température du modèle 4.

## Implémentation simple conseillée

Le modèle 3 doit fournir une fonction auxiliaire shortwave propre et le modèle 4 doit
l'utiliser dans sa boucle. Principe :

1. Pré-calculer une moyenne mensuelle de `S0 * max(cos(i), 0)` sur la grille.
2. Construire `tau_SW_mensuel` depuis `era5_sw_down_surface_w_m2`.
3. À chaque pas du modèle 4, recalculer `S0 * max(cos(i(t)), 0)`.
4. Appliquer `tau_SW_mensuel` puis l'albédo de surface.
5. Appeler `calculer_colonne_radiative` pour obtenir les flux longwave :
   `LW_down_absorbe_surface`, `LW_up_surface`, diagnostics CO2/H2O.

```python
def shortwave_surface_modele4(colonne, jour_annee, heure_solaire):
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

Le bilan du modèle 4 devient alors :

```text
C_surface dT/dt =
    SW_absorbe_surface_corrige
  + LW_down_absorbe_surface_modele3
  - LW_up_surface_modele3
  - autres termes de surface du modèle 0
```

## Optimisation du temps par colonne

Le coût principal du modèle 3 vient de `flux_corps_noir_dans_bande`, qui
intègre numériquement Planck avec 2000 pas pour chaque bande et beaucoup de
températures. Ce calcul est répété alors que les températures de couches
mensuelles changent peu ou pas pendant la boucle du modèle 4.

Des mesures locales indicatives ont montré que ce cache réduit les recalculs
de Planck quand les mêmes températures et bandes reviennent souvent. Ces
mesures ne sont pas versionnées comme benchmark; elles ne doivent donc pas être
citées comme garantie de performance.

Optimisation recommandée, simple et lisible :

1. Ajouter un cache `functools.lru_cache` autour de l'intégrale de Planck par
   bande et température.
2. Commencer par un cache exact pour ne pas changer les résultats.
3. Si besoin, arrondir la température à 0.1 K ou 0.05 K pour augmenter les
   répétitions, en documentant l'erreur numérique.
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

Ce n'est pas de l'informatique avancée : c'est seulement éviter de refaire une
intégrale numérique identique des milliers de fois.

## Décision finale

Pour le modèle 4, le meilleur compromis rigueur/simplicité est :

```text
flux shortwave par défaut =
    S0*cos(i,t) * transmissivite_SW_mensuelle_ERA5 * (1 - albedo_surface)

mode entree imposee simple =
    ERA5 SW_down_surface * (1 - albedo_surface)

mode validation =
    ERA5 SW_net_surface direct

mode ancien TOA =
    diagnostic seulement
```

Cette décision est mieux sourcée que l'ancien coefficient nuageux, s'appuie sur
les flux de surface déjà présents dans le paquet, garde le cœur solaire du
projet, et reste facile à expliquer à l'oral. Son contrôle reproductible est
celui des tests physiques simples du modèle 3, pas une précision climatique
globale.
