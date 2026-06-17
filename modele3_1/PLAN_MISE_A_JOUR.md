# Plan de mise a jour modele 3.1 - court-onde et optimisation

Objectif : preparer le modele 3.1 pour le modele 4 sans casser l'enjeu du
projet. On garde le calcul solaire historique `S0 * max(cos(i), 0)`, mais on
ajoute une transmissivite atmospherique mensuelle derivee d'ERA5 pour eviter de
surestimer fortement le court-onde de surface.

Ce plan ne remplace pas le plan principal 3.1. Il decrit la prochaine mise a
jour ciblee :

1. nouveau court-onde surface utilisable par le modele 4 ;
2. optimisation rapide et lisible du calcul long-onde par cache Planck.

## 1. Origine de S0 dans le projet

Dans le modele 0 :

```text
modele0_maintenance/codes_python/physique/solaire.py
constante_solaire = 1361.0  # W m-2, irradiance au sommet de l'atmosphere
```

Dans le modele 3 et le modele 3.1 :

```text
CONSTANTE_SOLAIRE = 1361.0  # W m-2
```

Cette valeur est l'irradiance solaire totale au sommet de l'atmosphere, sur une
surface perpendiculaire aux rayons solaires, a environ `1 au`. Elle est
coherente avec la valeur nominale adoptee par l'IAU en 2015 pour l'irradiance
solaire totale nominale :

```text
https://arxiv.org/abs/1605.09788
```

Donc `S0` n'etait pas une constante inventee localement : c'est une constante
solaire standard, deja presente dans le modele 0 et reprise ensuite dans les
modeles 3 et 3.1.

## 2. Probleme a corriger

Le modele 3.1 calcule actuellement :

```text
SW_TOA_local = S0 * max(cos(i), 0)
SW_absorbe_surface =
    SW_TOA_local
  * (1 - albedo_nuages_effectif)
  * (1 - albedo_surface)
```

Ce calcul garde bien la geometrie solaire locale, mais il ne represente pas le
transfert court-onde atmospherique complet. Il applique un effet nuageux CERES
au sommet de l'atmosphere comme si c'etait une transmission surface, et ignore
la diffusion Rayleigh, les aerosols, l'absorption court-onde par vapeur d'eau,
ozone, et autres effets clairs.

Le modele 0 avait la meme limite de fond :

```text
P_abs_atm_solar(...) = 0
```

Donc il ne faut pas supprimer `S0*cos(i)`, mais il faut corriger le passage
entre sommet de l'atmosphere et surface.

## 3. Choix retenu

Ne pas utiliser directement ERA5 comme unique court-onde a chaque pas. Utiliser
ERA5 pour construire une transmissivite atmospherique mensuelle :

```text
SW_TOA_moyen_mensuel =
    moyenne_mensuelle(S0 * max(cos(i), 0))

transmissivite_sw_mensuelle =
    era5_sw_down_surface_w_m2 / SW_TOA_moyen_mensuel
```

Puis, dans la boucle temporelle du modele 4 :

```text
SW_TOA_local(t) =
    S0 * max(cos(i(t)), 0)

SW_down_surface(t) =
    transmissivite_sw_mensuelle * SW_TOA_local(t)

SW_absorbe_surface_corrige(t) =
    SW_down_surface(t) * (1 - albedo_surface)
```

Interet :

- on garde le calcul solaire du projet ;
- on garde le cycle jour/nuit ;
- on garde la saisonnalite et la dependance latitude/heure ;
- ERA5 sert seulement a representer l'effet atmospherique mensuel moyen ;
- la correction est sourcee et mesurable, pas sortie du chapeau.

## 4. Changements de donnees a faire

Dans `modele3_1/generer_donnees.py`, ajouter pendant la generation du paquet :

```text
sw_toa_moyen_mensuel_w_m2[mois, lat]
transmissivite_sw_mensuelle[mois, lat, lon]
```

`sw_toa_moyen_mensuel_w_m2` peut dependre seulement de la latitude si on moyenne
sur des jours solaires complets. Le stocker en `[mois, lat]` suffit et reste
tres compact.

Calcul conseille :

```text
pour chaque mois:
  pour chaque latitude:
    moyenner S0 * max(cos(i), 0)
    sur tous les jours du mois
    et sur 48 ou 96 pas horaires solaires
```

Puis :

```text
si sw_toa_moyen_mensuel_w_m2 > epsilon:
    transmissivite = era5_sw_down_surface_w_m2 / sw_toa_moyen_mensuel_w_m2
sinon:
    transmissivite = 0

transmissivite = borne(transmissivite, 0, 1)
```

Bornage :

- minimum `0.0` ;
- maximum `1.0` ;
- si une valeur est bornee, garder un diagnostic global dans `metadata.json`.

Quantification :

```text
sw_toa_moyen_mensuel_w_m2     uint16, scale 0.1,  offset 0.0
transmissivite_sw_mensuelle   uint16, scale 1e-4, offset 0.0
```

Metadata :

```text
source sw_toa_moyen_mensuel = geometrie solaire modele 3.1, S0=1361 W/m2
source transmissivite_sw = ERA5 sw_down_surface / SW_TOA_moyen_mensuel
```

## 5. Changements de code physique

Dans `modele3_1/physique.py`, garder les fonctions existantes :

```text
flux_solaire_incident(latitude_deg, jour_annee, heure_solaire)
flux_solaire_moyen_journalier(latitude_deg, jour_annee)
```

Ajouter une fonction explicite :

```python
def flux_sw_surface_transmis(
    latitude_deg,
    jour_annee,
    heure_solaire,
    transmissivite_sw,
    albedo_surface,
):
    sw_toa = flux_solaire_incident(latitude_deg, jour_annee, heure_solaire)
    sw_down = sw_toa * fraction(transmissivite_sw, defaut=0.0)
    return sw_down * (1.0 - fraction(albedo_surface, defaut=0.30))
```

Option utile pour le modele 4 :

```python
def flux_sw_surface_transmis_moyenne_journaliere(...):
    # moyenne sur 96 heures solaires, comme l'ancien helper
```

Ne pas supprimer tout de suite l'ancien `flux_sw_absorbe_surface`. Le garder
comme mode diagnostic `toa_nuages_ceres`, mais ne plus le recommander pour le
bilan thermique du modele 4.

## 6. Changements de chargement colonne

Dans `modele3_1/donnees.py`, `extraire_colonne` doit ajouter dans `surface` :

```text
sw_toa_moyen_mensuel_w_m2
transmissivite_sw_mensuelle
source_transmissivite_sw_mensuelle
```

Si on extrait avec `jour_annee`, interpoler cycliquement la transmissivite entre
les mois comme les autres champs mensuels.

## 7. Changements dans `calculer_colonne_radiative`

Ajouter un parametre optionnel :

```python
mode_court_onde="transmissivite_sw"
```

Modes :

```text
transmissivite_sw  # nouveau mode recommande
era5_down_albedo   # forcage mensuel simple, utile pour validation
era5_net           # validation directe
toa_nuages_ceres   # ancien mode 3.1, diagnostic seulement
```

Pour garder la compatibilite, on peut commencer avec :

```text
mode_court_onde="toa_nuages_ceres"
```

puis basculer le modele 4 explicitement sur :

```text
mode_court_onde="transmissivite_sw"
```

Mais pour eviter les erreurs futures, le README doit dire clairement que le
mode recommande pour integrer `T_surface(t)` est `transmissivite_sw`.

## 8. Tests a ajouter

Dans `modele3_1/tests/tester_modele3_1.py` :

1. `transmissivite_sw_mensuelle` existe dans le paquet et est bornee dans
   `[0, 1]`.
2. En moyenne mensuelle, le nouveau mode reconstruit bien le flux ERA5 :

```text
moyenne_mensuelle(SW_down_surface_modele)
approx era5_sw_down_surface_w_m2
```

3. Pour Paris juillet :

```text
SW_absorbe_surface_transmis proche de
era5_sw_down_surface_w_m2 * (1 - albedo_surface)
```

4. Le mode `toa_nuages_ceres` reste disponible comme diagnostic.
5. Le mode `era5_net` renvoie exactement `era5_sw_net_surface_w_m2`.

## 9. Optimisation rapide du long-onde

Probleme actuel : `flux_corps_noir_dans_bande` integre Planck avec 2000 pas a
chaque appel. Dans une grille, les memes temperatures de couches mensuelles et
les memes bandes sont recalculees beaucoup de fois.

Optimisation simple :

1. Renommer la fonction actuelle en :

```python
def flux_corps_noir_dans_bande_direct(...):
```

2. Ajouter un wrapper cache :

```python
from functools import lru_cache

@lru_cache(maxsize=50000)
def _flux_corps_noir_dans_bande_cache(temperature_k, lambda_min_um, lambda_max_um):
    return flux_corps_noir_dans_bande_direct(
        temperature_k,
        lambda_min_um,
        lambda_max_um,
    )

def flux_corps_noir_dans_bande(temperature_k, lambda_min_um, lambda_max_um, nombre_pas=2000):
    return _flux_corps_noir_dans_bande_cache(
        round(float(temperature_k), 3),
        float(lambda_min_um),
        float(lambda_max_um),
    )
```

Commencer avec `round(..., 3)` pour ne changer presque aucun resultat. Si le
gain est insuffisant, tester `round(..., 1)` et documenter l'erreur maximale.

Mesure locale deja observee :

```text
100 colonnes sans cache : environ 6.56 s
100 colonnes avec cache : environ 0.87 s au premier passage
gain observe            : x7.6
```

Ce n'est pas une optimisation complexe : c'est juste eviter de refaire la meme
integrale de Planck des milliers de fois.

## 10. Ordre d'implementation recommande

1. Ajouter les fonctions de moyenne mensuelle TOA solaire dans
   `generer_donnees.py`.
2. Ajouter `transmissivite_sw_mensuelle` au paquet et a `metadata.json`.
3. Ajouter le chargement de la transmissivite dans `extraire_colonne`.
4. Ajouter le helper physique `flux_sw_surface_transmis`.
5. Ajouter `mode_court_onde` dans `calculer_colonne_radiative`.
6. Ajouter les tests court-onde.
7. Ajouter le cache Planck.
8. Refaire les mesures de temps par colonne.
9. Mettre a jour `README.md`, `THEORIE.md` et `PROVENANCE_DONNEES.md`.
10. Regenerer le paquet compact et verifier sa taille.

## 11. Critere d'acceptation

La mise a jour est prete quand :

- le paquet contient la transmissivite SW mensuelle ;
- Paris juillet ne donne plus un court-onde absorbe grossierement trop eleve en
  mode recommande ;
- le mode ancien reste disponible comme diagnostic ;
- les tests 3.1 passent ;
- le temps par colonne baisse fortement avec le cache Planck ;
- la documentation dit clairement que le modele garde `S0*cos(i)` et utilise
  ERA5 seulement pour corriger la transmission atmospherique moyenne.
