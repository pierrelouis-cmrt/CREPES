# Consignes modèle 3 CREPES

## Objectif

Créer un `modele3` qui simule **une seule colonne atmosphère-surface** pour une position `(latitude, longitude)` donnée.

Cette colonne doit représenter une future cellule du maillage terrestre. Le modèle complet en grille viendra plus tard et pourra appeler ce modèle colonne en boucle.

Pour le premier cas de test, utiliser un point arbitraire :

```text
Paris : lat = 48.8566, lon = 2.3522
```

Le modèle 3 doit prédire l’évolution de la température de surface :

```text
T_surface(t)
```

sur les jours de l’année, en utilisant :

- le cycle solaire local ;
- les données atmosphériques mensuelles disponibles ;
- un bilan énergétique de surface simple ;
- le noyau radiatif du modèle 2.5 adapté à une colonne locale.

## Philosophie

Rester simple et cohérent. Ne pas essayer de faire un vrai modèle climatique couplé.

Le modèle 3 est un modèle de surface forcé par une atmosphère prescrite :

- la surface évolue ;
- l’atmosphère ne réagit pas encore à la surface ;
- les profils atmosphériques viennent des données disponibles ;
- ERA5 `skin temperature` peut servir à initialiser ou valider, mais ne doit pas être imposée comme sortie.

## Données disponibles à exploiter

Le modèle doit pouvoir exploiter, quand les fichiers sont présents :

### Données atmosphériques verticales

Variables disponibles par mois, latitude, longitude et niveau de pression :

- température atmosphérique `T(p)` ;
- humidité spécifique `q(p)` ;
- fraction nuageuse verticale éventuelle ;
- géopotentiel éventuel.

Usage :

- remplacer l’atmosphère standard 1976 du modèle 2.5 ;
- construire les couches locales de la colonne ;
- calculer l’opacité H2O simple.

### Données de surface et validation

Variables disponibles par mois, latitude, longitude :

- pression de surface ;
- température à 2 m ;
- température de peau/skin temperature ;
- température de surface mer ;
- couverture nuageuse totale, basse, moyenne, haute ;
- masque terre-mer ;
- albédo ;
- neige/glace ;
- flux radiatifs ERA5 pour validation.

Usage :

- pression de surface : déterminer la masse de colonne ;
- skin temperature : initialisation ou comparaison seulement ;
- cloud cover : approximation nuageuse simple ;
- albédo : solaire absorbé ;
- flux ERA5 : validation.

### Données MODIS émissivité

Variables utiles :

- `Emis_31`
- `Emis_32`

Usage simple :

```text
epsilon_surface_land = mean(Emis_31, Emis_32)
epsilon_ocean = 0.985
epsilon_snow_ice ≈ 0.98
```

Si MODIS n’est pas encore disponible, utiliser une constante :

```text
epsilon_surface = 0.98
```

## Bilan de surface

Le modèle doit intégrer :

```text
C_surface dT_surface/dt =
    SW_absorbé_surface
  + LW_descendant_surface
  - LW_émis_surface
  - flux_latent
  - flux_sensible
```

Pour commencer :

- `C_surface` peut reprendre la capacité thermique du modèle 0 si disponible ;
- sinon utiliser une valeur constante raisonnable ;
- `flux_latent` peut reprendre le modèle 0 ou une approximation simple ;
- `flux_sensible` peut reprendre la convection du modèle 0 avec `T_air` local approximé par température 2 m ou première couche atmosphérique.

## Cycle solaire

Réutiliser la logique du modèle 0 :

- jour de l’année ;
- heure solaire locale ;
- latitude ;
- longitude ;
- cosinus d’incidence solaire.

Flux solaire incident :

```text
SW_in = S0 * max(cos_incidence, 0)
```

Flux absorbé :

```text
SW_absorbé = SW_in * (1 - albedo_surface) * (1 - albedo_cloud)
```

Rester simple : ne pas encore modéliser la diffusion atmosphérique détaillée.

## Colonne verticale

La colonne doit être construite à partir de la pression de surface locale.

Utiliser une grille proche du modèle 2.5 :

```text
p_edges_ref = [850, 700, 500, 300, 200, 100, 50, 20, 10, 1] hPa
```

Puis :

```text
p_edges = [p_surface_hPa] + tous les niveaux ref strictement inférieurs à p_surface_hPa
```

Exemple à basse altitude :

```text
[1010, 850, 700, 500, ..., 1]
```

Exemple en montagne :

```text
[750, 700, 500, ..., 1]
```

Chaque couche a :

```text
p_bas
p_haut
delta_p = p_bas - p_haut
T_moyen
q_moyen
CO2_moyen
cloud_fraction éventuelle
```

Les moyennes `T` et `q` doivent être interpolées/moyennées depuis les données pression disponibles.

## CO2

Pour l’instant, garder simple :

```text
CO2_ppm = 420
```

ou une valeur configurable.

Même valeur dans toutes les couches.

## Vapeur d’eau

Utiliser l’humidité spécifique `q`.

Pour chaque couche :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

Ajouter une opacité H2O effective simple.

Forme attendue :

```text
tau_total_bande = tau_CO2_bande + tau_H2O_bande
transmission = exp(-D * tau_total_bande)
emissivite = 1 - transmission
```

Important :

- ne pas additionner les effets radiatifs CO2 et H2O après calcul ;
- additionner les opacités avant l’exponentielle.

## Coefficients optiques

Ne pas faire RADIS pour l’instant.

Garder la logique du modèle 2.5 :

```text
tau_CO2 = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
```

Ajouter une version simple pour H2O :

```text
tau_H2O = a_H2O_bande * facteur_humidite
```

Les coefficients H2O peuvent être calibrés grossièrement plus tard. Pour l’instant l’objectif est architecture + ordre de grandeur.

## Nuages

Rester minimal.

Utiliser :

- couverture nuageuse basse ;
- moyenne ;
- haute ;
- ou couverture totale si les trois niveaux ne sont pas disponibles.

Placement simple :

```text
low cloud    -> couche basse
medium cloud -> couche moyenne
high cloud   -> couche haute
```

Effet court-onde :

```text
albedo_cloud = coefficient_simple * cloud_fraction
```

Effet long-onde :

- traiter le nuage comme une couche IR supplémentaire ou comme une augmentation d’émissivité dans la couche correspondante ;
- commencer avec une approximation simple ;
- ne pas chercher une microphysique détaillée.

## Surface

Flux thermique émis :

```text
LW_émis_surface = epsilon_surface * sigma * T_surface^4
```

Flux descendant absorbé :

```text
LW_absorbé_surface = epsilon_surface * LW_down_surface
```

Si émissivité indisponible :

```text
epsilon_surface = 0.98
```

## Temporalité

Le modèle doit fonctionner avec des données mensuelles.

Pour simuler jour par jour :

- interpoler les données mensuelles vers le jour courant ;
- ou utiliser la valeur du mois courant pour une première version.

Pas besoin d’une météo journalière réelle maintenant.

Le pas de temps peut reprendre le modèle 0 :

```text
dt = 1800 s
```

## Sorties attendues

Pour un point `(lat, lon)`, le modèle doit produire au minimum :

- série temporelle `T_surface_K` ;
- `SW_absorbé_surface` ;
- `LW_down_surface` ;
- `LW_up_surface` ;
- flux latent ;
- flux sensible ;
- éventuellement bilan net.

Il doit afficher ou sauvegarder un résumé clair :

```text
lat, lon
date/jour
T_surface finale
T_surface min/max
flux moyens
```

Un graphique simple est utile :

- température de surface en °C ;
- flux principaux.

## Validation minimale

Pour Paris, comparer qualitativement :

- la température simulée à ERA5 `skin temperature` ou `2m temperature` ;
- le LW descendant simulé à ERA5 `surface downward long-wave radiation`;
- le SW net simulé à ERA5 `surface net short-wave radiation`.

La validation n’a pas besoin d’être parfaite. L’objectif est que les ordres de grandeur soient cohérents et que le modèle réagisse correctement aux saisons, à l’humidité, aux nuages et à la pression locale.

## Hors périmètre pour la première version

Ne pas faire maintenant :

- modèle global multi-cellules ;
- échanges horizontaux entre colonnes ;
- dynamique atmosphérique ;
- rétroaction de `T_surface` sur `T_atm(p)` ;
- RADIS/HITRAN ;
- correlated-k ;
- ozone, CH4, N2O ;
- microphysique détaillée des nuages.

## Priorité d’implémentation

Ordre recommandé :

1. créer une colonne locale `(lat, lon)` ;
2. charger/interpoler les données mensuelles disponibles ;
3. construire les couches pression locales ;
4. remplacer `T_atm` standard par `T(p)` local ;
5. intégrer `T_surface(t)` avec bilan énergétique ;
6. ajouter H2O simple ;
7. ajouter pression de surface ;
8. ajouter émissivité de surface ;
9. ajouter nuages simples ;
10. comparer aux flux ERA5.

Le modèle 3 doit rester un module colonne propre, réutilisable plus tard par une grille terrestre.

```

```
