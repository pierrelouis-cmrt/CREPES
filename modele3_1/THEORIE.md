# Theorie du modele 3.1

Le modele 3.1 est une colonne radiative locale :

```text
(colonne, T_surface, CO2_ppm) -> flux radiatifs
```

La temperature de surface est imposee. Le modele 4 utilisera ces flux pour
faire evoluer `T_surface(t)`.

## Court-onde

La geometrie solaire reste celle du projet :

```text
cos(i) =
    sin(latitude) * sin(declinaison)
  + cos(latitude) * cos(declinaison) * cos(angle_horaire)

SW_TOA_local = 1361 * max(cos(i), 0)
```

Le transfert atmosphere-surface est represente par une transmissivite mensuelle
issue d'ERA5 :

```text
transmissivite_sw_mensuelle =
    ERA5_SW_down_surface / moyenne_mensuelle(SW_TOA_local)
```

Le flux absorbe par la surface est ensuite :

```text
SW_down_surface = transmissivite_sw_mensuelle * SW_TOA_local
SW_absorbe_surface = SW_down_surface * (1 - albedo_surface)
```

Il n'y a plus de mode court-onde alternatif dans 3.1.

## Long-onde

La surface emet :

```text
LW_up_surface = epsilon_surface * sigma * T_surface^4
epsilon_surface = 0.98
```

Pour chaque couche et chaque bande infrarouge :

```text
tau_CO2 = a_CO2_bande * (CO2_ppm / 280) * (delta_p / 101325)
tau_H2O = a_H2O_bande * (masse_H2O / 10 kg m-2)
tau_total = tau_CO2 + tau_H2O
transmission = exp(-1.66 * tau_total)
emissivite_couche = 1 - transmission
```

Les coefficients de bandes sont effectifs. Ils gardent un noyau CO2 + H2O
simple et lisible ; ils ne remplacent pas HITRAN, RADIS ou une methode
correlated-k.

## Vapeur d'eau

Le generateur moyenne l'humidite specifique ERA5 `q` dans chaque couche :

```text
masse_air = delta_p / g
masse_H2O = q_moyen * masse_air
```

La reference `MASSE_H2O_REFERENCE_KG_M2 = 10.0` fixe l'echelle des coefficients
H2O.

## Nuages

Les nuages ne sont pas un terme radiatif explicite dans 3.1. Leur effet moyen
sur le court-onde de surface est inclus dans la transmissivite ERA5 mensuelle :

```text
tau_total = tau_CO2 + tau_H2O
```

## Validation

Le paquet conserve des flux ERA5 mensuels pour comparer les ordres de grandeur.
Pour Paris, point de grille `47.5 N, 2.5 E`, juillet, `T_surface = 293 K`,
moyenne journaliere SW :

```text
SW_absorbe_surface   = 190.45 W m-2
ERA5 SW_net_surface  = 188.80 W m-2
LW_down_surface      = 349.91 W m-2
ERA5 LW_down_surface = 364.20 W m-2
OLR modele           = 286.15 W m-2
ERA5 OLR             = 252.90 W m-2
```

Le court-onde est volontairement cale sur une transmissivite mensuelle moyenne.
Le long-onde reste un modele effectif CO2 + H2O.
