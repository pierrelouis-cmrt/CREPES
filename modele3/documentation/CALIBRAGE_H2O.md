# Calibrage H2O du modele 3

Le modele 3 represente la vapeur d'eau avec quelques coefficients de bande
effectifs, pas avec un transfert radiatif ligne par ligne. Le script
`modele3/codes_python/calibrer_coefficients_h2o.py` fabrique ces coefficients
a partir de transmissions HITRAN/RADIS.

Les coefficients H2O utilises par le calcul normal ne sont pas ecrits dans
`physique.py`. Ils sont lus depuis :

```text
modele3/ressources/coefficients_opacite_modele3.npz
```

La methode volontairement simple est :

1. prendre des couches atmospheriques deja preparees dans le paquet modele 3 ;
2. convertir la masse de vapeur d'eau de chaque couche en fraction molaire
   homogene pour RADIS ;
3. calculer une transmission H2O HITRAN/RADIS pour chaque couche humide et
   chaque bande qui porte un coefficient `a_h2o` ;
4. moyenner la transmission sur la bande avec un poids de Planck ;
5. convertir cette transmission moyenne en profondeur optique equivalente ;
6. retenir la mediane de `tau_eq / X` pour chaque bande, avec
   `X = masse_h2o_kg_m2 / 10`.

## Formule du modele

Dans `modele3/codes_python/physique.py`, la profondeur optique H2O d'une couche
`k` et d'une bande `b` vaut :

$$
\tau_{\mathrm{H2O},k,b}
= a_{\mathrm{H2O},b}
\frac{M_{\mathrm{H2O},k}}{10\ \mathrm{kg\,m^{-2}}}
$$

La transmission utilise ensuite :

$$
T_{k,b} = \exp[-1.66(\tau_{\mathrm{CO2},k,b}+\tau_{\mathrm{H2O},k,b})]
$$

Le facteur `1.66` reste l'approximation diffusif/two-stream du modele.

## Commandes

Installer les dependances optionnelles :

```bash
./.venv/bin/python -m pip install -r modele3/requirements-calibrage.txt
```

Voir le volume de calcul sans appeler RADIS :

```bash
./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_h2o --dry-run
```

Lancer un calibrage un peu plus representatif que le defaut rapide :

```bash
./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_h2o \
  --latitudes=-60,-30,0,30,60 \
  --longitudes=0 \
  --mois=1,4,7,10
```

L'option `--h2o-scale-values` existe pour explorer la sensibilite a des masses
H2O multipliees, mais le defaut `1` est le choix le plus propre physiquement :
il utilise directement les profils humides du paquet.

## Sortie

Le script met a jour le paquet commun :

```text
modele3/ressources/coefficients_opacite_modele3.npz
```

Le script calibre toutes les bandes ou `a_h2o > 0`, y compris les intervalles
CO2 ou la vapeur d'eau contribue aussi a l'opacite.

## Limites

- Les coefficients restent des coefficients effectifs par grandes bandes.
- Le modele ne devient pas un modele ligne par ligne.
- Les couches RADIS sont supposees homogenes.
- La dependance fine en temperature, pression et continuum H2O n'est pas
  remplacee par un vrai schema correlated-k.
- Les flux ERA5 servent a la validation du modele, pas au calibrage H2O.
