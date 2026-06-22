# Calibrage CO2 du modele 3

Le modele 3 utilise des coefficients de bande simples, pas un transfert
radiatif ligne par ligne. Le script
`modele3/codes_python/calibrer_coefficients_co2.py` sert
uniquement a fabriquer ces coefficients de facon tracable depuis HITRAN/RADIS.
Il ne fait pas partie du calcul normal du modele.

La methode gardee est volontairement courte :

1. calculer une transmission HITRAN/RADIS pour chaque couche, bande CO2 et
   concentration de CO2 choisie ;
2. moyenner cette transmission sur la bande avec un poids de Planck ;
3. convertir la transmission moyenne en profondeur optique equivalente ;
4. prendre la mediane de `tau_eq / X` pour chaque bande ;
5. appliquer un facteur global pour retrouver le forcage cible `280 -> 560 ppm`.

Tout le reste a ete retire du script : diagnostics coeur/ailes, RMSE,
quantiles, liste de sources dans le JSON, sauvegarde de toutes les mesures
intermediaires et options secondaires.

## Formule du modele

Dans `modele3/codes_python/physique.py`, la profondeur optique CO2 d'une couche
`k` et d'une bande `b` vaut :

$$
\tau_{\mathrm{CO2},k,b}
= a_{\mathrm{CO2},b}
\frac{C}{280}
\frac{\Delta p_k}{101325}
$$

avec :

- `C` : concentration de CO2 en ppm ;
- `delta_p_k` : epaisseur de pression de la couche, en Pa ;
- `a_CO2,b` : coefficient effectif de bande.

La transmission utilise ensuite :

$$
T_{k,b} = \exp[-1.66(\tau_{\mathrm{CO2},k,b}+\tau_{\mathrm{H2O},k,b})]
$$

Le facteur `1.66` est l'approximation diffusif/two-stream deja utilisee par le
modele. Ce n'est pas une donnee HITRAN.

## Compression d'un spectre RADIS

Pour une couche homogene, RADIS fournit une transmission spectrale
`T_ref(nu)`. Comme le modele ne garde qu'une transmission moyenne par bande, on
la pondere par le flux thermique de la couche :

$$
\overline{T}_{\mathrm{ref}}
=
\frac{\int_b B_\nu(T_k)T_{\mathrm{ref}}(\nu)d\nu}
{\int_b B_\nu(T_k)d\nu}
$$

Cette moyenne est l'etape physique importante : elle donne plus de poids aux
frequences qui contribuent vraiment au flux infrarouge de la bande.

On convertit ensuite la transmission moyenne en profondeur optique compatible
avec le modele :

$$
\tau_{\mathrm{eq}} = -\frac{\ln(\overline{T}_{\mathrm{ref}})}{1.66}
$$

Pour chaque mesure, le facteur connu du modele est :

$$
X =
\frac{C}{280}
\frac{\Delta p}{101325}
$$

Chaque mesure propose donc un coefficient :

$$
a_i = \frac{\tau_{\mathrm{eq},i}}{X_i}
$$

Le coefficient de bande retenu est :

$$
a_{\mathrm{CO2},b} = \mathrm{mediane}(a_i)
$$

La mediane est gardee parce qu'elle resume le coefficient typique de bande sans
laisser une couche atypique dominer le resultat. C'est plus coherent avec le
niveau de simplification du modele qu'un ajustement fin par moindres carres.

## Recalage du forcage

Apres le calibrage par bande, le script cherche un facteur global `s` tel que :

$$
\mathrm{OLR}_{280} - \mathrm{OLR}_{560}
\approx 5.35\ln(2)
$$

La cible par defaut vaut donc environ `3.71 W m-2`. L'option
`--cible-forcage` permet d'utiliser une autre valeur, par exemple `3.93`.

## Commandes

Installer les dependances optionnelles :

```bash
./.venv/bin/python -m pip install -r modele3/requirements-calibrage.txt
```

Voir le volume de calcul sans appeler RADIS :

```bash
./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_co2 --dry-run
```

Lancer le calibrage :

```bash
./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_co2 \
  --latitudes=-60,-30,0,30,60 \
  --longitudes=0 \
  --mois=1,4,7,10 \
  --co2-values=280,420,560,1120
```

## Sorties

Le script ecrit deux fichiers :

```text
modele3/ressources/calibrage_opacite_co2/calibrage_coefficients_co2.json
modele3/ressources/calibrage_opacite_co2/coefficients_co2_calibres.py
```

Le JSON contient seulement la methode, l'echantillon, le forcage et les
coefficients. Le fichier Python permet de tester les coefficients avec
`calculer_colonne_radiative(..., bandes=...)` sans modifier
`codes_python/physique.py`.

## Limites

- Les coefficients restent des coefficients effectifs de bande.
- Le modele ne devient pas un modele ligne par ligne.
- Les couches RADIS sont supposees homogenes.
- Les coefficients H2O ne sont pas calibres ici.
- Le facteur global cale un ordre de grandeur de forcage, pas une perfection
  spectrale bande par bande.
