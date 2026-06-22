# Rappel LaTeX - calibrage CO2/H2O du modele 3

Cette note fixe la methode exacte pour passer de spectres HITRAN/RADIS a des
coefficients d'opacite utilisables par le modele 3. Le point central est que le
modele 3 ne fait pas de transfert ligne par ligne : il compresse chaque bande
spectrale en un seul coefficient effectif.

## 1. Ce que le modele utilise en production

Pour une couche verticale \(k\) et une bande infrarouge \(b\), le calcul normal
du modele 3 lit les coefficients dans :

```text
modele3/ressources/coefficients_opacite_modele3.npz
```

Il applique ensuite :

$$
\tau_{\mathrm{CO2},k,b}
= a_{\mathrm{CO2},b}
\frac{C}{C_0}
\frac{\Delta p_k}{p_0},
\qquad
C_0 = 280\ \mathrm{ppm},
\qquad
p_0 = 101325\ \mathrm{Pa}.
$$

$$
\tau_{\mathrm{H2O},k,b}
= a_{\mathrm{H2O},b}
\frac{M_{\mathrm{H2O},k}}{M_0},
\qquad
M_0 = 10\ \mathrm{kg\,m^{-2}}.
$$

Les opacites sont additionnees avant la transmission :

$$
\tau_{k,b}
= \tau_{\mathrm{CO2},k,b}
+ \tau_{\mathrm{H2O},k,b}
+ \tau_{\mathrm{nuage},k},
\qquad
\mathcal T_{k,b}=\exp(-D\,\tau_{k,b}),
\qquad
D = 1.66.
$$

Donc le calibrage doit produire \(a_{\mathrm{CO2},b}\) et
\(a_{\mathrm{H2O},b}\), pas des sections efficaces spectrales. Ces coefficients
sont des profondeurs optiques effectives normalisees.

## 2. Principe general du calibrage

Le calibrage suit la meme chaine pour CO2 et H2O :

1. choisir un echantillon de colonnes du paquet modele 3 ;
2. transformer chaque couche en dalle homogene compatible avec RADIS ;
3. calculer une transmission spectrale HITRAN/RADIS par gaz, couche et bande ;
4. moyenner cette transmission dans la bande avec un poids de Planck ;
5. convertir la transmission moyenne en profondeur optique equivalente ;
6. diviser par le facteur de normalisation \(X\) que le modele utilisera ;
7. retenir la mediane des rapports \(\tau_{\mathrm{eq}}/X\) par bande.

Sous forme compacte :

$$
a_b
= \operatorname{mediane}_i
\left(
\frac{\tau_{\mathrm{eq},i}}{X_i}
\right).
$$

La mediane est volontaire : elle donne la pente typique de la bande, limite
l'influence des couches atypiques et reste coherente avec un modele a un seul
coefficient par bande. Un ajustement aux moindres carres donnerait une
precision apparente plus forte, mais trop dependante des cas extremes et de la
saturation des coeurs de raies.

## 3. Echantillon de couches

Les scripts `calibrer_coefficients_co2.py` et
`calibrer_coefficients_h2o.py` chargent des colonnes avec
`charger_colonnes_calibration`. Par defaut :

```text
latitudes  = -45,0,45
longitudes = 0
mois       = 1,7
```

Chaque colonne vient du paquet compact du modele 3. Une couche fournit au
minimum :

$$
p_{\mathrm{bas}},\quad p_{\mathrm{haut}},\quad T_k,\quad
M_{\mathrm{air},k},\quad M_{\mathrm{H2O},k}.
$$

On pose :

$$
\Delta p_k = p_{\mathrm{bas}}-p_{\mathrm{haut}},
\qquad
p_{\mathrm{milieu}}=\frac{p_{\mathrm{bas}}+p_{\mathrm{haut}}}{2}.
$$

Si la masse d'air n'est pas fournie, elle est reconstruite par :

$$
M_{\mathrm{air},k}=\frac{\Delta p_k}{g}.
$$

Pour RADIS, la couche est remplacee par une dalle homogene. Sa densite moyenne
est estimee par le gaz parfait :

$$
\rho_{\mathrm{air},k}
= \frac{p_{\mathrm{milieu}}M_{\mathrm{air,mol}}}{R T_k}.
$$

La longueur optique transmise a RADIS vaut donc :

$$
L_k
= \frac{M_{\mathrm{air},k}}{\rho_{\mathrm{air},k}},
$$

puis elle est convertie en centimetres. Cette etape est une approximation de
couche homogene : pression, temperature et fraction molaire sont constantes
dans la dalle RADIS.

## 4. Appel HITRAN/RADIS

Pour chaque bande du modele :

$$
\lambda_{\min},\lambda_{\max}
\quad\Longrightarrow\quad
\tilde\nu_{\min}=\frac{10000}{\lambda_{\max}},
\qquad
\tilde\nu_{\max}=\frac{10000}{\lambda_{\min}},
$$

avec \(\lambda\) en \(\mu\mathrm m\) et \(\tilde\nu\) en
\(\mathrm{cm^{-1}}\).

RADIS calcule ensuite une transmission spectrale :

$$
\mathcal T_{\mathrm{ref}}(\tilde\nu)
$$

avec la base `("hitran", "range")`, le milieu `air`, la pression
\(p_{\mathrm{milieu}}\), la temperature \(T_k\), la longueur \(L_k\), un
elargissement tronque a \(50\ \mathrm{cm^{-1}}\), et les isotopes principaux.

Pour CO2 :

$$
\chi_{\mathrm{CO2}} = C \times 10^{-6},
$$

ou \(C\) parcourt les concentrations testees. Le defaut actuel est :

```text
co2_values = 280,420,700
```

Pour H2O, la fraction molaire est deduite de la masse d'eau et de la masse
d'air de la couche. Avec :

$$
q_k = \frac{M_{\mathrm{H2O},k}}{M_{\mathrm{air},k}},
$$

le script utilise :

$$
n_{\mathrm{H2O}}^\ast=\frac{q_k}{M_{\mathrm{H2O,mol}}},
\qquad
n_{\mathrm{air\,sec}}^\ast=\frac{1-q_k}{M_{\mathrm{air,mol}}},
$$

$$
\chi_{\mathrm{H2O},k}
=
\frac{n_{\mathrm{H2O}}^\ast}
{n_{\mathrm{H2O}}^\ast+n_{\mathrm{air\,sec}}^\ast}.
$$

Les couches seches sont ignorees. L'option `--h2o-scale-values` peut multiplier
les masses H2O pour tester la sensibilite, mais le defaut `1` est le calibrage
physiquement le plus direct : il utilise les profils humides du paquet.

## 5. Moyenne de transmission par poids de Planck

RADIS fournit une transmission fine en nombre d'onde. Le modele 3, lui, ne
conserve qu'une transmission moyenne par bande. La moyenne doit donc ponderer
les frequences qui contribuent vraiment au flux thermique de la couche.

Le poids utilise est proportionnel a la loi de Planck en nombre d'onde :

$$
w(\tilde\nu,T_k)
\propto
\frac{\tilde\nu_m^3}
{\exp\!\left(\frac{hc\tilde\nu_m}{k_B T_k}\right)-1},
\qquad
\tilde\nu_m = 100\,\tilde\nu.
$$

La constante multiplicative s'annule dans la moyenne :

$$
\overline{\mathcal T}_{\mathrm{ref},i}
=
\frac{
\int_{\tilde\nu_{\min}}^{\tilde\nu_{\max}}
w(\tilde\nu,T_k)\,
\mathcal T_{\mathrm{ref}}(\tilde\nu)\,
d\tilde\nu
}{
\int_{\tilde\nu_{\min}}^{\tilde\nu_{\max}}
w(\tilde\nu,T_k)\,
d\tilde\nu
}.
$$

C'est l'etape qui relie le calcul spectroscopique au flux radiatif : une raie
situee dans une zone peu emise par la couche pese peu dans le coefficient final.

## 6. Profondeur optique equivalente

Le modele 3 utilise :

$$
\mathcal T=\exp(-D\tau).
$$

On inverse donc cette relation, avec un plancher numerique sur la transmission :

$$
\tau_{\mathrm{eq},i}
=
-\frac{\ln(\max(\overline{\mathcal T}_{\mathrm{ref},i},10^{-12}))}{D}.
$$

Il faut diviser par \(D=1.66\) ici, parce que \(D\) est deja applique dans le
modele lors du calcul de transmission. Sans cette division, le coefficient
serait trop grand d'un facteur diffusif.

## 7. Normalisation CO2

Pour une mesure CO2 \(i=(k,b,C)\), le facteur deja connu par le modele est :

$$
X_{\mathrm{CO2},i}
=
\frac{C}{C_0}
\frac{\Delta p_k}{p_0}.
$$

La mesure propose donc :

$$
\widehat a_{\mathrm{CO2},i}
=
\frac{\tau_{\mathrm{eq},i}}{X_{\mathrm{CO2},i}}.
$$

Pour chaque bande CO2 :

$$
a_{\mathrm{CO2},b}
=
\operatorname{mediane}_{i\in b}
\left(\widehat a_{\mathrm{CO2},i}\right).
$$

Le script actuel ne fait pas de recalage global sur un forcage
\(280\rightarrow560\ \mathrm{ppm}\). L'ancienne note CO2 en parlait, mais ce
n'est plus l'etat du code : le flux actuel est strictement
`HITRAN/RADIS -> moyenne Planck -> tau equivalent -> mediane(tau/X)`.

## 8. Normalisation H2O

Pour une mesure H2O \(i=(k,b)\), le facteur de modele est :

$$
X_{\mathrm{H2O},i}
=
\frac{M_{\mathrm{H2O},k}}{M_0}.
$$

La mesure propose :

$$
\widehat a_{\mathrm{H2O},i}
=
\frac{\tau_{\mathrm{eq},i}}{X_{\mathrm{H2O},i}}.
$$

Pour chaque bande portant une contribution H2O :

$$
a_{\mathrm{H2O},b}^{\mathrm{HITRAN}}
=
\operatorname{mediane}_{i\in b}
\left(\widehat a_{\mathrm{H2O},i}\right).
$$

Le script H2O possede ensuite un facteur explicite optionnel :

$$
a_{\mathrm{H2O},b}^{\mathrm{modele3}}
=
s_{\mathrm{H2O}}\,
a_{\mathrm{H2O},b}^{\mathrm{HITRAN}}.
$$

Par defaut \(s_{\mathrm{H2O}}=1\). Il ne faut utiliser un autre facteur que si
un recalage separe est documente ; sinon on melange calibrage spectroscopique
et correction empirique.

## 9. Bandes concernees

Le calibrage CO2 parcourt les bandes ou \(a_{\mathrm{CO2}}>0\), notamment le
decoupage coeur/ailes autour de \(15\ \mu\mathrm m\) et \(4.3\ \mu\mathrm m\).

Le calibrage H2O parcourt toutes les bandes ou \(a_{\mathrm{H2O}}>0\). Cela
inclut les bandes propres a la vapeur d'eau et les intervalles CO2 ou la vapeur
d'eau contribue aussi a l'opacite. En production, l'addition :

$$
\tau_{\mathrm{CO2}}+\tau_{\mathrm{H2O}}
$$

revient a supposer que les profondeurs optiques effectives s'additionnent comme
dans Beer-Lambert. Le calibrage ne fait pas un ajustement simultane complet des
chevauchements de raies CO2/H2O ; c'est une approximation de grande bande.

## 10. Ecriture du resultat

Les deux scripts mettent a jour le meme paquet :

```text
modele3/ressources/coefficients_opacite_modele3.npz
```

Quand on ecrit CO2, les coefficients H2O et nuages existants sont conserves.
Quand on ecrit H2O, les coefficients CO2 et nuages existants sont conserves.
La generation complete lance CO2 puis H2O.

Commandes utiles depuis la racine :

```bash
./.venv/bin/python -m pip install -r requirements.txt

./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_co2 --dry-run
./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_h2o --dry-run

./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_co2 \
  --latitudes=-60,-30,0,30,60 \
  --longitudes=0 \
  --mois=1,4,7,10 \
  --co2-values=280,420,560,1120

./.venv/bin/python -m modele3.codes_python.calibrer_coefficients_h2o \
  --latitudes=-60,-30,0,30,60 \
  --longitudes=0 \
  --mois=1,4,7,10
```

## 11. Message a retenir

Le calibrage ne cherche pas la meilleure physique spectrale possible. Il cherche
la meilleure traduction, dans le formalisme simplifie du modele 3, d'un calcul
HITRAN/RADIS de reference :

$$
\boxed{
\text{HITRAN/RADIS}
\rightarrow
\overline{\mathcal T}_{\mathrm{Planck}}
\rightarrow
\tau_{\mathrm{eq}}
\rightarrow
\operatorname{mediane}\!\left(\frac{\tau_{\mathrm{eq}}}{X}\right)
\rightarrow
a_b
}
$$

Les coefficients obtenus sont donc des pentes effectives de profondeur optique
par bande. Ils sont faits pour le noyau radiatif du modele 3, avec ses
normalisations \(280\ \mathrm{ppm}\), \(101325\ \mathrm{Pa}\),
\(10\ \mathrm{kg\,m^{-2}}\) et son facteur diffusif \(1.66\). Ils ne doivent
pas etre lus comme des sections efficaces HITRAN, ni extrapoles hors du domaine
de couches, temperatures, pressions et humidites utilise pour le calibrage.
