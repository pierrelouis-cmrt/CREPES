# Modèle 2 - colonne CO2 à 6 couches

Mini-glossaire : OLR = flux infrarouge sortant vers l'espace ; LW = infrarouge thermique à grande longueur d'onde.

Faire Ctrl + shift + V pour visualiser le REAME en LATEX

Le modèle 2 est un prototype de colonne atmosphérique verticale. Il sert à
remplacer une correction atmosphérique globale par un calcul explicite de
l'absorption et de la réémission infrarouge par couches.

L'idée centrale est la même que dans le plan d'attaque :

```text
CO2 -> opacité infrarouge -> transmission / émissivité -> flux IR montants et descendants
```

Le script ne prédit pas encore l'évolution temporelle du climat. Les
températures sont imposées, puis le modèle calcule les flux infrarouges
correspondants. Cette étape est volontairement limitée : elle permet de vérifier
que le noyau radiatif est compréhensible avant d'ajouter une dynamique thermique.

## Fichiers du dossier

| Fichier | Rôle |
| --- | --- |
| `modele2.py` | Point d'entrée du modèle 2. Calcule les opacités des couches, les transmissions, les émissivités, le flux infrarouge sortant au sommet et le flux infrarouge descendant à la surface. |
| `ressources/profil_vertical_atmosphere_co2.py` | Outil de profil vertical. Calcule pression, température standard, CO2, pression partielle du CO2 et concentration moléculaire en fonction de l'altitude. |
| `ressources/profil_vertical_atmosphere_co2.csv` | Export numérique du profil vertical de référence. |
| `ressources/profil_vertical_atmosphere_co2.png` | Graphique de diagnostic du profil vertical de référence. |
| `requirements.txt` | Dépendances Python nécessaires aux scripts du modèle 2. |

Les fichiers `evolution_co2.py` et `spectre_absorbance_co2.py` ne font plus
partie de cette version nettoyée du modèle 2. Le plan d'attaque prévoit bien un
meilleur découpage spectral et une calibration plus poussée, mais ce n'est pas
encore intégré dans ce dossier.

## Installation

Depuis la racine du dépôt :

```bash
./.venv/bin/python -m pip install -r modele2/requirements.txt
```

Les dépendances actuelles sont volontairement réduites :

- `numpy` pour les calculs numériques ;
- `matplotlib` pour le graphique du profil vertical.

## Lancer le modèle 2

Depuis la racine du dépôt :

```bash
./.venv/bin/python modele2/modele2.py
```

Le script affiche trois blocs :

- `couches_atmospheriques` : altitude, température, pressions basse/haute et
  CO2 moyen de chaque couche ;
- `opacites_par_couche` : profondeur optique, transmission et émissivité pour
  chaque couple couche/bande infrarouge ;
- les deux flux globaux calculés :
  - `flux_infrarouge_sortant_sommet_W_m2`, c'est-à-dire le flux IR sortant de la couche limite avec le vide;
  - `flux_infrarouge_descendant_surface_W_m2`, c'est-à-dire le flux IR
    atmosphérique reçu par la surface.

## Générer le profil vertical

Pour régénérer le CSV et le graphique du profil vertical :

```bash
./.venv/bin/python modele2/ressources/profil_vertical_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/ressources/profil_vertical_atmosphere_co2.png --csv modele2/ressources/profil_vertical_atmosphere_co2.csv --no-plot
```

Options utiles :

| Option | Signification |
| --- | --- |
| `--max-altitude-km` | Altitude maximale du profil. Doit rester inférieure ou égale à 84,852 km. |
| `--step-m` | Pas vertical du profil en mètres. |
| `--surface-co2-ppm` | Concentration de CO2 à la surface. |
| `--co2-gradient-ppm-per-km` | Gradient linéaire du CO2 en ppm/km. La valeur `0` signifie que le CO2 est bien mélangé. |
| `--surface-pressure-pa` | Pression de surface en pascals. |
| `--surface-temperature-k` | Température de surface en kelvins. |
| `--output` | Chemin du graphique produit. |
| `--csv` | Chemin du CSV produit. |
| `--no-plot` | Calcule sans ouvrir de fenêtre graphique. |

## Hypothèses actuelles

Le modèle 2 repose sur des hypothèses simples :

- une seule colonne atmosphérique verticale ;
- 6 couches imposées entre 0 et 80 km ;
- températures des couches imposées, donc pas encore de bilan d'énergie
  évolutif ;
- pression issue de l'atmosphère standard 1976 ;
- CO2 constant par défaut à 420 ppm ;
- deux bandes absorbantes simplifiées pour le CO2 ;
- pas de vapeur d'eau, pas de nuages, pas de convection, pas de diffusion et
  pas d'échanges horizontaux ;
- le reste du spectre infrarouge est traité comme transparent.

Ces choix ne cherchent pas à reproduire toute l'atmosphère réelle. Ils servent à
obtenir un noyau radiatif lisible, testable et améliorable.

## Couches utilisées

Les températures sont des moyennes calculée au préalable à la main. Les pressions sont calculées avec
le profil d'atmosphère standard.

| Couche | Altitude | Zone | Température |
| --- | ---: | --- | ---: |
| 1 | 0-5 km | Troposphère basse | 271 K |
| 2 | 5-10 km | Troposphère moyenne | 236 K |
| 3 | 10-30 km | Tropopause | 223 K |
| 4 | 30-50 km | Stratosphère | 257 K |
| 5 | 50-65 km | Mésosphère basse | 252 K |
| 6 | 65-80 km | Mésosphère haute | 212 K |

Ce découpage est encore grossier. Dans le plan d'attaque, le modèle 2 devait
plutôt évoluer vers 8 à 10 couches et vérifier que les résultats changent peu
quand on affine la grille verticale.

## Moyenne de CO2 par couche

Le script `ressources/profil_vertical_atmosphere_co2.py` produit un profil :

```text
altitude -> pression, température, CO2
```

Pour une couche \(k\), le modèle calcule une moyenne de CO2 pondérée par la
masse d'air. En équilibre hydrostatique, la masse d'air par unité de surface est
proportionnelle à la différence de pression. On utilise donc :

$$
\overline{C}_k
=
\frac{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} C(p)\,dp}
{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} dp}
$$

Dans la version actuelle, le profil de CO2 est constant :

$$
C(p)=420\ \mathrm{ppm}
$$

Donc toutes les couches ont :

$$
\overline{C}_k=420\ \mathrm{ppm}
$$

La méthode reste toutefois correcte si on ajoute plus tard un gradient vertical
de CO2.

## Opacité infrarouge

Dans une bande infrarouge \(b\), la loi de Beer-Lambert donne :

$$
dI_b=-\sigma_b n_{\mathrm{CO}_2} I_b\,ds
$$

Après intégration sur une couche :

$$
\frac{I_{b,\mathrm{sortie}}}{I_{b,\mathrm{entree}}}
=
\exp(-\tau_b)
$$

La profondeur optique est :

$$
\tau_b
=
\int_{\mathrm{couche}}\sigma_b n_{\mathrm{CO}_2}\,ds
$$

Pour un trajet vertical et un CO2 bien mélangé :

$$
n_{\mathrm{CO}_2}=\chi_{\mathrm{CO}_2}n_{\mathrm{air}}
$$

avec :

$$
\chi_{\mathrm{CO}_2}=C_{\mathrm{CO}_2}\times10^{-6}
$$

si la concentration de CO2 est exprimée en ppm.

L'équilibre hydrostatique donne :

$$
\frac{dp}{dz}=-\rho g
$$

Donc la masse d'air d'une couche est proportionnelle à :

$$
\Delta p_k=p_{\mathrm{bas},k}-p_{\mathrm{haut},k}
$$

Le modèle regroupe toute la complexité spectrale dans un coefficient effectif
\(a_b\). La profondeur optique utilisée dans le code est :

$$
\boxed{
\Delta\tau_{k,b}
=
a_b
\frac{\overline{C}_k}{C_0}
\frac{\Delta p_k}{p_s}
}
$$

avec :

$$
\begin{aligned}
C_0 &= 280\ \mathrm{ppm},\\
p_s &= 101325\ \mathrm{Pa}
\end{aligned}
$$

Dans `modele2.py`, cela correspond à :

```python
tau = a_bande * (co2_moyen_ppm / 280.0) * (delta_p / p_surface)
```

Le coefficient \(a_b\) n'est pas une constante fondamentale. C'est un paramètre
de modèle à calibrer pour obtenir un ordre de grandeur réaliste du forçage CO2.

## Transmission et émissivité

La transmission de la couche est :

$$
\mathcal{T}_{k,b}=\exp(-D\Delta\tau_{k,b})
$$

Le facteur diffusif \(D\) vaut actuellement :

$$
D=1
$$

Une valeur proche de \(1{,}66\) pourra être testée plus tard pour représenter
des trajets radiatifs obliques moyens.

Sans diffusion ni réflexion, la fraction non transmise est absorbée :

$$
\alpha_{k,b}=1-\mathcal{T}_{k,b}
$$

Par la loi de Kirchhoff, à l'équilibre thermique local :

$$
\varepsilon_{k,b}=\alpha_{k,b}
$$

Donc :

$$
\boxed{\varepsilon_{k,b}=1-\mathcal{T}_{k,b}}
$$

## Flux infrarouges

Pour une bande \(b\), le flux de corps noir de température \(T\) est :

$$
E_b(T)=\int_{\lambda_1}^{\lambda_2}\pi B_\lambda(T)\,d\lambda
$$

Le flux montant part de la surface :

$$
F^\uparrow_{0,b}=E_b(T_s)
$$

Puis chaque couche transforme le flux montant selon :

$$
F^\uparrow_{k+1,b}
=
\mathcal{T}_{k,b}F^\uparrow_{k,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k)
$$

Le flux descendant part du sommet de l'atmosphère avec :

$$
F^\downarrow_{N,b}=0
$$

Puis chaque couche transforme le flux descendant selon :

$$
F^\downarrow_{k,b}
=
\mathcal{T}_{k,b}F^\downarrow_{k+1,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k)
$$

Les sorties principales sont :
OLR est le flux d'énergie IR sortant de la dernière couche d'atmosphère vers le vide 

$$
OLR=\sum_b F^\uparrow_{N,b}
$$

et :

$$
LW_{\mathrm{down},\mathrm{surface}}=\sum_b F^\downarrow_{0,b}
$$

Dans le script, les bandes CO2 sont traitées explicitement. Le reste du spectre
est considéré transparent et sort directement vers l'espace.

## Limites et validations à faire

Cette version est propre pour lire et tester le noyau, mais elle n'est pas
encore calibrée scientifiquement. Les prochaines validations importantes sont :

- vérifier que si les coefficients d'opacité valent zéro, alors
  \(F^\downarrow_{\mathrm{IR}}(0)=0\) et \(OLR=\sigma T_s^4\) ;
- vérifier qu'à températures fixées, augmenter le CO2 diminue l'OLR ;
- calibrer les coefficients \(a_b\) pour que le doublement
  \(280 \to 560\ \mathrm{ppm}\) donne un forçage proche de
  \(3{,}7\) à \(3{,}9\ \mathrm{W\,m^{-2}}\) ;
- tester une grille verticale plus fine, par exemple 8 à 10 couches ;
- découper la bande CO2 en sous-bandes plus crédibles, notamment ailes faibles
  et coeur saturé ;
- ajouter ensuite seulement les bilans d'énergie et l'évolution temporelle des
  températures.
