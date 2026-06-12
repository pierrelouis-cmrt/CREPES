# Projet Climat — plan simplifié pour un modèle CO₂ multicouche

**But :** remplacer le terme atmosphérique global de l'ancien modèle par une atmosphère verticale en couches, où le CO₂ agit sur les flux infrarouges.  
**Idée centrale :** ne jamais imposer directement `T = f(CO2)`. Le CO₂ modifie l'opacité infrarouge ; les températures changent ensuite via les bilans d'énergie.

```text
CO2 -> opacité IR des couches -> flux IR montants/descendants -> bilans d'énergie -> températures
```

L'ancien terme de surface

$$
+\sigma T_{atm}^4
$$

doit donc être remplacé par un flux infrarouge descendant calculé explicitement :

$$
F^{\downarrow}_{IR}(z=0).
$$

---

## 1. Noyau commun à tous les modèles

### 1.1 Variables

Pour chaque colonne atmosphérique :

- surface : température $T_s(t)$ ;
- atmosphère : $N$ couches de températures $T_k(t)$, $k=0,\dots,N-1$ ;
- interfaces : $k=0$ à la surface, $k=N$ au sommet de l'atmosphère ;
- flux infrarouge montant : $F^\uparrow_k$ ;
- flux infrarouge descendant : $F^\downarrow_k$.

On travaille de préférence en **coordonnée pression**, plus simple que l'altitude au début.

### 1.2 Capacité thermique des couches

Pour une couche comprise entre deux pressions $p_k$ et $p_{k+1}$ :

$$
C_k = c_{p,air}\frac{\Delta p_k}{g}
$$

avec :

$$
\Delta p_k = p_k - p_{k+1} > 0.
$$

Valeurs de référence :

$$
c_{p,air}\simeq 1004\ \mathrm{J\,kg^{-1}\,K^{-1}},
\qquad
 g\simeq 9.81\ \mathrm{m\,s^{-2}}.
$$

### 1.3 Émission thermique par bande spectrale

Pour une bande infrarouge $b=[\lambda_1,\lambda_2]$, l'émission de corps noir hémisphérique est :

$$
E_b(T)=\int_{\lambda_1}^{\lambda_2}\pi B_\lambda(T)\,d\lambda.
$$

Avec la loi de Planck :

$$
B_\lambda(T)=\frac{2hc^2}{\lambda^5}\frac{1}{\exp\left(\frac{hc}{\lambda k_BT}\right)-1}.
$$

Si on ne découpe pas le spectre, on retrouve :

$$
\sum_b E_b(T) \simeq \sigma T^4.
$$

### 1.4 Transmission d'une couche

Pour une couche $k$ et une bande $b$ :

$$
\mathcal{T}_{k,b}=\exp(-D\Delta\tau_{k,b}),
$$

$$
\varepsilon_{k,b}=1-\mathcal{T}_{k,b}.
$$

- $\mathcal{T}_{k,b}$ : transmission ;
- $\varepsilon_{k,b}$ : émissivité/absorptivité effective ;
- $D$ : facteur diffusif. Prendre $D=1$ au début ; tester ensuite $D\simeq1.66$.

### 1.5 Opacité CO₂ simplifiée

Au début, on ne fait pas de spectroscopie ligne par ligne. On utilise une opacité effective par bande :

$$
\Delta\tau_{k,b}
=
\Delta\tau^{fixe}_{k,b}
+
 a_b\frac{C_{CO2}}{C_{0,CO2}}\frac{\Delta p_k}{p_s}.
$$

- $C_{CO2}$ : concentration en ppm ;
- $C_{0,CO2}$ : référence, par exemple 280 ppm ;
- $a_b$ : coefficient optique de la bande $b$, à calibrer ;
- $p_s\simeq101325\ \mathrm{Pa}$.

Le but n'est pas que $a_b$ soit une constante fondamentale. C'est un paramètre de modèle, à régler pour reproduire le bon ordre de grandeur du forçage CO₂.

### 1.6 Flux infrarouges dans les couches

Condition à la surface :

$$
F^\uparrow_{0,b}=E_b(T_s).
$$

Condition au sommet de l'atmosphère :

$$
F^\downarrow_{N,b}=0.
$$

Propagation vers le haut :

$$
F^\uparrow_{k+1,b}
=
\mathcal{T}_{k,b}F^\uparrow_{k,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k).
$$

Propagation vers le bas :

$$
F^\downarrow_{k,b}
=
\mathcal{T}_{k,b}F^\downarrow_{k+1,b}
+
(1-\mathcal{T}_{k,b})E_b(T_k).
$$

Les grandeurs importantes à sortir sont :

$$
OLR = \sum_b F^\uparrow_{N,b}
$$

et

$$
LW_{down,surface}=\sum_b F^\downarrow_{0,b}.
$$

### 1.7 Bilans d'énergie

Surface :

$$
C_s\frac{dT_s}{dt}
=
F_{SW,surface}
+
F^\downarrow_{IR}(0)
-
\sigma T_s^4
-
Q_{latent}
-
Q_{sensible}.
$$

Au début :

$$
Q_{latent}=Q_{sensible}=0.
$$

Couche atmosphérique $k$ :

$$
C_k\frac{dT_k}{dt}
=
\sum_b\left[
(F^\uparrow_{k,b}-F^\uparrow_{k+1,b})
+
(F^\downarrow_{k+1,b}-F^\downarrow_{k,b})
\right].
$$

Si le terme est positif, la couche se réchauffe.

---

## 2. Les 5 modèles à construire

## Modèle 1 — Colonne globale moyenne, atmosphère multicouche minimale

**Objectif :** avoir tout de suite une vraie atmosphère multicouche, mais avec peu de complexité.

### Choix simples

- Une seule colonne représentant la moyenne globale terrestre.
- $N=3$ ou $N=4$ couches en pression.
- Exemple :

```python
p_edges_hPa = [1000, 700, 300, 100, 10]  # 4 couches
```

- Deux ou trois bandes IR :
  - fenêtre atmosphérique, environ $8$-$13\ \mu m$, presque transparente ;
  - bande CO₂ effective, environ $13$-$17\ \mu m$ ;
  - reste de l'IR, transparent ou faiblement absorbant.
- Pas de nuages, pas de vapeur d'eau, pas de convection, pas d'échanges horizontaux.
- Solaire global moyen :

$$
F_{SW,surface}=\frac{S_0(1-A)}{4}.
$$

avec $S_0\simeq1361\ \mathrm{W\,m^{-2}}$ et $A\simeq0.30$.

### À faire

- [ ] Créer `constants.py`.
- [ ] Créer `vertical_grid.py` avec `p_edges`, `p_mid`, `C_atm`.
- [ ] Créer `radiation_longwave.py` avec les flux $F^\uparrow$ et $F^\downarrow$.
- [ ] Créer `column_model.py` avec $T_s$, $T_k$ et les équations d'évolution.
- [ ] Sortir au minimum : `T_surface`, `T_layers`, `OLR`, `LW_down_surface`, `energy_imbalance`.

### Calibration minimale

À profil de température fixé, calibrer $a_b$ pour que :

$$
OLR(280\ ppm)-OLR(560\ ppm)\approx 3.7\text{ à }3.9\ \mathrm{W\,m^{-2}}.
$$

Ce test est essentiel : il vérifie que l'effet marginal du CO₂ est du bon ordre de grandeur.

### Tests obligatoires

- [ ] Si toutes les opacités valent zéro :

$$
F^\downarrow_{IR}(0)=0,
\qquad
OLR=\sigma T_s^4.
$$

- [ ] Si on augmente le CO₂ à températures fixées, l'OLR doit diminuer.
- [ ] Si on laisse évoluer les températures, le système doit tendre vers un nouvel équilibre.

---

## Modèle 2 — Même colonne, mais meilleure verticale et meilleur spectre

**Objectif :** rendre le noyau radiatif plus crédible sans passer à un vrai modèle complexe.

### Améliorations

- Passer à $N=8$ ou $N=10$ couches en pression.
- Utiliser un profil initial proche de l'atmosphère standard : troposphère froide avec l'altitude, tropopause, stratosphère simplifiée.
- Découper la bande CO₂ en sous-bandes :
  - ailes faibles ;
  - bande centrale saturée ;
  - éventuellement fenêtre atmosphérique séparée.
- Tester le facteur diffusif $D=1.66$.
- Ajouter un **ajustement convectif très simple** seulement si le profil devient irréaliste : si le gradient vertical dépasse un seuil, mélanger les couches concernées en conservant l'énergie.

### À faire

- [ ] Remplacer les 3-4 couches par 8-10 couches.
- [ ] Ajouter un fichier `radiative_params.py` qui contient les bandes et les coefficients $a_b$.
- [ ] Recalibrer les coefficients optiques.
- [ ] Comparer les résultats entre 4, 8 et 10 couches.
- [ ] Vérifier que les résultats ne changent pas violemment quand on affine la grille.

### Tests obligatoires

- [ ] Le forçage $280\to560$ ppm reste proche de $3.7$-$3.9\ \mathrm{W\,m^{-2}}$.
- [ ] Le comportement avec le CO₂ est à peu près logarithmique :

$$
\Delta F_{CO2}\propto\ln(C/C_0).
$$

- [ ] Le bilan d'énergie colonne est cohérent :

$$
\frac{dE_{colonne}}{dt}
\approx
F_{SW,absorbé}-OLR
$$

si latent, convection et diffusion sont désactivés.

---

## Modèle 3 — Grille horizontale de colonnes indépendantes

**Objectif :** passer de la colonne globale à une Terre discrétisée, sans encore ajouter de dynamique horizontale.

### Choix simples

- Chaque point de grille possède sa propre colonne multicouche.
- Pas de vent, pas de diffusion horizontale : les colonnes sont indépendantes.
- On réutilise exactement le même noyau vertical que dans le modèle 2.
- Le solaire devient local :

$$
F_{SW,surface}(\theta,\varphi,t)
=
S_0\max(\cos i,0)(1-A_{surface}).
$$

### À faire

- [ ] Définir une grille simple : par exemple $5^\circ\times5^\circ$ ou la grille déjà utilisée dans le projet.
- [ ] Brancher l'angle solaire local et le cycle jour/nuit.
- [ ] Brancher l'albédo local si les données existent déjà.
- [ ] Vectoriser le code pour avoir :

```text
T_surface[lat, lon]
T_atm[layer, lat, lon]
```

- [ ] Sortir des cartes de $T_s$, $OLR$, $LW_{down,surface}$.

### Tests obligatoires

- [ ] La moyenne globale du solaire absorbé doit rester proche de $S_0(1-A)/4$ si l'albédo est uniforme.
- [ ] En moyenne globale, le modèle 3 doit rester proche du modèle 2 dans le cas uniforme.
- [ ] Les pôles et les zones de nuit ne doivent pas recevoir de flux solaire direct négatif.

---

## Modèle 4 — Réintégration des anciennes briques de surface

**Objectif :** récupérer ce qui était utile dans le projet pré-CO₂ sans double-compter l'atmosphère.

### À réutiliser progressivement

- Capacité thermique de surface déjà documentée.
- Albédo de surface mensuel.
- Nuages, si les données sont déjà propres.
- Chaleur latente / évapotranspiration.
- Convection sensible, seulement si l'interface est claire.

### Point crucial

Ne pas garder simultanément :

$$
+\sigma T_{atm}^4
$$

et

$$
+F^\downarrow_{IR}(0).
$$

Le premier terme était une approximation globale de l'effet atmosphérique ; le second est désormais calculé par les couches. Les deux ensemble risquent de compter deux fois le même effet.

Le bilan de surface devient :

$$
C_s\frac{dT_s}{dt}
=
F_{SW,surface}
+F^\downarrow_{IR}(0)
-\sigma T_s^4
-Q_{latent}
-Q_{sensible}.
$$

### À faire

- [ ] Remplacer définitivement `sigma*T_atm**4` par `LW_down_surface`.
- [ ] Ajouter les modules anciens un par un, avec un interrupteur `on/off`.
- [ ] Vérifier à chaque ajout que le bilan d'énergie reste compréhensible.
- [ ] Documenter les unités de chaque flux en W/m².

### Tests obligatoires

- [ ] Si tous les modules de surface sont désactivés, on retrouve le modèle 3.
- [ ] Si le latent est activé, il refroidit bien la surface quand il est positif.
- [ ] Si les nuages sont activés, ils modifient clairement le solaire et/ou l'IR selon le choix retenu.

---

## Modèle 5 — Validation, autres gaz et amélioration scientifique

**Objectif :** transformer le prototype en modèle défendable dans le rapport.

### Validation minimale

Comparer les sorties globales du modèle à des ordres de grandeur observés :

- flux solaire absorbé global moyen ;
- OLR global moyen ;
- flux IR descendant à la surface ;
- réponse au doublement du CO₂.

La validation ne doit pas dire « le modèle reproduit parfaitement la Terre ». Elle doit dire clairement ce qui est bien reproduit et ce qui ne l'est pas.

### Autres gaz, en option

Après le CO₂ :

- vapeur d'eau : très importante, mais difficile car elle dépend de la température ;
- CH₄ et N₂O : ajoutables via des formules de forçage ou des bandes effectives ;
- ozone : utile pour la stratosphère, mais pas prioritaire.

### Améliorations possibles

- Utiliser des formules de forçage publiées au lieu d'une calibration maison pour CH₄/N₂O.
- Utiliser des données HITRAN ou un modèle type RRTM/ecRad si le groupe veut aller beaucoup plus loin.
- Comparer aux données CERES pour OLR et flux radiatifs.

### Tests obligatoires

- [ ] Le doublement du CO₂ donne un forçage proche de la littérature.
- [ ] Les sorties sont comparées à des données ou à des ordres de grandeur sourcés.
- [ ] Les hypothèses simplificatrices sont listées explicitement dans le rapport.

---

## 3. Ce qu'il faut éviter

- Ne pas écrire directement `T_surface = T_surface + f(CO2)`.
- Ne pas ajouter une formule de forçage CO₂ en plus de l'opacité CO₂ des couches, sauf comme test ou calibration.
- Ne pas garder l'ancien terme `sigma*T_atm**4` si le flux descendant des couches est déjà utilisé.
- Ne pas prétendre qu'un modèle CO₂ seul reproduit toute la température terrestre : vapeur d'eau, nuages, convection et océans sont majeurs.
- Ne pas mélanger les unités : CO₂ en ppm, pressions en Pa, flux en W/m², températures en K.

---

## 4. Organisation du code conseillée

```text
project/
  constants.py
  vertical_grid.py
  planck.py
  radiation_longwave.py
  surface_budget.py
  column_model.py
  grid_model.py
  radiative_params.py
  validation.py
  run_column.py
  run_grid.py
  README.md
```

Chaque module doit avoir des tests simples. Les tests les plus importants sont ceux de `radiation_longwave.py` et `column_model.py`.

---

## 5. Sources scientifiques utiles

### Sources du projet

- `THEORIE.md` : résumé des équations pré-CO₂ et des modules déjà présents.
- `attendus et consignes projet climat.pdf` : objectifs du projet, notamment l'effet de la composition atmosphérique sur la puissance surfacique reçue.

### Forçage radiatif du CO₂

- IPCC AR6 WGI, Chapter 7 — définition du forçage radiatif effectif et bilan énergétique :  
  https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/
- IPCC AR6 WGI, Chapter 7 Supplementary Material — valeur $F_{2\times CO2}=3.93\ \mathrm{W\,m^{-2}}$ :  
  https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf
- Myhre et al. (1998), *New estimates of radiative forcing due to well mixed greenhouse gases*, Geophysical Research Letters. Formule classique :  
  $\Delta F = 5.35\ln(C/C_0)$.  
  https://doi.org/10.1029/98GL01908
- Etminan et al. (2016), *Radiative forcing of carbon dioxide, methane, and nitrous oxide*, Geophysical Research Letters. Formules plus récentes pour CO₂, CH₄, N₂O :  
  https://doi.org/10.1002/2016GL071930

### Transfert radiatif et spectroscopie

- Gordon et al. (2022), *The HITRAN2020 molecular spectroscopic database*, Journal of Quantitative Spectroscopy and Radiative Transfer :  
  https://doi.org/10.1016/j.jqsrt.2021.107949
- Mlawer et al. (1997), *Radiative transfer for inhomogeneous atmospheres: RRTM*, Journal of Geophysical Research :  
  https://doi.org/10.1029/97JD00237
- Hogan & Bozzo (2018), *A Flexible and Efficient Radiation Scheme for the ECMWF Model*, Journal of Advances in Modeling Earth Systems :  
  https://doi.org/10.1029/2018MS001364

### Données et validation

- NASA — explication du budget radiatif terrestre :  
  https://science.nasa.gov/ems/13_radiationbudget/
- NASA CERES — données de bilan radiatif surface et sommet de l'atmosphère :  
  https://ceres.larc.nasa.gov/
- NASA Earthdata CERES EBAF — produit de référence pour flux TOA/surface :  
  https://www.earthdata.nasa.gov/data/catalog/larc-cloud-ceres-ebaf-edition4.2
- NOAA Global Monitoring Laboratory — concentration atmosphérique de CO₂ :  
  https://gml.noaa.gov/ccgg/trends/
- U.S. Standard Atmosphere 1976 — profils pression/température de référence :  
  https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf

---

## 6. Résumé en une phrase

Commencer directement par une **colonne atmosphérique multicouche**, faire dépendre le CO₂ de l'**opacité infrarouge** des couches, calculer les flux $F^\uparrow$ et $F^\downarrow$, puis faire évoluer les températures par bilan d'énergie ; ensuite seulement ajouter plus de couches, une grille terrestre, puis les anciens modules de surface.
