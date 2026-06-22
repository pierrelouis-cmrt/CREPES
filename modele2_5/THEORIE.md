# Théorie — modèle 2.5

## Évolution par rapport au modèle 2

Le modèle 2.5 conserve une colonne radiative à températures imposées, mais utilise dix couches définies par pression, un profil standard de température, un découpage spectral cœur/ailes et des opacités calibrées.

## Grille verticale et profil standard

Les interfaces de pression sont :

```text
1013.25, 850, 700, 500, 300, 200, 100, 50, 20, 10, 1 hPa
```

La température de couche est la moyenne verticale du profil standard 1976 :

$$
\overline{T}_k = \frac{1}{z_{haut,k}-z_{bas,k}}\int_{z_{bas,k}}^{z_{haut,k}}T(z)\,dz
$$

En altitude géopotentielle $h$ :

$$
h = \frac{r_0z}{r_0+z}, \qquad T(h)=T_b+L_b(h-h_b)
$$

Si $L_b\ne0$ :

$$
p(h)=p_b\left(\frac{T(h)}{T_b}\right)^{-g_0/(R_{air}L_b)}
$$

Sinon :

$$
p(h)=p_b\exp\left[-\frac{g_0(h-h_b)}{R_{air}T_b}\right]
$$

## CO₂ et bandes spectrales

Le CO₂ est bien mélangé par défaut : $C(z)=420\ \mathrm{ppm}$. Sa moyenne de couche est pondérée par la masse d’air, donc par $\Delta p$.

Les bandes retenues sont la bande thermique de 15 µm et la bande de 4,3 µm. Le découpage cœur/ailes limite l’effet artificiel de saturation d’une seule bande large. La bande de 2,7 µm est négligée, sa contribution diagnostiquée étant très faible.

## Opacité et flux

La profondeur optique effective est :

$$
\Delta\tau_{k,b}=a_b\frac{\overline{C}_k}{280\ \mathrm{ppm}}\frac{\Delta p_k}{101325\ \mathrm{Pa}}
$$

Le facteur diffusif est $D=1.66$ :

$$
\mathcal{T}_{k,b}=\exp(-D\Delta\tau_{k,b}), \qquad \varepsilon_{k,b}=1-\mathcal{T}_{k,b}
$$

Les flux montants et descendants sont propagés couche par couche comme dans le modèle 2 :

$$
F^\uparrow_{k+1,b}=\mathcal{T}_{k,b}F^\uparrow_{k,b}+(1-\mathcal{T}_{k,b})E_b(T_k)
$$

$$
F^\downarrow_{k,b}=\mathcal{T}_{k,b}F^\downarrow_{k+1,b}+(1-\mathcal{T}_{k,b})E_b(T_k)
$$

## Calibration et validation

Les coefficients $a_b$ sont effectifs, non des constantes HITRAN ligne par ligne. Ils sont calibrés à températures fixées pour obtenir :

$$
OLR(280\ \mathrm{ppm})-OLR(560\ \mathrm{ppm})=3.93\ \mathrm{W\,m^{-2}}
$$

Les tests vérifient l’opacité nulle, la structure des dix couches, la diminution de l’OLR quand le CO₂ augmente et une réponse proche du logarithme lors de doubles successifs.

## Limites et sources

Le modèle ne représente pas la vapeur d’eau, les nuages, l’ozone, les autres gaz à effet de serre, la convection ni l’évolution thermique. Références : U.S. Standard Atmosphere 1976, IPCC AR6, Myhre et al. (1998), HITRAN et Amundsen et al. (2014).
