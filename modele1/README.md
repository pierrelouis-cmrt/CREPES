# Modèle 1 CO₂ - colonne radiative simplifiée

`modele1.py` repars de 0 et ne se base pas pour l'instant sur le modèle 0 (provenant de l'année dernière). Le script calcule uniquement deux flux infrarouges :

- le flux ascendant sortant en haut de l'atmosphère ;
- le flux descendant reçu par la surface.

## 1. Structure du modèle

Le modèle représente une colonne atmosphérique globale moyenne, sans latitude,
sans cycle jour/nuit et sans évolution temporelle. Les températures sont fixées.

Surface : $T_s = 288.15\ \mathrm{K} = 15\ ^\circ\mathrm{C}$

Flux infrarouge total émis par la surface, avec la loi de Stefan-Boltzmann : $F_s = \sigma T_s^4$

Atmosphère : trois couches arbitraires, toutes à la même température et au même
taux de CO₂.

| Nom dans le script | Altitude basse | Altitude haute | Température |        CO₂ |
| ------------------ | -------------: | -------------: | ----------: | ---------: |
| `couche_1`         |           0 km |           5 km |    253.15 K | 425.65 ppm |
| `couche_2`         |           5 km |          10 km |    253.15 K | 425.65 ppm |
| `couche_3`         |          10 km |          20 km |    253.15 K | 425.65 ppm |

La température atmosphérique retenue est : $T_{\mathrm{atm}} = 253.15\ \mathrm{K} = -20\ ^\circ\mathrm{C}$

Elle correspond à la température radiative effective de la Terre vue depuis
l'espace, donnée par NASA Earth Observatory.

Le taux de CO₂ retenu est : $C_{\mathrm{CO2}} = 425.65\ \mathrm{ppm}$

Cette valeur vient de la moyenne globale annuelle NOAA GML pour 2025, dernier
millésime annuel complet disponible dans la série globale au moment de la
rédaction.

## 2. Solaire et albédo

Le script garde le flux solaire et l'albédo comme grandeurs identifiables :

$$
S_0 = 1360\ \mathrm{W\,m^{-2}},
\qquad
A = 0.30
$$

Flux solaire global moyen absorbé : $F_{\mathrm{SW}} = \frac{S_0(1-A)}{4}$

## 3. Bandes infrarouges du CO₂

Le spectre infrarouge est séparé en trois parties :

- une bande CO₂ autour de $15\ \mu\mathrm{m}$ ;
- une bande CO₂ autour de $4.3\ \mu\mathrm{m}$ ;
- le reste du spectre, entièrement transparent.

Bandes codées :

| Bande       |          Intervalle spectral | Absorbance moyenne |
| ----------- | ---------------------------: | -----------------: |
| `CO2_15um`  | $14.25-15.75\ \mu\mathrm{m}$ |        $A_b = 1.0$ |
| `CO2_4_3um` |   $4.20-4.35\ \mu\mathrm{m}$ |       $A_b = 3.25$ |

Les bornes et les absorbances viennent du script local
`modélisation absorbance/absorbance CO2.py`, qui calcule un spectre CO₂ avec
RADIS/HITRAN et récupère :

```text
s.get("absorbance")
```

Le modèle 1 ne relance pas RADIS : il reprend ces résultats sous forme de deux
bandes obtenues avec une lecture graphique.

Les positions spectrales sont cohérentes avec les bandes IR usuelles du CO₂. La
ressource ENS Lyon indique notamment des bandes à `667 cm^-1` et `2349 cm^-1`,
ce qui donne :

$$
\lambda(\mu\mathrm{m}) = \frac{10\,000}{\tilde{\nu}(\mathrm{cm^{-1}})}
$$

$$
667\ \mathrm{cm^{-1}} \rightarrow 14.99\ \mu\mathrm{m}
$$

$$
2349\ \mathrm{cm^{-1}} \rightarrow 4.26\ \mu\mathrm{m}
$$

## 4. Émission de Planck par bande

La surface émet d'abord un flux total $\sigma T_s^4$. Ensuite, le script
intègre la loi de Planck seulement dans les deux bandes CO₂.

Luminance spectrale :

$$
B_\lambda(T)
=
\frac{2hc^2}{\lambda^5}
\frac{1}{\exp\left(\frac{hc}{\lambda k_B T}\right)-1}
$$

Flux hémisphérique dans une bande $[\lambda_1,\lambda_2]$ :

$$
E_b(T)
=
\int_{\lambda_1}^{\lambda_2}
\pi B_\lambda(T)\,d\lambda
$$

Le facteur $\pi$ vient de l'intégration sur l'angle solide d'un hémisphère et
convertit la luminance en flux hémisphérique pour une émission diffuse de corps
noir. L'intégration numérique est faite par méthode des milieux, avec `2000` pas
par bande.

Pour $T_s = 288.15\ \mathrm{K}$ :

$$
E_{\mathrm{CO2,15\mu m}}(T_s)
= 27.4827\ \mathrm{W\,m^{-2}}
$$

$$
E_{\mathrm{CO2,4.3\mu m}}(T_s)
= 0.3331\ \mathrm{W\,m^{-2}}
$$

Le flux hors bandes CO₂ est donc :

$$
F_{\mathrm{transparent}}
=
\sigma T_s^4 - \sum_b E_b(T_s)
$$

Ce flux traverse directement les trois couches.

## 5. Absorbance, transmission et émissivité

L'absorbance $A_{b}$ RADIS est traitée comme une épaisseur optique effective. Elle peut
donc être supérieure à 1. L'émissivité, elle, doit rester entre 0 et 1.

Transmission de Beer-Lambert :

$$
\mathcal{T}_b = \exp(-A_b)
$$

Émissivité effective :

$$
\varepsilon_b = 1-\mathcal{T}_b
$$

Valeurs obtenues :

$$
A_b = 1.0
\Rightarrow
\mathcal{T}_b = 0.3679,
\qquad
\varepsilon_b = 0.6321
$$

$$
A_b = 3.25
\Rightarrow
\mathcal{T}_b = 0.0388,
\qquad
\varepsilon_b = 0.9612
$$

## 6. Propagation radiative

On note $N=3$ le nombre de couches. L'indice $k$ désigne une interface, pas
une couche :

- $k=0$ : surface ;
- $k=1$ : interface entre `couche_1` et `couche_2` ;
- $k=2$ : interface entre `couche_2` et `couche_3` ;
- $k=3=N$ : sommet de l'atmosphère.

La couche traversée vers le haut entre les interfaces $k$ et $k+1$ est donc
la couche située juste au-dessus de l'interface $k$. Pour le flux descendant,
on lit la même relation dans l'autre sens.

Pour chaque bande CO₂, le flux montant part de la surface :

$$
F^\uparrow_{0,b} = E_b(T_s)
$$

Chaque couche transmet une partie du flux incident et ajoute sa propre émission :

$$
F^\uparrow_{k+1,b}
=
\mathcal{T}_b F^\uparrow_{k,b}
+
\varepsilon_b E_b(T_{\mathrm{atm}})
$$

Vers le bas, il n'y a pas de flux infrarouge entrant depuis l'espace :

$$
F^\downarrow_{N,b} = 0
$$

Puis :

$$
F^\downarrow_{k,b}
=
\mathcal{T}_b F^\downarrow_{k+1,b}
+
\varepsilon_b E_b(T_{\mathrm{atm}})
$$

Ici, $b$ désigne une des deux bandes CO₂ :

$$
b \in \mathcal{B}
=
\{\mathrm{CO2\_15um},\mathrm{CO2\_4\_3um}\}
$$

Le flux hors bandes CO₂ est transparent :

$$
F_{\mathrm{transparent}}
=
\sigma T_s^4 - \sum_{b\in\mathcal{B}} E_b(T_s)
$$

Le flux montant total à une interface $k$ est donc :

$$
F^\uparrow_{k,\mathrm{total}}
=
F_{\mathrm{transparent}}
+
\sum_{b\in\mathcal{B}} F^\uparrow_{k,b}
$$

Le flux descendant total à une interface $k$ est :

$$
F^\downarrow_{k,\mathrm{total}}
=
\sum_{b\in\mathcal{B}} F^\downarrow_{k,b}
$$

Le bilan radiatif net à une interface intérieure, avec une couche en dessous et
une couche au-dessus, est simplement :

$$
F_{k,\mathrm{net}}
=
F^\uparrow_{k,\mathrm{total}}
-
F^\downarrow_{k,\mathrm{total}}
$$

Le flux sortant au sommet additionne :

- le flux hors bandes, transparent ;
- les flux des deux bandes après propagation dans les trois couches.

Le flux descendant à la surface additionne les émissions descendantes des deux
bandes CO₂.

## 7. Résultats

Commande :

```bash
python3 modele1/modele1.py
```

Sortie actuelle :

```text
flux_infrarouge_sortant_sommet_atmosphere_W_m2 = 380.782258
flux_infrarouge_descendant_surface_W_m2 = 16.311241
```

Le premier flux est l'OLR simplifié du modèle. Le second est le flux infrarouge
descendant reçu par la surface.

## 8. Vérification limite

Si les absorbances valent zéro :

$$
A_b = 0
$$

alors :

$$
\mathcal{T}_b = 1,
\qquad
\varepsilon_b = 0
$$

Le modèle doit donc donner :

$$
F^\downarrow_{\mathrm{surface}} = 0
$$

$$
F^\uparrow_{\mathrm{TOA}} = \sigma T_s^4
$$

Test exécuté :

```text
flux_infrarouge_sortant_sommet_atmosphere = 390.918507769 W m^-2
flux_infrarouge_descendant_surface = 0 W m^-2
```

## 9. Sources utiles

| Élément                    |                                       Valeur utilisée | Source                                                                        |
| -------------------------- | ----------------------------------------------------: | ----------------------------------------------------------------------------- |
| `TEMPERATURE_ATMOSPHERE_K` |                                  $253.15\ \mathrm{K}$ | Température radiative effective terrestre indiquée par NASA Earth Observatory |
| `CONCENTRATION_CO2_PPM`    |                                $425.65\ \mathrm{ppm}$ | Moyenne globale annuelle NOAA GML pour 2025                                   |
| `IRRADIANCE_SOLAIRE`       |                            $1360\ \mathrm{W\,m^{-2}}$ | Modèle des Carcajous callipyges de l'année dernière                           |
| `ALBEDO_SURFACE`           |                                                $0.30$ | Modèle des Carcajous callipyges de l'année dernière                           |
| Bande `CO2_15um`           |               $14.25-15.75\ \mu\mathrm{m}$, $A_b=1.0$ | `modélisation absorbance/absorbance CO2.py`, RADIS/HITRAN                     |
| Bande `CO2_4_3um`          |                $4.20-4.35\ \mu\mathrm{m}$, $A_b=3.25$ | `modélisation absorbance/absorbance CO2.py`, RADIS/HITRAN                     |
| Positions IR du CO₂        | autour de $15\ \mu\mathrm{m}$ et $4.3\ \mu\mathrm{m}$ | ENS Lyon ACCES                                                                |

Liens :

- NASA Earth Observatory, bilan énergétique terrestre : https://science.nasa.gov/earth/earth-observatory/climate-and-earths-energy-budget/
- NOAA GML, tendances du CO₂ atmosphérique : https://gml.noaa.gov/ccgg/trends/
- ENS Lyon ACCES, spectre IR du CO₂ : https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2
- RADIS documentation : https://radis.readthedocs.io/en/latest/
- HITRAN database : https://hitran.org/
