# Théorie — modèle 1

## Objet du modèle

Le modèle représente une colonne atmosphérique globale moyenne, sans latitude, sans cycle jour/nuit et sans évolution temporelle. Les températures sont fixes. Il calcule le flux infrarouge sortant vers l’espace (OLR) et le flux infrarouge descendant vers la surface.

## Colonne atmosphérique

La surface est à :

$$
T_s = 288.15\ \mathrm{K} = 15\ ^\circ\mathrm{C}
$$

Son flux infrarouge total suit Stefan-Boltzmann :

$$
F_s = \sigma T_s^4
$$

L’atmosphère est découpée en trois couches, toutes à la même température et à la même concentration en CO₂ :

| Couche | Altitude | Température | CO₂ |
| --- | --- | ---: | ---: |
| `couche_1` | 0–5 km | 253.15 K | 425.65 ppm |
| `couche_2` | 5–10 km | 253.15 K | 425.65 ppm |
| `couche_3` | 10–20 km | 253.15 K | 425.65 ppm |

La température atmosphérique de 253.15 K correspond à la température radiative effective terrestre. La concentration de CO₂ est la moyenne globale annuelle NOAA GML de 2025 retenue lors de la rédaction.

## Solaire et albédo

Le script conserve ces grandeurs comme références, sans les inclure dans un bilan thermique évolutif :

$$
S_0 = 1360\ \mathrm{W\,m^{-2}}, \qquad A = 0.30
$$

$$
F_{\mathrm{SW}} = \frac{S_0(1-A)}{4}
$$

## Bandes infrarouges du CO₂

Le spectre est séparé en deux bandes absorbantes de CO₂ et un reste transparent :

| Bande | Intervalle spectral | Absorbance effective |
| --- | ---: | ---: |
| `CO2_15um` | 14.25–15.75 µm | 1.0 |
| `CO2_4_3um` | 4.20–4.35 µm | 3.25 |

Les absorbances viennent d’une lecture des résultats du script local `modélisation absorbance/absorbance CO2.py`, fondé sur RADIS/HITRAN. Le modèle 1 ne relance pas RADIS. Les positions sont cohérentes avec les bandes usuelles à 667 cm⁻¹ (environ 15 µm) et 2349 cm⁻¹ (environ 4.26 µm), avec :

$$
\lambda(\mu\mathrm{m}) = \frac{10\,000}{\tilde{\nu}(\mathrm{cm^{-1}})}
$$

## Émission par bande

La luminance spectrale de Planck est :

$$
B_\lambda(T) = \frac{2hc^2}{\lambda^5}\frac{1}{\exp\left(\frac{hc}{\lambda k_B T}\right)-1}
$$

Le flux hémisphérique dans une bande est :

$$
E_b(T) = \int_{\lambda_1}^{\lambda_2} \pi B_\lambda(T)\,d\lambda
$$

L’intégration numérique emploie la méthode des milieux avec 2 000 pas par bande. Le facteur $\pi$ convertit la luminance en flux hémisphérique. Pour la surface, le flux hors des bandes CO₂ est :

$$
F_{\mathrm{transparent}} = \sigma T_s^4 - \sum_b E_b(T_s)
$$

Il traverse directement les trois couches.

## Absorption et propagation

L’absorbance effective $A_b$ est convertie en transmission et en émissivité :

$$
\mathcal{T}_b = \exp(-A_b), \qquad \varepsilon_b = 1 - \mathcal{T}_b
$$

L’absorbance n’est pas elle-même une fraction : elle peut être supérieure à 1. L’émissivité, elle, reste comprise entre 0 et 1.

Pour chaque bande, le flux montant part de la surface :

$$
F^\uparrow_{0,b} = E_b(T_s)
$$

Chaque couche transmet une partie du flux incident et ajoute sa propre émission :

$$
F^\uparrow_{k+1,b} = \mathcal{T}_b F^\uparrow_{k,b} + \varepsilon_b E_b(T_{\mathrm{atm}})
$$

Depuis le sommet, aucun flux infrarouge extérieur n’entre :

$$
F^\downarrow_{N,b} = 0
$$

Le flux descendant est propagé dans l’autre sens :

$$
F^\downarrow_{k,b} = \mathcal{T}_b F^\downarrow_{k+1,b} + \varepsilon_b E_b(T_{\mathrm{atm}})
$$

L’OLR est la somme du flux transparent et des deux bandes après leur propagation dans les trois couches. Le flux descendant de surface additionne les émissions descendantes dans ces deux bandes.

## Vérification limite

Si $A_b = 0$, alors :

$$
\mathcal{T}_b = 1, \qquad \varepsilon_b = 0
$$

Le résultat attendu est donc :

$$
F^\downarrow_{\mathrm{surface}} = 0, \qquad F^\uparrow_{\mathrm{TOA}} = \sigma T_s^4
$$

Cette limite est vérifiée par le script.

## Limites et sources

Le modèle ne représente pas le profil vertical réel de température, la vapeur d’eau, les nuages, les autres gaz à effet de serre, le transport horizontal ni l’ajustement radiatif de la température.

- [NASA Earth Observatory — bilan énergétique terrestre](https://science.nasa.gov/earth/earth-observatory/climate-and-earths-energy-budget/)
- [NOAA GML — tendances du CO₂ atmosphérique](https://gml.noaa.gov/ccgg/trends/)
- [ENS Lyon ACCES — spectre infrarouge du CO₂](https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2)
- [Documentation RADIS](https://radis.readthedocs.io/en/latest/)
- [Base HITRAN](https://hitran.org/)
