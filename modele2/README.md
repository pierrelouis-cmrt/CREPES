# Modele 2 - colonne CO2 a 6 couches

Le fichier `modele2.py` est une premiere version simple du modele 2. Il reprend
le noyau radiatif du modele 1, mais avec :

- 6 couches verticales ;
- une temperature differente par couche, lue sur l'image du tableau ;
- une pression basse/haute calculee avec `profil_atmosphere_co2.py` ;
- une moyenne de CO2 par couche ;
- une emissivite calculee avec une epaisseur optique effective.

Le but n'est pas encore de faire un modele complet. Le but est d'avoir un noyau
clair :

```text
CO2 moyen + epaisseur en pression -> opacite IR -> transmission/emissivite -> flux
```

## Lancement

Depuis la racine du depot :

```bash
./.venv/bin/python modele2/modele2.py
```

Le script affiche :

- les 6 couches utilisees ;
- la pression basse/haute de chaque couche ;
- le CO2 moyen par couche ;
- l'epaisseur optique, la transmission et l'emissivite par bande ;
- le flux infrarouge sortant au sommet ;
- le flux infrarouge descendant a la surface.

## Couches utilisees

Les temperatures viennent de l'image fournie.

| Couche | Altitude | Zone | Temperature |
| --- | ---: | --- | ---: |
| 1 | 0-5 km | troposphere basse | 271 K |
| 2 | 5-10 km | troposphere moyenne | 236 K |
| 3 | 10-30 km | tropopause | 223 K |
| 4 | 30-50 km | stratosphere | 257 K |
| 5 | 50-65 km | mesosphere basse | 252 K |
| 6 | 65-80 km | mesosphere haute | 212 K |

Les pressions ne sont pas lues sur l'image. Elles sont deduites avec le profil
d'atmosphere standard deja code dans `profil_atmosphere_co2.py`.

## Moyenne de CO2 par couche

Le script `profil_atmosphere_co2.py` donne un profil vertical :

```text
altitude -> pression, temperature, CO2
```

Pour une couche \(k\), on calcule une moyenne ponderee par la masse d'air. Comme
en hydrostatique la masse d'air par unite de surface est proportionnelle a
\(\Delta p\), on utilise :

$$
\overline{C}_{k}
=
\frac{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} C(p)\,dp}
{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} dp}.
$$

Dans la version actuelle, le profil CO2 est constant a \(420\ \mathrm{ppm}\).
Donc toutes les couches ont :

$$
\overline{C}_{k} = 420\ \mathrm{ppm}.
$$

Mais la methode restera valable si on ajoute ensuite un gradient vertical de
CO2.

## Demonstration de la formule d'opacite

On travaille dans une bande infrarouge \(b\), par exemple autour de
\(15\ \mu\mathrm{m}\). On suppose une couche plane-parallele, sans diffusion ni
reflexion.

Sur une petite distance \(ds\), la loi de Beer-Lambert donne :

$$
dI_b = -\sigma_b n_{\mathrm{CO_2}} I_b\,ds.
$$

Donc :

$$
\frac{dI_b}{I_b}
=
-\sigma_b n_{\mathrm{CO_2}}\,ds.
$$

En integrant sur une couche :

$$
\ln\left(\frac{I_{b,\mathrm{sortie}}}{I_{b,\mathrm{entree}}}\right)
=
-\int_{\mathrm{couche}} \sigma_b n_{\mathrm{CO_2}}\,ds.
$$

On definit la profondeur optique :

$$
\tau_b
=
\int_{\mathrm{couche}} \sigma_b n_{\mathrm{CO_2}}\,ds.
$$

Alors :

$$
\mathcal{T}_b
=
\frac{I_{b,\mathrm{sortie}}}{I_{b,\mathrm{entree}}}
=
e^{-\tau_b}.
$$

Si le trajet est vertical, \(ds=dz\). Le CO2 est suppose bien melange :

$$
n_{\mathrm{CO_2}}
=
\chi_{\mathrm{CO_2}} n_{\mathrm{air}},
$$

avec :

$$
\chi_{\mathrm{CO_2}}
=
C_{\mathrm{CO_2}}\times 10^{-6}
$$

si \(C_{\mathrm{CO_2}}\) est donne en ppm.

Donc :

$$
\tau_b
=
\sigma_b \chi_{\mathrm{CO_2}}
\int_{\mathrm{couche}} n_{\mathrm{air}}\,dz.
$$

L'equilibre hydrostatique donne :

$$
\frac{dp}{dz} = -\rho g.
$$

Donc :

$$
\rho\,dz = -\frac{dp}{g}.
$$

En integrant sur une couche :

$$
\int_{\mathrm{couche}} \rho\,dz
=
\frac{\Delta p_k}{g},
$$

avec :

$$
\Delta p_k
=
p_{\mathrm{bas},k} - p_{\mathrm{haut},k}.
$$

La quantite de CO2 dans une couche est donc proportionnelle a :

$$
C_{\mathrm{CO2}}\,\Delta p_k.
$$

On normalise avec une concentration de reference :

$$
C_0 = 280\ \mathrm{ppm},
$$

et avec la pression de surface :

$$
p_s \simeq 101325\ \mathrm{Pa}.
$$

Toute la physique spectrale de la bande est regroupee dans un coefficient
effectif \(a_b\). On obtient la parametrisation :

$$
\boxed{
\Delta\tau_{k,b}
=
a_b
\frac{\overline{C}_{k}}{C_0}
\frac{\Delta p_k}{p_s}
}
$$

Dans le script :

```python
tau = a_bande * (co2_moyen_ppm / 280.0) * (delta_p / p_surface)
```

Pour tenir compte de trajets obliques moyens, on peut ajouter un facteur
diffusif \(D\). Pour l'instant :

$$
D = 1.
$$

Donc :

$$
\boxed{
\mathcal{T}_{k,b}
=
\exp(-D\Delta\tau_{k,b})
}
$$

Dans le script :

```python
transmission = exp(-D * tau)
```

Enfin, sans diffusion ni reflexion, la fraction non transmise est absorbee :

$$
\alpha_{k,b}
=
1-\mathcal{T}_{k,b}.
$$

Par la loi de Kirchhoff, a l'equilibre thermique local :

$$
\varepsilon_{k,b}
=
\alpha_{k,b}.
$$

Donc :

$$
\boxed{
\varepsilon_{k,b}
=
1-\mathcal{T}_{k,b}
}
$$

Dans le script :

```python
emissivite = 1.0 - transmission
```

Le coefficient \(a_b\) n'est pas une constante fondamentale. C'est une opacite
effective de bande. Il faudra ensuite le recalibrer pour que le doublement du
CO2 donne un forcage radiatif proche de l'ordre de grandeur attendu.

## Spectre d'absorbance du CO2 avec RADIS

Le script `spectre_absorbance_co2.py` reste utile comme outil d'exploration et
de calibration. Il calcule un spectre d'absorbance infrarouge du CO2 avec RADIS
et les raies HITRAN. La concentration, la pression, la temperature et la longueur
du trajet optique sont parametrables.

```bash
./.venv/bin/python modele2/spectre_absorbance_co2.py --co2-ppm 800 --pressure-bar 0.8 --output modele2/spectre_800ppm.png --csv modele2/spectre_800ppm.csv --no-plot
```

Options principales :

- `--co2-ppm` : concentration volumique en ppm ;
- `--pressure-bar` : pression totale en bar ;
- `--temperature-k` : temperature en kelvins ;
- `--path-length-m` : trajet optique en metres ;
- `--output` : fichier image produit ;
- `--csv` : export des valeurs numeriques.

L'absorbance est definie par :

$$
A = -\ln(\mathcal{T}).
$$

## Profil vertical de l'atmosphere

Le script `profil_atmosphere_co2.py` calcule, en fonction de l'altitude, la
pression atmospherique, le rapport de melange du CO2 en ppm, sa pression
partielle et sa concentration en molecules par metre cube.

```bash
./.venv/bin/python modele2/profil_atmosphere_co2.py --max-altitude-km 50 --surface-co2-ppm 420 --output modele2/profil_atmosphere_co2.png --csv modele2/profil_atmosphere_co2.csv --no-plot
```

Par defaut, le rapport de melange reste constant a 420 ppm. Une variation
lineaire peut etre testee avec `--co2-gradient-ppm-per-km`. Par exemple,
`--co2-gradient-ppm-per-km -0.2` retire 0,2 ppm par kilometre.
