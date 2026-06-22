# Théorie — modèle 2

## Objet du modèle

Le modèle 2 est une colonne atmosphérique verticale. Il remplace une correction atmosphérique globale par un calcul explicite de l’absorption et de la réémission infrarouge par couches :

```text
CO₂ → opacité infrarouge → transmission / émissivité → flux IR montants et descendants
```

Les températures sont imposées : le modèle ne prédit pas encore l’évolution temporelle du climat. Il sert à tester un noyau radiatif lisible avant l’ajout d’une dynamique thermique.

## Hypothèses

- Une seule colonne atmosphérique verticale.
- Six couches entre 0 et 80 km.
- Températures des couches imposées.
- Pression issue de l’atmosphère standard 1976.
- CO₂ constant par défaut à 420 ppm.
- Deux bandes absorbantes simplifiées pour le CO₂.
- Pas de vapeur d’eau, nuages, convection, diffusion ni échange horizontal.
- Reste du spectre infrarouge transparent.

## Couches utilisées

| Couche | Altitude | Zone | Température |
| --- | ---: | --- | ---: |
| 1 | 0–5 km | Troposphère basse | 271 K |
| 2 | 5–10 km | Troposphère moyenne | 236 K |
| 3 | 10–30 km | Tropopause | 223 K |
| 4 | 30–50 km | Stratosphère | 257 K |
| 5 | 50–65 km | Mésosphère basse | 252 K |
| 6 | 65–80 km | Mésosphère haute | 212 K |

Ce découpage reste grossier. Une étape d’amélioration consiste à vérifier la sensibilité des résultats à une grille de 8 à 10 couches.

## Moyenne de CO₂ par couche

Le profil vertical fournit altitude, pression, température et concentration de CO₂. La concentration moyenne d’une couche est pondérée par la masse d’air ; en équilibre hydrostatique, celle-ci est proportionnelle à la différence de pression :

$$
\overline{C}_k = \frac{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} C(p)\,dp}{\int_{p_{\mathrm{haut}}}^{p_{\mathrm{bas}}} dp}
$$

Le profil actuel est constant :

$$
C(p) = 420\ \mathrm{ppm}, \qquad \overline{C}_k = 420\ \mathrm{ppm}
$$

Cette méthode permet d’ajouter ultérieurement un gradient vertical de CO₂.

## Opacité infrarouge

Dans une bande $b$, la loi de Beer-Lambert est :

$$
dI_b = -\sigma_b n_{\mathrm{CO_2}} I_b\,ds
$$

Après intégration sur une couche :

$$
\frac{I_{b,\mathrm{sortie}}}{I_{b,\mathrm{entrée}}} = \exp(-\tau_b)
$$

Le modèle regroupe la complexité spectrale dans un coefficient effectif $a_b$ :

$$
\Delta\tau_{k,b} = a_b\frac{\overline{C}_k}{C_0}\frac{\Delta p_k}{p_s}
$$

avec $C_0 = 280\ \mathrm{ppm}$, $p_s = 101325\ \mathrm{Pa}$ et $\Delta p_k = p_{\mathrm{bas},k} - p_{\mathrm{haut},k}$. Dans le code :

```python
tau = a_bande * (co2_moyen_ppm / 280.0) * (delta_p / p_surface)
```

Le coefficient $a_b$ est un paramètre de modèle à calibrer ; ce n’est pas une constante fondamentale.

## Transmission et émissivité

La transmission de couche est :

$$
\mathcal{T}_{k,b} = \exp(-D\Delta\tau_{k,b})
$$

Le facteur diffusif est actuellement $D = 1$. Une valeur proche de $1,66$ pourra être testée pour représenter les trajets radiatifs obliques moyens.

Sans diffusion ni réflexion :

$$
\alpha_{k,b} = 1 - \mathcal{T}_{k,b}
$$

Par la loi de Kirchhoff :

$$
\varepsilon_{k,b} = \alpha_{k,b} = 1 - \mathcal{T}_{k,b}
$$

## Flux infrarouges

Pour une bande $b$, le flux de corps noir est :

$$
E_b(T) = \int_{\lambda_1}^{\lambda_2}\pi B_\lambda(T)\,d\lambda
$$

Le flux montant part de la surface :

$$
F^\uparrow_{0,b} = E_b(T_s)
$$

Chaque couche transforme ce flux :

$$
F^\uparrow_{k+1,b} = \mathcal{T}_{k,b}F^\uparrow_{k,b} + (1-\mathcal{T}_{k,b})E_b(T_k)
$$

Le flux descendant part du sommet :

$$
F^\downarrow_{N,b} = 0
$$

Puis :

$$
F^\downarrow_{k,b} = \mathcal{T}_{k,b}F^\downarrow_{k+1,b} + (1-\mathcal{T}_{k,b})E_b(T_k)
$$

Les sorties sont :

$$
OLR = \sum_b F^\uparrow_{N,b}, \qquad LW_{\mathrm{down,surface}} = \sum_b F^\downarrow_{0,b}
$$

Les bandes de CO₂ sont traitées explicitement et le reste du spectre est transparent.

## Validations à faire

- Si les coefficients d’opacité sont nuls : $F^\downarrow_{\mathrm{IR}}(0)=0$ et $OLR=\sigma T_s^4$.
- À températures fixées, une augmentation du CO₂ doit diminuer l’OLR.
- Les coefficients $a_b$ doivent être calibrés pour viser un forçage de 3,7 à 3,9 W m⁻² lors du doublement de 280 à 560 ppm.
- Le découpage spectral et la résolution verticale doivent être testés.

## Limites

Cette version est un noyau radiatif pédagogique non calibré scientifiquement. La vapeur d’eau, les nuages, les autres gaz à effet de serre et les bilans d’énergie évolutifs ne sont pas encore représentés.
