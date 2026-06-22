# Théorie — modèle 5

## Objet du modèle

Le modèle 5 reprend le bilan de surface du modèle 4 et ajoute un échange radiatif infrarouge horizontal entre les couches des colonnes voisines. Il sert à étudier un couplage spatial simplifié, sans représenter une circulation atmosphérique complète.

## Bilan de surface

$$
C_{surface}\frac{dT_{surface}}{dt}=SW_{absorbé}+LW_{descendant}-LW_{émis}-Q_{latent}-Q_{convection}+Q_{horizontal}
$$

$Q_{horizontal}$ est évalué explicitement à chaque pas de temps ; le rayonnement sortant et la convection suivent le schéma semi-implicite du modèle 4.

## Échange entre colonnes

Pour chaque couche $k$, bande infrarouge $b$ et face commune entre les cellules $A$ et $B$ :

$$
E(k,b)=\varepsilon(k,b)B_b(T(k))
$$

L’échange net à travers l’interface est :

$$
P(A\leftarrow B)=A_{face}[E_B(k,b)-E_A(k,b)]
$$

Chaque interface est calculée une seule fois : le gain d’une cellule est exactement la perte de l’autre en watts. Les aires et longueurs de faces suivent la géométrie sphérique. Les longitudes sont périodiques sur la grille globale et les bords restent fermés sur une sous-grille.

## Convergence et surface

La convergence d’une couche est son gain net d’énergie infrarouge provenant des cellules voisines. Une convergence positive représente un gain ; une convergence négative une perte.

Seule la part de cette convergence transmise à travers les couches inférieures peut atteindre le sol. Le modèle la pondère donc par la transmission cumulée jusqu’à la surface, puis somme les couches et les bandes :

$$
Q_{horizontal}=\sum_{k,b}(convergence_{k,b}\times transmission_{k,b\rightarrow sol})
$$

## Données et limites

Les profils de pression, d’humidité et de température de référence sont mensuels et issus du paquet ERA5 du modèle 3. Les couches n’ont pas leur propre capacité thermique ou équation de mouvement.

L’échange est radiatif infrarouge uniquement. Le modèle n’inclut pas l’advection de masse, les vents horizontaux, la diffusion turbulente, l’océan dynamique, les nuages ou une prévision climatique validée.
