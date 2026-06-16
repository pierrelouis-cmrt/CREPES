# Physique du modèle 3

Ce dossier regroupe les fonctions de calcul physique élémentaire utilisées par
`modele3.py`.

Il ne contient pas la logique complète de la colonne : pas de lecture de
données, pas de CLI, pas de boucle principale. Le fichier principal construit
les couches, propage les flux et prépare les sorties.

`calculs.py` contient notamment :

- géométrie solaire ;
- moyenne verticale en pression ;
- masse d'air et masse de vapeur d'eau par couche ;
- loi de Planck et flux de corps noir par bande ;
- opacités CO2, H2O et nuages ;
- albédo nuage effectif ;
- flux court-onde absorbé à la surface.
