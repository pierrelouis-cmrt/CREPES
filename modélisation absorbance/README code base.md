# absorbance du CO2

[Source bande absorbance CO2](https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2)

# 1er modèle
mesure prise a 440 ppm
bande 14,25 à 15,75 micrometre   Absorbance moyenne: 1
bande 4,2 à 4,35 micrometre      Absorbance moyenne : 3,25


# Utilité du code

Courbe du pourcentage d'absorbance du CO2 en fonction de la longeur d'onde, a 440

# Fonctionnement du code 

- utilisation de l'API Radis
    1. Entrer la pression souhaiter, a combien de ppm on souhaite la mesure et la température.
- récupération de la transmittance aux niveaux des bandes d'absorbance (voir lien ci-dessus)
- génère un tableau de donné utilisé pour tracer la courbe de l'absorbance en fonction de la longueur d'onde.


