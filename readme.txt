modelisation.py : Fichier à exécuter. Permet de prédire la température en chaque point de la surface en fonction du mois de l'année et de l'heure de la journée

fonctions.py : Fichier contenant l'ensemble des fonctions utilisées dans Modelisation.py

construction_csv.py : Permet la création du dossier albedo et des fichier albedo{i}.csv

remplissage_csv.py : Remplissage de chaque fichier csv d'albedo

journal de bord global.pdf : Evolution global du groupe dans l'élaboration de notre modélisation

echelle_temp.py : Programme permettant de tracer l'échelle de température (il faut connaitre la valeur max et min des températures de notre carte)

échelle_temperature.png : échelle des température pour décrypter la modélisation

documentation commune.pdf : documentation détaillée du projet

/albedo : contient les fichiers de données de l'albedo

/data : contient les données pour tracer les frontières sur le globe



/!\ Il faut installer les modules "matplotlib", "numpy", "pandas", "requests", "csv", "time", "shapefile" 

Il n'est pas nécessaire d'exécuter "construction_csv.py" et "remplissage_csv.py", en effet les CSV albedo sont déjà remplis.