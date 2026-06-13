# absorbance du CO2

[Source bande absorbance CO2](https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2)

Bande 600, 760 cm^-1
Bande 2100, 2450 cm^-1
Bande 1200, 1500 cm^-1

# Utilité du code

Courbe du pourcentage d'absorbance du CO2 en fonction de la longueur d'onde

# Fonctionnement du code

- utilisation de l'API RADIS
- récupération de la transmittance au niveau des bandes d'absorbance (voir lien ci-dessus)

## Lancement

Depuis la racine du dépôt :

```bash
./.venv/bin/python "modélisation absorbance/absorbance CO2.py"
```

Sous Windows, depuis PowerShell :

```powershell
.\.venv\Scripts\python.exe "modélisation absorbance\absorbance CO2.py"
```

Dans un environnement non interactif (CI, sandbox), ce lancement enregistre
automatiquement `modélisation absorbance/absorbance_CO2.png` au lieu d'ouvrir
une fenêtre.

Pour tester sans ouvrir de fenêtre graphique :

```bash
./.venv/bin/python "modélisation absorbance/absorbance CO2.py" --no-plot
```

Pour générer une image :

```bash
./.venv/bin/python "modélisation absorbance/absorbance CO2.py" --output "modélisation absorbance/absorbance_CO2.png" --no-plot
```

RADIS télécharge les raies HITRAN au premier lancement. Le script stocke maintenant
ce cache dans le cache utilisateur du système :

- Windows : `%LOCALAPPDATA%\CREPES\absorbance_co2\`
- macOS/Linux : `~/.cache/crepes/absorbance_co2/`

Si un téléchargement est interrompu, relancer avec :

```bash
./.venv/bin/python "modélisation absorbance/absorbance CO2.py" --regen-cache --no-plot
```
