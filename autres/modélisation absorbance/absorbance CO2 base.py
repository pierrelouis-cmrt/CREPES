# Modèle simplifié de l'absorption infrarouge du CO₂ autour de la longueur d’onde de 15 μm
'''# Absorption infrarouge du CO₂

Courbe de l'absorbance du CO₂ en fonction de la longueur d'onde, à 440 ppm sur 1 m.

---

# Absorbance du CO₂

## Source 

[Source bande absorbance CO2](https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2)

## Valeur en micrometre

| Bande (µm)    |
|---------------|
| 4,20 – 4,35   |
| 7,20 – 7,50   |
| 14,25 – 15,75 |


# Valeur pour le 1er modèle
Mesures prises à 440 ppm sur 1 m de colonne d'air, relevé manuellement sur le graphique :

| Bande (µm)        | Absorbance moyenne |
|-------------------|--------------------|
| 14,25 – 15,75     | 1                  |
| 4,20 – 4,35       | 3,25               |
| 7,20 – 7,50       | ≈ 0                |

# Utilité du code

Génère la courbe de l'absorbance du CO₂ en fonction de la longueur d'onde, à 440 ppm sur 1 m de colonne d'air, à partir des données spectroscopiques HITRAN.

# Fonctionnement du code

- Utilisation de l'API RADIS
    1. Paramètres : pression (1,013 atm), concentration (440 ppm), température (255 K)
    2. Calcul de l'absorbance sur 3 bandes spectrales (voir source ci-dessus)
- Récupération de l'absorbance aux niveaux des bandes d'absorption
- Interpolation et concaténation des bandes pour former une courbe continue
- Génération du graphique de l'absorbance en fonction de la longueur d'onde (4 – 17 µm)

# Installation

```bash
pip install radis numpy scipy matplotlib
```

# Utilisation

```bash
python co2_absorbance.py```'''

import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from radis import calc_spectrum

from radis.misc.warning import LinestrengthCutoffWarning




def make_cross_section_CO2_all_bands():
    warnings.filterwarnings("ignore", category=LinestrengthCutoffWarning)
    #d'après:  https://acces.ens-lyon.fr/acces/thematiques/CCCIC/ressources/irspco2
    bands = [
        (600, 760),   # bande a 667 cm⁻¹ ~ 15 µm 
        (1200,1500),  # bande à 1388 cm⁻¹ ~ 7.2 µm
        (2100, 2450),  # bande à 2349 cm⁻¹ ~ 4.3 µm
        
    ]
    all_wavelengths = []       #va permettre de stocker les longueurs d'ondes de tous les spectres
    all_absorbance = []        #va permettre de stocker les absorbances de tous les spectres
    a= 425e-6  # fraction molaire du CO2 (440 ppm)
    for wmin, wmax in bands:
        
        s = calc_spectrum(
            wmin=wmin, wmax=wmax,
            molecule="CO2", isotope="1,2,3",
            Tgas=255, pressure=1.013,
            mole_fraction=a,   # 425 ppm
            path_length=100,        # 1m = 100 cm   #epaisseur de la colonne d'air
            databank="hitran", verbose=False
        )
        longueur_onde_nm, absorbance = s.get("absorbance", wunit="nm")
        longueur_onde_um = longueur_onde_nm * 1e-3
        
        all_wavelengths.append(longueur_onde_um)
        all_absorbance.append(absorbance)
    wavelengths = np.concatenate(all_wavelengths)
    absorbances = np.concatenate(all_absorbance)
    sort_idx = np.argsort(wavelengths)    #trie les longueurs d'ondes pour garantir que l'interpolation fonctionne correctement
    return interp1d(
        wavelengths[sort_idx], absorbances[sort_idx],
        kind="linear", bounds_error=False, fill_value=0.0
    )

absorption_CO2 = make_cross_section_CO2_all_bands()

wl_um = np.linspace(4, 17, 10000)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wl_um, absorption_CO2(wl_um), color="steelblue", linewidth=0.8)
ax.set_xlabel("Longueur d'onde (µm)")
ax.set_ylabel("Absorbance ")
ax.set_title("Absorption du CO₂ (425 ppm, 1m)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
