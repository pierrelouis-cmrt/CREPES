# Modèle simplifié de l'absorption infrarouge du CO₂ autour de la longueur d’onde de 15 μm
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
    for wmin, wmax in bands:
        
        s = calc_spectrum(
            wmin=wmin, wmax=wmax,
            molecule="CO2", isotope="1,2,3",
            Tgas=255, pressure=1.013,
            mole_fraction=440e-6,   # 440 ppm
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
ax.set_title("Absorption du CO₂ (400 ppm, 1m)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
