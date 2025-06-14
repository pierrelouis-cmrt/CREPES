import xarray as xr

# Charger les données NetCDF
ds = xr.open_dataset(r"C:\Users\chloe\Documents\CapECL\CREPE\Modélisation\CERES_EBAF-TOA_Ed4.2_Subset.nc")

# Variables disponibles dans ton fichier
toa_sw_all = ds['toa_sw_all_mon']        # Réflexion all-sky
toa_sw_clr = ds['toa_sw_clr_c_mon']      # Réflexion clear-sky
solar_incident = ds['solar_mon']         # Flux solaire incident TOA

# Calcul de l'albédo dû aux nuages (approximation TOA)
cloud_albedo = (toa_sw_all - toa_sw_clr) / solar_incident
cloud_albedo = cloud_albedo.clip(min=0)

# Extraction à 45°N, 5°E en janvier 2020
cloud_jan2020 = cloud_albedo.sel(time='2024-01', lat=9, lon=8.6, method='nearest').values

print(f"Approx. Albédo des nuages (janv 2025, 45°N 5°E) : {cloud_jan2020:.3f}")
