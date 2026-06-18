# téléchargement des données CO2 de Copernicus (CAMS) et traitement pour obtenir un tableau CSV avec les valeurs de CO2 
# à différentes pressions (1, 10, 300, 500, 1000 hPa) sur une grille de 5° x 5°.
#⚠️ les valeurs obtenues dans le CSV sont en : kg de CO2 / kg d'air
# Pour mettre en ppm : valeur *  10^6 * (28.97 / 44.01)  (masse molaire de l'air SUPPOSEE CONSTANTE/ masse molaire du CO2)
import cdsapi
import zipfile
import os
import xarray as xr
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ⚠️⚠️⚠️POUR TELECHARGER LES DONNEES, METTRE VOTRE CLE API APRES AVOIR CREER UN COMPTE !!! ⚠️⚠️⚠️
URL_API = "https://ads.atmosphere.copernicus.eu/api"
CLE_API = "    ⚠️JUSTE ICI⚠️     " 


dossier_script = os.path.dirname(os.path.abspath(__file__))
fichier_zip = os.path.join(dossier_script, "donnees_cams.zip")
fichier_csv = os.path.join(dossier_script, "co2_monde_5deg_final.csv")


print("1. Demande des données à Copernicus (cela peut prendre quelques minutes)...")

dataset = "cams-global-greenhouse-gas-forecasts"
request = {
    "pressure_level": ["1", "10", "300", "500", "1000"], # Valeurs des 5 pression pour 50; 30; 10; 5; 0 km d'altitude
    "date": ["2025-04-15"], # date choisie arbitrairement 
    "leadtime_hour": ["0"],
    "data_format": "netcdf_zip",
    "variable": ["carbon_dioxide"]
}

try:
    
    client = cdsapi.Client(url=URL_API, key=CLE_API)
    client.retrieve(dataset, request, fichier_zip)
except Exception as e:
    print(f"\n❌ Erreur lors du téléchargement. Vérifiez votre URL et votre Clé API.\nDétail : {e}")
    exit()


print("\n2. Décompression du fichier ZIP...")
fichier_nc_extrait = None

try:
    with zipfile.ZipFile(fichier_zip, 'r') as zip_ref:
        zip_ref.extractall(dossier_script)
        # On cherche le fichier .nc fraîchement extrait
        for nom_fichier in zip_ref.namelist():
            if nom_fichier.endswith('.nc'):
                fichier_nc_extrait = os.path.join(dossier_script, nom_fichier)
                
    if not fichier_nc_extrait:
        raise FileNotFoundError("Aucun fichier .nc n'a été trouvé dans le ZIP.")
except Exception as e:
    print(f"\n❌ Erreur lors de la décompression : {e}")
    exit()


print("\n3. Traitement mathématique et filtrage de la grille (5°)...")

try:
    ds = xr.open_dataset(fichier_nc_extrait)

    # Gestion du temps
    if 'valid_time' in ds.dims:
        ds = ds.isel(valid_time=0)
    elif 'forecast_reference_time' in ds.dims:
        ds = ds.isel(forecast_reference_time=0)

    # --- CORRECTION : On réduit la taille AVANT de tout charger en mémoire ---
    # Le fichier Copernicus est en résolution 0.1°. 
    # Pour avoir 5°, on prend 1 point tous les 50 (5 / 0.1 = 50).
    # Cela divise la charge mémoire par 2500 !
    ds_subset = ds.isel(latitude=slice(None, None, 50), longitude=slice(None, None, 50))
    
    # Conversion en DataFrame (sur la version allégée)
    df = ds_subset.to_dataframe().reset_index()

    col_pression = 'pressure_level' if 'pressure_level' in df.columns else 'isobaricInhPa'
    col_co2 = 'co2' if 'co2' in df.columns else 'carbon_dioxide'

    # Création des colonnes par niveau de pression
    df_pivot = df.pivot_table(index=['latitude', 'longitude'], columns=col_pression, values=col_co2).reset_index()

    # Conversion des longitudes et arrondi
    df_pivot['longitude'] = np.where(df_pivot['longitude'] > 180, df_pivot['longitude'] - 360, df_pivot['longitude'])
    df_pivot['latitude'] = df_pivot['latitude'].round(1)
    df_pivot['longitude'] = df_pivot['longitude'].round(1)

    # Filtrage final (on garde les multiples de 5)
    df_grille = df_pivot[(df_pivot['latitude'] % 5 == 0) & (df_pivot['longitude'] % 5 == 0)].copy()

    # Renommage
    pressions_demandees = [1000.0, 500.0, 300.0, 10.0, 1.0]
    colonnes_a_renommer = {p: f'CO2_{int(p)}_hPa' for p in pressions_demandees if p in df_grille.columns}
            
    df_final = df_grille[['latitude', 'longitude'] + list(colonnes_a_renommer.keys())].rename(columns=colonnes_a_renommer)
    df_final = df_final.sort_values(by=['latitude', 'longitude'], ascending=[False, True])

    
    df_final.to_csv(fichier_csv, index=False)
    print(f"\n✅ SUCCÈS TOTAL ! Votre tableau contient {len(df_final)} points.")
    print(f"Le fichier a été créé ici : {fichier_csv}")

    # Nettoyage
    os.remove(fichier_zip)
    os.remove(fichier_nc_extrait)
    
except Exception as e:
    print(f"\n❌ Une erreur est survenue : {e}")

    
    df_final.to_csv(fichier_csv, index=False)
    print(f"\n✅ SUCCÈS TOTAL ! Votre tableau contient {len(df_final)} points géographiques.")
    print(f"Le fichier a été créé ici : {fichier_csv}")

    # Nettoyage
    os.remove(fichier_zip)
    os.remove(fichier_nc_extrait)
    print("(Les fichiers temporaires volumineux ont été effacés automatiquement).")

