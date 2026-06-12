import os
import zipfile
import cdsapi
import xarray as xr

print("🚀 Démarrage du téléchargeur annuel par lots (Mois par Mois)...")

# Configuration des dossiers automatiques sur votre Bureau
DOSSIER_SCRIPT = os.path.dirname(os.path.abspath(__file__))
dossier_final = os.path.join(DOSSIER_SCRIPT, "co2_reduit_2025")
dossier_temp = os.path.join(DOSSIER_SCRIPT, "temp_extraction")

os.makedirs(dossier_final, exist_ok=True)
os.makedirs(dossier_temp, exist_ok=True)

client = cdsapi.Client(
    url="https://ads.atmosphere.copernicus.eu/api",
    key="12408dd0-50dd-45af-a0c9-2a22b046579d"
)

# Calendrier des jours par mois pour l'année 2025
jours_par_mois = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}

# Boucle automatique de Janvier (1) à Décembre (12)
for mois in range(1, 13):
    nom_fichier_mois = f"co2_2025_{mois:02d}_reduit_1deg.nc"
    chemin_final_mois = os.path.join(dossier_final, nom_fichier_mois)
    
    # Sécurité : évite de retélécharger si le mois existe déjà
    if os.path.exists(chemin_final_mois):
        print(f"✅ Mois {mois:02d} déjà traité et sauvegardé. Passage au suivant.")
        continue
        
    print(f"\n📅 --- TRAITEMENT DU MOIS : 2025-{mois:02d} ---")
    
    # Création de la liste complète des jours du mois en cours
    jours_du_mois = [f"2025-{mois:02d}-{jour:02d}" for jour in range(1, jours_par_mois[mois] + 1)]
    
    chemin_zip_temp = os.path.join(DOSSIER_SCRIPT, f"temp_co2_mois_{mois:02d}.zip")
    
    request = {
        "variable": ["carbon_dioxide"],
        "pressure_level": ["1000", "925", "850", "700", "500", "300", "100"],
        "date": jours_du_mois,
        "leadtime_hour": ["0"],
        "data_format": "netcdf_zip"
    }
    
    try:
        # 1. Téléchargement du bloc mensuel brut
        print(f"⏳ En attente des serveurs Copernicus pour les {len(jours_du_mois)} jours du mois...")
        client.retrieve("cams-global-greenhouse-gas-forecasts", request, chemin_zip_temp)
        
        # Nettoyage préventif du dossier temporaire d'extraction
        for f in os.listdir(dossier_temp):
            os.remove(os.path.join(dossier_temp, f))
            
        # 2. Extraction du ZIP
        print("📦 Extraction des données brutes lourdes...")
        with zipfile.ZipFile(chemin_zip_temp, "r") as zip_ref:
            zip_ref.extractall(dossier_temp)
            
        # Trouver le fichier .nc extrait
        nc_file = [f for f in os.listdir(dossier_temp) if f.endswith(".nc")][0]
        chemin_nc_brut = os.path.join(dossier_temp, nc_file)
        
        # 3. Lecture et réduction immédiate à 1°
        print("📊 Réduction de la résolution spatiale à 1°...")
        ds_mois = xr.open_dataset(chemin_nc_brut)
        ds_mois_reduit = ds_mois.isel(latitude=slice(0, None, 10), longitude=slice(0, None, 10))
        
        # 4. Sauvegarde définitive du fichier léger mensuel
        ds_mois_reduit.to_netcdf(chemin_final_mois)
        ds_mois.close()
        
        # 5. NETTOYAGE : Suppression des fichiers de plusieurs Giga-octets devenus inutiles
        os.remove(chemin_zip_temp)
        os.remove(chemin_nc_brut)
        print(f"💾 SUCCÈS : Fichier mensuel allégé créé avec succès : {nom_fichier_mois}")
        
    except Exception as e:
        print(f"❌ Une erreur est survenue pour le mois {mois:02d} : {e}")
        print("Le script s'interrompt pour protéger vos données. Relancez-le quand vous voulez.")
        break

print("\n🎉 Processus annuel terminé ! Vos 12 fichiers mensuels légers vous attendent sur le Bureau dans le dossier 'co2_reduit_2025'.")