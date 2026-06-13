import requests
import pprint

# 1. Mettez la vraie URL de votre API ici
URL_API = "https://api.votre-fournisseur.com/v1/co2_data"

# Paramètres de test pour un point
parametres = {
    "latitude": 45.0,
    "longitude": 2.0,
    "pressures": "1000,500,300,10,1"
}

print("Interrogations de l'API en cours...")

try:
    reponse = requests.get(URL_API, params=parametres, timeout=10)
    reponse.raise_for_status()
    
    print("\n--- COPIEZ LE TEXTE CI-DESSOUS ET ENVOYEZ-LE MOI ---")
    pprint.pprint(reponse.json())
    print("--------------------------------------------------")
    
except Exception as e:
    print(f"\nErreur lors de l'appel : {e}")
    if 'reponse' in locals():
        print(f"Format brut reçu du serveur : {reponse.text}")