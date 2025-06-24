import pandas as pd
import cartopy.io.shapereader as shapereader
from pathlib import Path
from shapely.geometry import Point
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# Nouvelle valeur RZSM à assigner UNIQUEMENT pour les zones de glace.
RZSM_FOR_ICE = 0.35
# Résolution de la grille pour l'optimisation. 1.0 est un bon compromis.
GRID_RESOLUTION_DEG = 1.0


def update_csv_with_ice_fix(
    input_csv_path: Path,
    output_csv_path: Path,
    antarctica_shp_path: Path,
    greenland_shp_path: Path,
):
    """
    Met à jour un CSV en remplaçant la valeur RZSM par une constante
    uniquement pour les points situés dans les calottes glaciaires.
    Les autres valeurs restent inchangées.
    """
    # --- Étape 1: Charger les géométries de glace ---
    print("Étape 1/5 : Chargement des géométries de glace...")
    try:
        anta_geoms = list(shapereader.Reader(str(antarctica_shp_path)).geometries())
        groe_geoms = list(shapereader.Reader(str(greenland_shp_path)).geometries())
        ice_geometries = anta_geoms + groe_geoms
    except Exception as e:
        print(f"Erreur critique : Impossible de charger les shapefiles : {e}")
        return

    # --- Étape 2: Créer la grille-masque de glace (pour la vitesse) ---
    print(f"\nÉtape 2/5 : Création d'une grille-masque de glace ({GRID_RESOLUTION_DEG}°)...")
    lon_range = np.arange(-180, 180 + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)
    lat_range = np.arange(-90, 90 + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)
    ice_mask_grid = np.zeros((len(lat_range), len(lon_range)), dtype=bool)

    for i, lat in tqdm(enumerate(lat_range), total=len(lat_range), desc="Création du masque"):
        for j, lon in enumerate(lon_range):
            if any(geom.contains(Point(lon, lat)) for geom in ice_geometries):
                ice_mask_grid[i, j] = True
    
    print("  -> Grille-masque de glace créée.")

    # --- Étape 3: Charger le fichier CSV original ---
    print(f"\nÉtape 3/5 : Chargement du fichier CSV '{input_csv_path.name}'...")
    df = pd.read_csv(input_csv_path)
    print(f"  -> Fichier chargé avec {len(df)} lignes.")

    # --- Étape 4: Appliquer le masque pour identifier les points à changer ---
    print("\nÉtape 4/5 : Identification des points de glace dans le CSV...")
    
    # Trouver les indices de la grille pour chaque point du CSV (rapide)
    lat_indices = np.clip(np.digitize(df['lat'], lat_range) - 1, 0, len(lat_range) - 1)
    lon_indices = np.clip(np.digitize(df['lon'], lon_range) - 1, 0, len(lon_range) - 1)

    # Créer un masque booléen pour le DataFrame (rapide)
    is_on_ice = ice_mask_grid[lat_indices, lon_indices]
    
    points_to_update = is_on_ice.sum()
    
    # --- LA CORRECTION CRUCIALE EST ICI ---
    # Mettre à jour la colonne RZSM SEULEMENT pour les lignes où is_on_ice est True
    if points_to_update > 0:
        print(f"  -> {points_to_update} points identifiés comme glace. Mise à jour...")
        df.loc[is_on_ice, 'RZSM'] = RZSM_FOR_ICE
    else:
        print("  -> Aucun point de glace trouvé à mettre à jour.")

    # --- Étape 5: Sauvegarder le résultat ---
    print(f"\nÉtape 5/5 : Sauvegarde du fichier corrigé...")
    df.to_csv(output_csv_path, index=False, float_format="%.6f")
    print(f"  -> Fichier sauvegardé avec succès dans : '{output_csv_path}'")


if __name__ == "__main__":
    base_path = Path(".")
    # IMPORTANT: Assurez-vous que le fichier d'entrée est bien l'original !
    input_csv = base_path / "temp" / "average_rzsm_tout.csv"
    output_csv = base_path / "temp" / "average_rzsm_tout_corrected.csv"
    antarctica_shp = base_path / "temp" / "anta.shp"
    greenland_shp = base_path / "temp" / "groe.shp"

    required_files = [input_csv, antarctica_shp, greenland_shp]
    if not all(f.exists() for f in required_files):
        print("Erreur : Un ou plusieurs fichiers d'entrée sont manquants.")
        for f in required_files:
            print(f"  - {f}: {'Trouvé' if f.exists() else 'MANQUANT'}")
    else:
        update_csv_with_ice_fix(
            input_csv, output_csv, antarctica_shp, greenland_shp
        )