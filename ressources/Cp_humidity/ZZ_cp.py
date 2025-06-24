import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
from pathlib import Path
import numpy as np
from scipy.stats import binned_statistic_2d

def compute_cp_from_rzsm(
    rzsm: np.ndarray,
    rho_w: float = 1000,
    rho_b: float = 1300,
    cp_sec: float = 0.8,
    cp_water: float = 4.187,
) -> np.ndarray:
    """
    Calcule la capacité calorifique de manière vectorielle.
    Gère la valeur spéciale pour la glace (RZSM=0.9).
    """
    CP_ICE = 2.09
    # Détecter la glace en utilisant la valeur RZSM convenue (0.9)
    is_ice = np.isclose(rzsm, 0.9)

    # Calcul standard pour tous les points
    rzsm_clipped = np.clip(rzsm, 1e-6, 0.999)
    w = (rho_w * rzsm_clipped) / (rho_b * (1 - rzsm_clipped) + rho_w * rzsm_clipped)
    cp = cp_sec + w * (cp_water - cp_sec)

    # Appliquer la valeur de la glace là où c'est nécessaire
    # np.where est une manière vectorielle de faire une condition if/else
    cp = np.where(is_ice, CP_ICE, cp)
    return cp

def plot_fast_from_updated_csv(
    csv_path: Path, countries_shp_path: Path, var_name: str = "RZSM"
) -> None:
    """
    Trace une carte de la capacité calorifique très rapidement à partir d'un CSV
    pré-traité, en utilisant un griddage statistique vectorisé.
    """
    print("Étape 1/3 : Chargement et traitement des données...")
    df = pd.read_csv(csv_path)
    df["lon"] = ((df["lon"] + 180) % 360) - 180
    # Le calcul est vectorisé et rapide
    df["Cp"] = compute_cp_from_rzsm(df[var_name].values)
    print(f"  -> {len(df)} points chargés et traités.")

    print("Étape 2/3 : Griddage rapide des données...")
    # Définir les "bacs" (cellules) de notre grille de destination
    grid_res = 1.0  # Résolution de 1 degré
    lon_bins = np.arange(-180, 180 + grid_res, grid_res)
    lat_bins = np.arange(-90, 90 + grid_res, grid_res)

    # C'est l'opération clé : elle fait tout le travail de griddage en une fois.
    statistic, _, _, _ = binned_statistic_2d(
        x=df['lon'],
        y=df['lat'],
        values=df['Cp'],
        statistic='mean', # On fait la moyenne des points dans chaque cellule
        bins=[lon_bins, lat_bins]
    )
    
    # La sortie doit être transposée pour correspondre à l'attente de pcolormesh
    cp_grid = statistic.T
    print("  -> Griddage terminé.")

    print("Étape 3/3 : Affichage de la carte...")
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    # Affichage des frontières depuis le shapefile local
    ax.add_geometries(
        shapereader.Reader(str(countries_shp_path)).geometries(),
        ccrs.PlateCarree(),
        edgecolor="black", facecolor="none", linewidth=0.5
    )
    ax.gridlines(draw_labels=False)

    # pcolormesh est très efficace pour afficher des grilles
    im = ax.pcolormesh(
        lon_bins, lat_bins, cp_grid,
        shading="auto", cmap="turbo_r",
        vmin=0.8, vmax=4.2, transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", label="Capacité calorifique (kJ/kg/K)")
    plt.title("Carte de la capacité calorifique (Affichage rapide)")
    plt.show()
    print("  -> Terminé.")

if __name__ == "__main__":
    # Assurez-vous que ce script utilise bien le fichier CSV mis à jour
    updated_csv_file = Path("temp/average_rzsm_tout.csv")
    countries_shp = Path("ressources/map/ne_110m_admin_0_countries.shp")

    if not updated_csv_file.exists():
        print(f"Erreur : Le fichier de données '{updated_csv_file}' est introuvable.")
        print("Veuillez d'abord exécuter le script de mise à jour du CSV (ex: update_csv_with_ice_optimized.py).")
    elif not countries_shp.exists():
        print(f"Erreur : Le fichier de frontières '{countries_shp}' est introuvable.")
    else:
        # Installer scipy si nécessaire : pip install scipy
        plot_fast_from_updated_csv(updated_csv_file, countries_shp)