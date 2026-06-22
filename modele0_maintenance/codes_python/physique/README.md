# Physique — modèle 0

Ce dossier regroupe les briques physiques utilisées par le moteur du modèle 0
ou conservées pour référence. Elles sont importées par les scripts du dossier
parent et ne constituent pas des points d'entrée indépendants.

## Structure

| Fichier | Rôle |
| --- | --- |
| `solaire.py` | Géométrie solaire et flux solaire absorbé. |
| `albedo.py` | Albédo de surface, albédo nuageux et repli de données. |
| `capacite_surface.py` | Capacité thermique de surface, notamment depuis RZSM. |
| `chaleur_latente.py` | Paramétrisation du flux latent terre/océan. |
| `convection.py` | Convection forcée et naturelle ; vent NASA POWER facultatif. |
| `diffusion.py` | Diffusion thermique radiale, conservée mais non couplée au moteur principal. |
| `co2.py` | Prototype de transfert radiatif CO₂, conservé mais non couplé. |

Pour lancer le moteur,
voir le [README des codes](../README.md).
