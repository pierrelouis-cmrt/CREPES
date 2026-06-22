# Physique

Ce dossier rassemble les briques physiques utilisées ou conservées par le
moteur du modèle 0. Le point d'entrée de simulation est documenté dans le
[README parent](../README.md).

## Modules

| Module | Rôle | Statut |
| --- | --- | --- |
| `solaire.py` | Déclinaison, angle d'incidence et flux solaire absorbé. | Utilisé. |
| `albedo.py` | Albédo du sol, albédo nuageux CERES et cache NASA POWER. | Utilisé. |
| `capacite_surface.py` | Capacité thermique depuis RZSM, avec repli sec. | Utilisé. |
| `chaleur_latente.py` | Flux latent moyen par continent ou océan. | Utilisé. |
| `convection.py` | Convection forcée et naturelle, vent NASA POWER facultatif. | Utilisé. |
| `diffusion.py` | Diffusion thermique radiale dans le sol. | Conservé, non couplé. |
| `co2.py` | Prototype de transfert radiatif du CO₂. | Conservé, non couplé. |

## Conventions et replis

- Flux en `W m⁻²`, températures en kelvins et longitudes normalisées dans
  `[-180, 180]` lors de la lecture des ressources.
- Les chemins doivent venir de `../chemins.py`.
- Sans CERES, l'albédo nuageux vaut zéro ; sans RZSM, la capacité devient une
  valeur sèche constante ; sans carte des pays, le flux latent océanique est
  utilisé.
- Le prototype `co2.py` requiert aussi `radis`, absent des dépendances standard.
