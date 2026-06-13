import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
H = 8.0            # hauteur de pression en km
zmax = 80.0        # altitude maximale en km
dz = 2.0           # épaisseur des couches en km
tau_tot = 3.0      # épaisseur optique totale de l'atmosphère

# Création des couches
z_bas = np.arange(0, zmax, dz)
z_haut = z_bas + dz
z_milieu = (z_bas + z_haut) / 2

# Calcul de Delta tau pour chaque couche
delta_tau = tau_tot * (
    np.exp(-z_bas / H) - np.exp(-z_haut / H)
) / (
    1 - np.exp(-zmax / H)
)

# Calcul de l'émissivité de chaque couche
epsilon = 1 - np.exp(-delta_tau)

# Affichage des valeurs
print("Couche altitude | Delta tau | Emissivité")
for i in range(len(z_bas)):
    print(
        f"{z_bas[i]:.0f}-{z_haut[i]:.0f} km "
        f"| Δτ = {delta_tau[i]:.4f} "
        f"| ε = {epsilon[i]:.4f}"
    )

# Tracé de l'émissivité en fonction de l'altitude
plt.figure(figsize=(8, 5))

plt.plot(z_milieu, epsilon, marker="o", label="Émissivité par couche")

plt.xlabel("Altitude au milieu de la couche (km)")
plt.ylabel("Émissivité ε")
plt.title("Émissivité atmosphérique en fonction de l'altitude")
plt.grid(True)
plt.legend()

plt.show()