import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
H = 8.0          # hauteur de pression en km
zmax = 20.0      # altitude maximale en km
N = 5            # nombre de couches
tau_tot = 3.0    # épaisseur optique totale

# Fonction tau(z)
z = np.linspace(0, zmax, 500)

tau = tau_tot * (1 - np.exp(-z / H)) / (1 - np.exp(-zmax / H))

# Frontières des couches pour avoir un Delta tau constant
i = np.arange(N + 1)

z_frontieres = -H * np.log(
    1 - (i / N) * (1 - np.exp(-zmax / H))
)

tau_frontieres = tau_tot * i / N

# Affichage des valeurs
print("Découpage en couches de même Δtau :")
for k in range(N):
    print(
        f"Couche {k+1} : "
        f"{z_frontieres[k]:.2f} km à {z_frontieres[k+1]:.2f} km "
        f"--> épaisseur = {z_frontieres[k+1]-z_frontieres[k]:.2f} km"
    )

print()
print(f"Delta tau constant = {tau_tot/N:.2f}")
print(f"Emissivité de chaque couche = {1 - np.exp(-tau_tot/N):.3f}")

# Tracé de tau(z)
plt.figure(figsize=(7, 5))

plt.plot(tau, z, label=r"$\tau(z)$")

# Lignes de découpage
for k in range(N + 1):
    plt.axhline(z_frontieres[k], linestyle="--", linewidth=0.8)
    plt.axvline(tau_frontieres[k], linestyle=":", linewidth=0.8)

plt.xlabel(r"Épaisseur optique cumulée $\tau(z)$")
plt.ylabel("Altitude z (km)")
plt.title(r"Découpage de l'atmosphère en couches de même $\Delta \tau$")
plt.grid(True)
plt.legend()
plt.show()

# Tracé de l'épaisseur géométrique des couches
epaisseurs = np.diff(z_frontieres)

plt.figure(figsize=(7, 5))

plt.bar(np.arange(1, N + 1), epaisseurs)

plt.xlabel("Numéro de la couche")
plt.ylabel("Épaisseur de la couche (km)")
plt.title(r"Épaisseur géométrique des couches pour un $\Delta \tau$ constant")
plt.grid(True)
plt.show()