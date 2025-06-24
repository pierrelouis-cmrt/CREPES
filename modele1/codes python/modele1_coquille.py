import numpy as np
import matplotlib.pyplot as plt

sigma = 5.67e-8        # W·m⁻²·K⁻⁴
R = 6.371e6            # m
C = 4.31e20            # J/K (valeur calculée)

# Condition initiale
Ti = 288               # K

# Paramètres temporels pour une simulation de 12 heures
dt = 3600              # pas de temps en secondes (1 h)
t_max = 12 * 3600      # durée de simulation : 12 h en secondes
n_steps = int(t_max / dt) + 1  # +1 pour inclure t=0

# Initialisation des tableaux
T = np.zeros(n_steps)
t = np.linspace(0, t_max, n_steps)
T[0] = Ti

# Boucle d'Euler explicite
for i in range(1, n_steps):
    dT = -(4 * np.pi * R**2 * sigma * T[i-1]**4) / C
    T[i] = T[i-1] + dT * dt

# Affichage du résultat
plt.figure(figsize=(8, 5))
plt.plot(t / 3600, T, label='Température (K)')
plt.xlabel('Temps (heures)')
plt.ylabel('Température (K)')
plt.title("Évolution de la température de la Terre sur 12 heures\n(modèle coquille-vide dans le vide)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
