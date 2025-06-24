import numpy as np
import matplotlib.pyplot as plt
import subprocess


try:
    import numpy
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", numpy])
    import numpy

try:
    import matplotlib
except ImportError:
    print("OpenCV non trouvé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", matplotlib])
    import matplotlib


# Constantes
sigma = 5.67e-8  # W·m⁻²·K⁻⁴
R = 6.371e6      # m
cap_eau = 4184   # J/kg·K
masse = 1.4e21   # kg
C = cap_eau * masse

# Condition initiale
Ti = 288  # K

# Paramètres temporels
dt = 3600  # pas de temps en secondes (1h)
t_max = 365 * 24 * 3600  # durée de simulation en secondes (1 an)
n_steps = int(t_max / dt)

# Initialisation des tableaux
T = np.zeros(n_steps)
t = np.linspace(0, t_max, n_steps)
T[0] = Ti

# Boucle d'Euler explicite
for i in range(1, n_steps):
    dT = -(4 * np.pi * R**2 * sigma * T[i-1]**4) / C
    T[i] = T[i-1] + dT * dt

# Affichage du résultat
plt.figure(figsize=(10,6))
plt.plot(t / (3600*24), T, label='Température (K)')
plt.xlabel('Temps en jours')
plt.ylabel('Température en K')
plt.title("Modélisation de la température de la Terre en fonction du temps pour une boule d'eau dans le vide")
plt.grid(True)
plt.legend()
plt.show()
