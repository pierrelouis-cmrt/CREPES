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


# Constantes physiques
sigma = 5.67e-8             # Constante de Stefan-Boltzmann (W/m²·K⁻⁴)
h = 3*10**-7                # Coefficient d'échange thermique (W/m²·K)
c = 4.31*10**20             # Capacité thermique surfacique (J·K)
T0 = 273                    # Température ambiante (K)
T_init = 288                # Température initiale (K)
r = 6750*10**3              # Rayon (m)
dr = 1                      # Épaisseur (m)

# Constantes regroupées
A = 4 * np.pi * h
B = 4 * np.pi * sigma * (r + dr)**2

# Domaine temporel en secondes
t_max_s = 43200  # en secondes
dt = 1
t_vals_s = np.arange(0, t_max_s, dt)
T_vals = np.zeros_like(t_vals_s, dtype=float)
T_vals[0] = T_init

# Intégration (Euler)
for i in range(1, len(t_vals_s)):
    T = T_vals[i-1]
    dTdt = -(A / c) * (T - T0) - (B / c) * T**4
    T_vals[i] = T + dTdt * dt

# Conversion en heures pour affichage
t_vals_h = t_vals_s / 3600 

# Tracé
plt.figure(figsize=(8, 5))
plt.plot(t_vals_h, T_vals, label="T(t)", linewidth=2)
plt.xlabel("Temps (heures)")
plt.ylabel("Température (K)")
plt.title("Refroidissement avec conduction + rayonnement")
plt.grid(True)
plt.legend()
plt.ylim(min(T_vals) - 1, max(T_vals) + 1)  # pour lisibilité
plt.ticklabel_format(useOffset=False)
plt.tight_layout()
plt.show()

# Affichage des valeurs extrêmes
print(f"T(0) = {T_vals[0]:.2f} K")
print(f"T(t_max = {t_max_s/3600:.1f} h) = {T_vals[-1]:.2f} K")