from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Constantes
sigma = 5.67e-8       # Constante de Stefan-Boltzmann
phi0 = 1360           # Flux solaire moyen (W/m²)
A = 0.3               # Albédo
T_a = 280.0           # Température de l’atmosphère (K), constante
lam = 0.321           # Conductivité thermique
delta = 1.0           # Épaisseur du sol
h = lam / delta       # Coefficient convectif
C_terre = 2e6         # Capacité thermique surfacique du sol (J/m²/K)

# Flux solaire diurne
def phi(t):
    return phi0 * max(0, np.sin(2 * np.pi * t / 86400))

# Équation différentielle
def system(t, y):
    T_sol = y[0]
    dT_dt = (
        (1 - A) * phi(t)
        + sigma * T_a**4
        - sigma * T_sol**4
        + h * (T_a - T_sol)
    ) / C_terre
    return [dT_dt]

# Conditions initiales
T0_sol = 290.0
y0 = [T0_sol]

# Temps : 1 an
t_span = (0, 365 * 86400)
t_eval = np.linspace(*t_span, 5000)

# Résolution numérique
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Affichage
plt.figure(figsize=(10, 5))
plt.plot(sol.t / 86400, sol.y[0], label='T_sol (K)')
plt.xlabel('Temps (jours)')
plt.ylabel('Température du sol (K)')
plt.title("Évolution de la température du sol sur un an")
plt.grid()
plt.legend()
plt.show()
