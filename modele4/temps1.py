from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Paramètres physiques
sigma = 5.67e-8    # Constante de Stefan-Boltzmann
phi0 = 1360        # Flux solaire moyen (W/m²)
A = 0.3            # Albédo
T2 = 280.0         # Température du sous-sol (K)
lam = 0.321        # Conductivité thermique
delta = 1.0        # Épaisseur de sol (m)
h = lam / delta    # Coefficient convectif (W/m²/K)

# Capacité thermique (valeurs réalistes)
S = S_ = 1.0       # Surface (m²)
C_terre = 2e6      # Capacité thermique surfacique du sol (J/m²/K)
C_atm = 1e5        # Capacité thermique surfacique de l'air (J/m²/K)

# Flux solaire avec cycle jour/nuit
def phi(t):
    return phi0 * max(0, np.sin(2 * np.pi * t / 86400))

# Système différentiel
def system(t, y):
    T_sol, T_atm = y
    dTsol_dt = ((1 - A) * phi(t) + sigma * T_atm**4 - sigma * T_sol**4) / C_terre
    dTatm_dt = (sigma * T_sol**4 -  2*sigma * T_atm**4) / C_atm
    return [dTsol_dt, dTatm_dt]

# Conditions initiales
T0_sol = 293.0  
T0_atm = 273.0
y0 = [T0_sol, T0_atm]

# Temps
t_span = (0, 365*86400)  # 3 jours
t_eval = np.linspace(*t_span, 3000)

# Résolution
sol = solve_ivp(system, t_span, y0, t_eval=t_eval)

# Affichage
plt.plot(sol.t / 3600, sol.y[0], label='T_sol (K)')
plt.plot(sol.t / 3600, sol.y[1], label='T_atm (K)')
plt.xlabel('Temps (heures)')
plt.ylabel('Température (K)')
plt.legend()
plt.grid()
plt.show()
