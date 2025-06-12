import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────
# CONSTANTES
# ────────────────────────────────────
sigma = 5.67e-8        # W·m⁻²·K⁻⁴
R = 6.371e6            # m
C = 4.31e20            # J/K (valeur calculée pour la chaleur thermique totale)

# ────────────────────────────────────
# CONDITIONS INITIALES
# ────────────────────────────────────
Ti = 288.0             # K (température initiale)

# ────────────────────────────────────
# PARAMÈTRES TEMPORELS (12 heures)
# ────────────────────────────────────
t_max_hours = 12.0               # durée totale en heures
dt_seconds = 3600.0              # pas de temps en secondes (1 h)
t_max_seconds = t_max_hours * 3600.0
n_steps = int(t_max_seconds / dt_seconds) + 1  # nombre de points (on ajoute 1 pour inclure l'instant t=0)

# ────────────────────────────────────
# INITIALISATION DES TABLEAUX
# ────────────────────────────────────
T = np.zeros(n_steps)
t_seconds = np.linspace(0.0, t_max_seconds, n_steps)
t_hours = t_seconds / 3600.0     # vecteur temps en heures (pour l’affichage)

T[0] = Ti

# ────────────────────────────────────
# BOUCLE D'EULER EXPLICITE
# ────────────────────────────────────
for i in range(1, n_steps):
    # dT/dt = - (Puissance rayonnée) / C, où Puissance = 4πR²σT⁴
    dT_dt = - (4 * np.pi * R**2 * sigma * T[i-1]**4) / C
    T[i] = T[i-1] + dT_dt * dt_seconds

# ────────────────────────────────────
# TRACÉ
# ────────────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(t_hours, T, lw=2)
plt.xlabel('Temps (heures)')
plt.ylabel('Température (K)')
plt.title("Évolution de la température de la Terre (coquille vide) sur 0–12 h")
plt.xlim(0, t_max_hours)
plt.grid(True)
plt.tight_layout()
plt.show()
