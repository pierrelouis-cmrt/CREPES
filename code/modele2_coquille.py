#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Modèle de refroidissement sur quelques jours (axe en jours).
T(t) = T0 + (Ti − T0)·exp(−k·t)
"""

# ────────────────────────────────────
# 1) PARAMÈTRES « MODULABLES »
# ────────────────────────────────────
C          = 4_31e20       # J·K⁻¹          (capacité thermique de l’eau)
h          = 3e-7          # W·m⁻²·K⁻¹      (coefficient d’échange très faible)

Ti         = 288.0         # K  (température initiale)
T0         = 273.0         # K  (température ambiante)

t_max_days = 5.0           # jours (durée totale du tracé)
n_points   = 500           # nombre de points pour tracer

# ────────────────────────────────────
# 2) CALCUL DE k (en s⁻¹)
# ────────────────────────────────────
k = (4 * np.pi * h) / C

# ────────────────────────────────────
# 3) FONCTION T(t)
# ────────────────────────────────────
import numpy as np

def T(t_seconds, T_inf=T0, T_init=Ti, k_val=k):
    """Température en fonction du temps t_seconds (en secondes)."""
    return T_inf + (T_init - T_inf) * np.exp(-k_val * t_seconds)

# ────────────────────────────────────
# 4) CRÉATION DES VECTEURS TEMPS
# ────────────────────────────────────
secondes_par_jour = 86_400.0
t_max_s          = t_max_days * secondes_par_jour

# Vecteur des temps en jours (pour l’affichage)
t_days = np.linspace(0.0, t_max_days, n_points)
# Conversion en secondes pour la fonction T()
t_seconds = t_days * secondes_par_jour

# Calcul de la température sur ces instants
T_vals = T(t_seconds)

# ────────────────────────────────────
# 5) TRACÉ
# ────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.plot(t_days, T_vals, lw=2)
plt.xlabel("Temps (jours)")
plt.ylabel("Température (K)")
plt.title("Refroidissement sur quelques jours (modèle 2)")
plt.grid(True)
plt.tight_layout()
plt.show()
