#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refroidissement (12 h) d’une coquille d’eau mince entourée d’atmosphère :
T(t) = T0 + (Ti − T0)·exp(−k·t)          avec  k = h A / C
Les pertes ne sont que conductives (h = 2,7 × 10⁻⁷ W m⁻² K⁻¹),
⇒ τ ≈ 1 × 10⁵ ans : variation imperceptible sur 12 h.
"""

# ───────────────────────── 1. Constantes ─────────────────────────
import numpy as np
import matplotlib.pyplot as plt

C   = 4.31e20          # J·K⁻¹  (capacité thermique globale)
h   = 2.7e-7           # W·m⁻²·K⁻¹  (conduction seule)
R_E = 6.371e6          # m       (rayon terrestre)

Ti  = 288.0            # K  (température initiale)
T0  = 273.0            # K  (température ambiante)

t_max_hours = 12.0     # h  (durée du tracé)
n_points    = 500

# ───────────────────────── 2. Constante k ─────────────────────────
A = 4 * np.pi * R_E**2
k = h * A / C

# ───────────────────────── 3. Fonction T(t) ───────────────────────
def T(t_seconds, T_inf=T0, T_init=Ti, k_val=k):
    return T_inf + (T_init - T_inf) * np.exp(-k_val * t_seconds)

# ───────────────────────── 4. Echelle de temps ────────────────────
sec_per_hour = 3600.0
t_hours   = np.linspace(0.0, t_max_hours, n_points)
t_seconds = t_hours * sec_per_hour
T_vals    = T(t_seconds)

# ───────────────────────── 5. Tracé ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_hours, T_vals, lw=2)

ax.set_xlabel("Temps (heures)")
ax.set_ylabel("Température (K)")
ax.set_title("Refroidissement sur 12 h – coquille d’eau mince (h = 2,7 × 10⁻⁷ W m⁻² K⁻¹)")
ax.grid(True)

# ✨ désactive l’offset ET la notation scientifique sur l’axe y
ax.ticklabel_format(axis='y', style='plain', useOffset=False)

# (facultatif) afficher un intervalle « visible » autour de Ti
# ax.set_ylim(Ti - 1, Ti + 1)

fig.tight_layout()
plt.show()
