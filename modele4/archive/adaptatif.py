#!/usr/bin/env python3
# ----------------------------------------------------------------
# Solveur moderne Scipy : solve_ivp (RK45 adaptatif)
# Trace la solution sur 1 jour puis sur 1 an.
# ----------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from scipy.integrate import solve_ivp

# ---------- constantes & paramètres ----------
S0, sigma = 1361.0, 5.670374419e-8
alpha, Tatm = 0.25, 253.15          # albédo / Tatm (-20 °C)
Csurf = 1.0e8                       # J m-2 K-1
lat = np.radians(49.0)              # Paris

# ---------- géométrie solaire ----------
def declination(d):
    return np.radians(23.44)*np.sin(2*pi*(284+d)/365)

def cos_incidence(d, h):
    δ = declination(d)
    H = np.radians(15*(h-12))
    return max(np.sin(lat)*np.sin(δ)+np.cos(lat)*np.cos(δ)*np.cos(H), 0.0)

def phi_net(d, h):
    return S0 * cos_incidence(d, h) * (1-alpha)

# ---------- RHS pour solve_ivp ----------
def dTdt(t, T):
    d  = int(t//86400) + 1
    h  = (t % 86400)/3600
    Φ  = phi_net(d, h)
    return (Φ + sigma*Tatm**4 - sigma*T**4) / Csurf

def solve_rk45(days, T0=288.0):
    t0, tf = 0.0, days*24*3600
    t_eval = np.arange(t0, tf+1800, 1800)          # enregistre chaque 30 min
    sol = solve_ivp(dTdt, (t0, tf), [T0],
                    method="RK45", rtol=1e-6, atol=1e-8, t_eval=t_eval)
    return sol.t, sol.y[0]

def tracer(t, T, titre):
    plt.figure(figsize=(10,4))
    plt.plot(t/86400, T-273.15, lw=1.2)
    plt.xlabel("Jour")
    plt.ylabel("Température surface (°C)")
    plt.title(titre)
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for jours in (1, 365):
        t, T = solve_rk45(jours)
        tracer(t, T, f"RK45 (adaptatif) – {jours} jour{'s' if jours>1 else ''}")