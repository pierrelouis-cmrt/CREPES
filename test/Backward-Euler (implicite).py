# ---------------------------------------------------------------
# Backward-Euler implicite pour l’équation 0-D de température
# Trace la courbe sur 1 jour (le 1er janvier) puis sur 1 an
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi

# ---------- constantes & paramètres physiques ----------
S0    = 1361.0                     # constante solaire (W m-2)
sigma = 5.670374419e-8             # Stefan-Boltzmann
alpha = 0.25                       # albédo de surface
Tatm  = 253.15                     # atmosphère radiative (-20 °C)
C     = 8.36e5                     # capacité surfacique (J m-2 K-1)
lat   = np.radians(49.0)           # Paris 49° N

dt    = 1800.0                     # pas (30 min)

# ---------- outils solaires ----------
def declination(day):
    return np.radians(23.44) * np.sin(2*pi*(284+day)/365)

def cos_incidence(day, hour):
    δ = declination(day)
    H = np.radians(15*(hour-12))
    ci = np.sin(lat)*np.sin(δ) + np.cos(lat)*np.cos(δ)*np.cos(H)
    return max(ci, 0.0)

def phi_net(day, hour):
    return S0 * cos_incidence(day, hour) * (1-alpha)

def f_rhs(T, phinet):
    return (phinet + sigma*Tatm**4 - sigma*T**4) / C

# ---------- intégrateur Backward-Euler (Newton) ----------
def backward_euler(days, T0=288.0):
    N = int(days*24*3600/dt)
    times = np.arange(N+1)*dt
    T = np.empty(N+1);  T[0] = T0
    for k in range(N):
        t_sec   = k*dt
        day     = int(t_sec//86400) + 1
        hour    = (t_sec % 86400)/3600
        phi_n   = phi_net(day, hour)

        # Newton pour résoudre F(X) = X - T[k] - dt*f(X) = 0
        X = T[k]
        for _ in range(8):
            F  = X - T[k] - dt * f_rhs(X, phi_n)
            dF = 1 - dt * (-4*sigma*X**3 / C)
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k+1] = X
    return times, T

def tracer(times, T, titre):
    plt.figure(figsize=(10,4))
    plt.plot(times/86400, T-273.15, lw=1.2)
    plt.xlabel("Jour")
    plt.ylabel("Température surface (°C)")
    plt.title(titre)
    plt.grid(ls=":")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    for jours in (1, 365):
        t, T = backward_euler(jours)
        tracer(t, T, f"Backward-Euler – {jours} jour{'s' if jours>1 else ''}")
