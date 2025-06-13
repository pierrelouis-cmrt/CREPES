import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 0.  CONSTANTES & PARAMÈTRES « propres »
# ==========================================================
S0     = 1361            # constante solaire (W m-2, TOA)
sigma  = 5.670374419e-8  # constante SB  (W m-2 K-4)

lat, lon = 49.0, 2.0     # Paris 49° N, 2° E
albedo   = 0.25          # albédo surfacique moyen
Tatm_C   = -20           # atmosphère radiative fixe (°C)
Tatm_K   = 273.15 + Tatm_C

# Capacité surfacique : 0,5 m de sol + 0,2 m d’eau ≈ 1 × 10⁸ J m-2 K-1
C_terre  = 1.0e8         # J m-2 K-1

# Intégration : 1 an, pas 30 min
dt      = 1800                      # s
Nsteps  = int(365*24*3600/dt)
time_s  = np.arange(Nsteps+1)*dt
time_d  = time_s/86400              # jours écoulés

# ==========================================================
# 1.  OUTILS SOLAIRES (géométrie cohérente)
# ==========================================================
def solar_declination(day_of_year: int) -> float:
    """Renvoie la déclinaison solaire δ (rad) – formule de Cooper."""
    return np.radians(23.44) * np.sin(2*np.pi*(284 + day_of_year)/365)

def cos_incidence(lat_deg: float, day: int, hour_local: float) -> float:
    """Cosinus de l'angle d'incidence soleil-surface plane horizontale."""
    phi_lat = np.radians(lat_deg)
    δ       = solar_declination(day)
    H       = np.radians(15*(hour_local - 12.0))  # heure → angle
    cos_i   = np.sin(phi_lat)*np.sin(δ) + np.cos(phi_lat)*np.cos(δ)*np.cos(H)
    return max(cos_i, 0.0)

# ==========================================================
# 2.  BOUCLE D’EULER EXPLICITE
# ==========================================================
T = np.empty(Nsteps+1)
T[0] = 288.0                              # 15 °C initial

for k in range(Nsteps):
    t  = time_s[k]
    day_of_year = int(t//86400) + 1
    hour_local  = (t % 86400)/3600

    # Flux solaire direct + albédo
    ci       = cos_incidence(lat, day_of_year, hour_local)
    phi_dir  = S0 * ci
    phi_net  = phi_dir * (1 - albedo)      # pas de (−α φ)

    # Équation énergétique
    dTdt = (phi_net + sigma*Tatm_K**4 - sigma*T[k]**4) / C_terre
    T[k+1] = T[k] + dTdt*dt

# ==========================================================
# 3.  TRACÉS ET CONTRÔLES
# ==========================================================
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(time_d, T-273.15)
ax.set_xlabel("Jour de l'année")
ax.set_ylabel("Température surface (°C)")
ax.set_title("Paris – Simulation 0-D sur 1 an\n(albédo 0.25, Tatm = −20 °C, pas 30 min)")
ax.grid(ls=':')
plt.tight_layout()
plt.show()

# --- Vérifications rapides ---
Tmin, Tmax = T.min()-273.15, T.max()-273.15
balance    = (sigma*Tatm_K**4 - sigma*T[-1]**4)*86400  # J m-2 d-1 le 31 déc.
print(f"👀  Plage annuelle:  T_min = {Tmin:.1f} °C   T_max = {Tmax:.1f} °C")
print(f"    Bilan IR (fin d'année) ≈ {balance/1e6:.2f} MJ m⁻² j⁻¹, proche de 0 indique équilibre.")
