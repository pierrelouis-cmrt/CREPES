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


# ────────────────────────────────────
# 1) PARAMÈTRES « MODULABLES »
# ────────────────────────────────────
c_eau      = 4_185.0       # J·kg⁻¹·K⁻¹    (chaleur massique de l’eau)
rho_eau    = 1_000.0       # kg·m⁻³         (masse volumique de l’eau)
h          = 2.7e-7        # W·m⁻²·K⁻¹      (coefficient d’échange – conduction)
R_T        = 6_400.0       # m              (rayon caractéristique RT)

Ti         = 288.0         # K  (+15 °C, température initiale)         # K  (température initiale)
T0         = 273.0         # K  (température ambiante)

t_max_hours = 12.0         # heures (durée totale du tracé)
n_points    = 500          # nombre de points pour tracer

# ────────────────────────────────────
# 2) CALCUL DE k (en s⁻¹)
# ────────────────────────────────────
k = (3 * h) / (c_eau * rho_eau * R_T**3)

# ────────────────────────────────────
# 3) FONCTION T(t)
# ────────────────────────────────────


def T(t_seconds, T_inf=T0, T_init=Ti, k_val=k):
    """Température en fonction du temps t_seconds (en secondes)."""
    return T_inf + (T_init - T_inf) * np.exp(-k_val * t_seconds)

# ────────────────────────────────────
# 4) CRÉATION DES VECTEURS TEMPS
# ────────────────────────────────────
secondes_par_heure = 3_600.0
t_max_s            = t_max_hours * secondes_par_heure

t_hours   = np.linspace(0.0, t_max_hours, n_points)
# Conversion en secondes pour la fonction T()
t_seconds = t_hours * secondes_par_heure

T_vals = T(t_seconds)

# ────────────────────────────────────
# 5) TRACÉ
# ────────────────────────────────────


plt.figure(figsize=(6,4))
plt.plot(t_hours, T_vals, lw=2)
plt.xlabel("Temps (heures)")
plt.ylabel("Température (K)")
plt.title("Refroidissement sur 12 heures (coefficient h = 2.7×10⁻⁷)")
plt.grid(True)
plt.tight_layout()
plt.show()
