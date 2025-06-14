import numpy as np
import matplotlib.pyplot as plt

# Constantes
constante_solaire = 1361  # W/m²
sigma = 5.67e-8  # W/m².K⁴

def update_sun_vector(mois, sun_vector):
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    return np.dot(rotation_matrix_saison, sun_vector)

def puissance_recue_point(lat_deg, lon_deg, mois, time, albedo=0.3):
    theta = np.radians(90 - lat_deg)  # colatitude
    phi = np.radians(lon_deg % 360)
    sun_vector = np.array([1, 0, 0])

    # Inclinaison saisonnière
    sun_vector = update_sun_vector(mois, sun_vector)

    # Rotation diurne
    angle_rotation = (time / 24) * 2 * np.pi
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation),  np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector = np.dot(rotation_matrix, sun_vector)

    # Vecteur normal du point
    normal = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    cos_incidence = max(np.dot(normal, sun_vector), 0)
    puissance_recue = constante_solaire * cos_incidence * (1 - albedo)
    temperature = (puissance_recue / sigma) ** 0.25
    return temperature, puissance_recue

# 🧪 Paramètres de simulation
lat = 49   # Paris
lon = 2
mois = 7
albedo = 0.3

# ⏱️ Simulation sur 24h
temps = np.linspace(0, 24, 200)
temperatures = []
puissances = []

for t in temps:
    temp, p = puissance_recue_point(lat, lon, mois, t, albedo)
    temperatures.append(temp)
    puissances.append(p)

# 📈 Tracé de la puissance et température au cours du temps
fig, ax1 = plt.subplots(figsize=(10, 5))

color = 'tab:red'
ax1.set_xlabel('Heure')
ax1.set_ylabel('Température reçue (K)', color=color)
ax1.plot(temps, temperatures, color=color, label="Température")
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Axe secondaire pour la puissance
color = 'tab:blue'
ax2.set_ylabel('Puissance reçue (W/m²)', color=color)
ax2.plot(temps, puissances, color=color, linestyle='--', label="Puissance")
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f"Évolution de la température et puissance solaire à Paris (juillet)")
plt.grid(True)
plt.tight_layout()
plt.show()
