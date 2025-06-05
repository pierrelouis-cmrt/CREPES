# Import necessary library
import math

# Constantes
S = 1361  # Constante solaire en W/m^2
A = 0.3  # Albédo terrestre
sigma = 5.67e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4

def calculate_temperature(S, A, sigma):
    """
    Calcule la température d'équilibre radiatif de la Terre.

    :param S: Constante solaire en W/m^2
    :param A: Albédo terrestre
    :param sigma: Constante de Stefan-Boltzmann en W/m^2/K^4
    :return: Température d'équilibre en Kelvin
    """
    # Énergie moyenne reçue par unité de surface
    energy_received = (1 - A) * S / 4

    # Température d'équilibre
    T = (energy_received / sigma) ** 0.25
    return T

# Calcul de la température moyenne de la Terre
T_earth = calculate_temperature(S, A, sigma)
print(f"La température moyenne de la Terre est de {T_earth:.2f} K")

# Convertir en Celsius
T_earth_Celsius = T_earth - 273.15
print(f"Ce qui correspond à {T_earth_Celsius:.2f} °C")


