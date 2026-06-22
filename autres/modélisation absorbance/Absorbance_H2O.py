import numpy as np
import matplotlib.pyplot as plt


# 1. Configuration de l'axe des abscisses (Longueur d'onde de 0.1 à 30 µm)
longueurs_onde = np.linspace(0.1, 30, 30000)

# 2. Paramètres physiques (Épaisseur optique tau pure)
tau = np.zeros_like(longueurs_onde)

# 3. Base de données des bandes majeures de l'eau (Source : HITRAN)
bandes_hitran = [
    (0.72, 0.03, 0.1),    # (centre, largeur, force)
    (0.82, 0.04, 0.5),    
    (0.94, 0.05, 2.0),    
    (1.13, 0.06, 8.0),    
    (1.38, 0.08, 150.0),  
    (1.88, 0.10, 250.0),  
    (2.68, 0.20, 1500.0), 
    (3.20, 0.25, 50.0),   
    (6.27, 0.50, 2000.0), 
]

# 4. Simulation des raies de rotation-vibration
for centre, largeur, force in bandes_hitran:
    enveloppe = force * np.exp(-((longueurs_onde - centre) / largeur)**2)
    raies_fines = np.abs(np.sin(longueurs_onde * 10000 / centre)) 
    tau += enveloppe * raies_fines

# 5. Le Continuum de l'eau (Bande de rotation pure)
# Sans plancher, l'absorption dans la fenêtre (8-14 µm) tombe à ~0%
continuum = np.zeros_like(longueurs_onde)
masque_ir_lointain = longueurs_onde > 12

# Montée progressive : la saturation (100%) est atteinte doucement vers 20-25 µm
continuum[masque_ir_lointain] = ((longueurs_onde[masque_ir_lointain] - 12) ** 1.5) * 0.2
tau += continuum

# 6. APPLICATION DE LA LOI DE BEER-LAMBERT
# L'absorption chute naturellement vers 0% là où tau est proche de 0
absorption_pourcentage = (1.0 - np.exp(-tau)) * 100

# 7. Création du graphique
plt.figure(figsize=(14, 6))

plt.plot(longueurs_onde, absorption_pourcentage, color='#004aad', linewidth=0.8, alpha=0.9)
plt.fill_between(longueurs_onde, absorption_pourcentage, color='#4a90e2', alpha=0.4)

# Configuration des axes
plt.xlim(0, 30)
plt.ylim(0, 105)

plt.title("Spectre d'absorption de la vapeur d'eau (basse altitude)", 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Longueur d'onde (µm)", fontsize=12, fontweight='bold')
plt.ylabel("Absorption (%)", fontsize=12, fontweight='bold')

# Ajout du repère visuel (Plafond uniquement)
plt.axhline(100, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
#plt.text(29.5, 102, "Saturation 100%", horizontalalignment='right', fontsize=9)

# Finitions
plt.grid(axis='both', linestyle=':', alpha=0.6)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.show()
