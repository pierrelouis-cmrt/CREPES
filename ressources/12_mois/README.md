# Températures mensuelles

CSV Ornithorynquietant utilisés par `affichage_3D_rapide.py`.

Chaque fichier contient 24 colonnes horaires pour les 1800 points de la grille
mensuelle.

Ils sont régénérables depuis la grille annuelle basse résolution actuelle :

```bash
python3 outils_generation_donnees/generer_donnees.py --run temperatures-12mois --force --yes
```
