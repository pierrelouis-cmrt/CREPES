# Albédo mensuel

`albedo01.csv` à `albedo12.csv` correspondent aux mois de janvier à décembre.

Ils sont régénérables via :

```bash
python3 outils_generation_donnees/generer_donnees.py --run albedo-surface-nasa --force --yes
```

Cette cible utilise `outils_generation_donnees/albedo/generer_albedo_surface.py`
et appelle NASA POWER.
