# Températures mensuelles

Ces douze CSV alimentent `codes_python/visualisation/affichage_3D_rapide.py`. Ils sont une vue compacte de la grille annuelle basse résolution, pas une nouvelle simulation.

## Format

Chaque fichier contient 1 800 lignes (`30 × 60` points), 24 colonnes horaires et des températures en kelvins. Les noms avec accents historiques doivent être conservés : le viewer les référence directement.

## Régénération

La source doit couvrir une année complète. La génération sélectionne la journée médiane de chaque mois puis écrit les 24 tranches horaires.

```bash
python modele0_maintenance/outils_generation_donnees/generer_donnees.py --run temperatures-12mois --force --yes
```

Voir le [README des générateurs](../../outils_generation_donnees/README.md) pour utiliser une grille source différente.
