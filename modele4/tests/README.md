# Tests — modèle 4

Ce dossier vérifie les termes de surface, une simulation courte et le format
des sorties des deux moteurs du modèle 4.

## Lancer

Depuis la racine du dépôt :

```bash
python modele4/tests/tester_modele4.py
python modele4/tests/tester_rapide.py
```

## Structure

| Fichier | Rôle |
| --- | --- |
| `tester_modele4.py` | Tests du moteur de référence. |
| `tester_rapide.py` | Tests du moteur rapide vectorisé. |

Les moteurs testés sont décrits dans le
[README des codes](../codes_python/README.md).
