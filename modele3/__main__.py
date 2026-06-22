"""Point d'entree pour lancer le modele 3 avec `python -m modele3` ou `python .`."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .modele3 import main
else:
    racine_depot = Path(__file__).resolve().parents[1]
    if str(racine_depot) not in sys.path:
        sys.path.insert(0, str(racine_depot))
    from modele3.modele3 import main


if __name__ == "__main__":
    main()
