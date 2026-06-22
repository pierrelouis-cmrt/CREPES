#!/usr/bin/env python3
"""Lanceur direct du modele 3 depuis le dossier `modele3`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

RACINE_DEPOT = Path(__file__).resolve().parents[1]
DOSSIER_VENV = RACINE_DEPOT / ".venv"
PYTHON_VENV = RACINE_DEPOT / ".venv" / "bin" / "python"

if PYTHON_VENV.exists() and Path(sys.prefix).resolve() != DOSSIER_VENV.resolve():
    os.execv(
        str(PYTHON_VENV),
        [str(PYTHON_VENV), str(Path(__file__).resolve()), *sys.argv[1:]],
    )

if str(RACINE_DEPOT) not in sys.path:
    sys.path.insert(0, str(RACINE_DEPOT))

from modele3.codes_python.modele3 import main


if __name__ == "__main__":
    main()
