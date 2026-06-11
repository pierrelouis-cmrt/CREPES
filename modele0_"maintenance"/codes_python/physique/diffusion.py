"""Diffusion radiale du sol.

Code repris d'Ornithorynquietant, avec une seule correction structurelle:
l'etat thermique n'est plus global pour toutes les cellules, il est stocke par
couple latitude/longitude.
"""

from __future__ import annotations

import numpy as np

_T_STATE_BY_CELL = {}


def _cell_key(lat, lon):
    return (round(float(lat), 4), round(float(lon), 4))


def reset_diffusion_state(lat=None, lon=None):
    """Reinitialise l'etat de diffusion, globalement ou pour une cellule."""
    if lat is None or lon is None:
        _T_STATE_BY_CELL.clear()
    else:
        _T_STATE_BY_CELL.pop(_cell_key(lat, lon), None)


def puissance_cond(T_surf, temps, lat, long):
    """Puissance surfacique moyenne recue pendant `temps`, selon Orni."""
    node_count = 13
    depth = 10.0
    conductivity = 0.75
    diffusion_coeff = 5e-5
    deep_temperature = 288

    dx = depth / (node_count - 1)
    dt = 0.25 * dx**2 / diffusion_coeff
    steps = int(np.ceil(temps / dt))
    key = _cell_key(lat, long)

    if key not in _T_STATE_BY_CELL:
        profile = np.ones(node_count) * deep_temperature
    else:
        profile = _T_STATE_BY_CELL[key].copy()

    profile[0], profile[-1] = deep_temperature, T_surf
    flux = np.zeros(steps + 1)
    flux[0] = -conductivity * (profile[1] - profile[0]) / dx

    for step in range(1, steps + 1):
        old = profile.copy()
        profile[1:-1] = (
            old[1:-1]
            + diffusion_coeff
            * dt
            * (old[2:] - 2 * old[1:-1] + old[:-2])
            / dx**2
        )
        profile[0], profile[-1] = deep_temperature, T_surf
        flux[step] = -conductivity * (profile[1] - profile[0]) / dx

    _T_STATE_BY_CELL[key] = profile.copy()
    return flux.mean()

