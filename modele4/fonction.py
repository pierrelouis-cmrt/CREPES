"""
Résolution equa diff par la méthode d'Euler explicite (récurrente)"""

from typing import Callable, Tuple, Any
import numpy as np


def solve_ode_recurrent(rhs: Callable[[float, np.ndarray, Any], np.ndarray],
                        t0: float,
                        t_end: float,
                        dt: float,
                        y0: np.ndarray,
                        *args,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if dt <= 0:
        raise ValueError("dt doit être positif")
    if t_end <= t0:
        raise ValueError("t_end doit être > t0")

    n_steps = int(np.ceil((t_end - t0) / dt))
    t = np.linspace(t0, t0 + n_steps * dt, n_steps + 1)
    y = np.empty((n_steps + 1,) + np.shape(y0))
    y[0] = y0

    for k in range(n_steps):
        y[k + 1] = y[k] + dt * rhs(t[k], y[k], *args, **kwargs)

    return t, y
