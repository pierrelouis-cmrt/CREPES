"""
fonction.py
Intégrateur explicite (Euler avant) pour les équations différentielles ordinaires (EDO).
"""

from typing import Callable, Tuple, Any
import numpy as np


def solve_ode_recurrent(rhs: Callable[[float, np.ndarray, Any], np.ndarray],
                        t0: float,
                        t_end: float,
                        dt: float,
                        y0: np.ndarray,
                        *args,
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Intègre ``dy/dt = rhs(t, y, *args, **kwargs)`` par la méthode d’Euler explicite.

    Parameters
    ----------
    rhs : callable
        Fonction membre droite *f(t, y, ...)* qui retourne *dy/dt*.
    t0 : float
        Instant initial (s).
    t_end : float
        Instant final (s). Doit être strictement supérieur à *t0*.
    dt : float
        Pas de temps fixe (s). Pour ce modèle, 60 s est un bon compromis.
    y0 : float ou ndarray
        Condition initiale à *t0*.
    *args, **kwargs :
        Arguments supplémentaires passés à *rhs* à chaque appel.

    Returns
    -------
    t : ndarray shape (N+1,)
        Vecteur temps, incluant début et fin.
    y : ndarray shape (N+1, …)
        Solution numérique pour chaque instant de *t*.
    """
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
