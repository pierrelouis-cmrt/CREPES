# ---------- RHS de l’EDO (via fonctions de flux) ----------

def f_rhs_mod(T, lat, lon, t, albedo, C):
    # Flux incident solaire
    Pinc = P_inc_solar(lat, lon, t)
    
    # Flux absorbé surface (avec albédo variable)
    Pabs_surf = P_abs_surf_solar(lat, lon, t, Pinc * (1 - albedo))
    
    # Flux émis thermique surface
    Pem_surf_th = P_em_surf_thermal(lat, lon, t, T)
    
    # Flux convectif et évaporatif constants
    Pem_conv = P_em_surf_conv(lat, lon, t)
    Pem_evap = P_em_surf_evap(lat, lon, t)
    
    # Flux net
    Pnet = Pabs_surf + P_abs_atm_thermal(lat, lon, t, T) + P_em_atm_thermal_down(lat, lon, t) \
        - Pem_surf_th - Pem_conv - Pem_evap
    
    return Pnet / C


# ---------- intégrateur Backward‑Euler modifié ----------

def backward_euler_mod(days, lat_deg=49.0, lon_deg=2.3, T0=288.0):
    """
    Intègre la température de surface avec les fonctions de flux.
    """
    N = int(days * 24 * 3600 / dt)
    times = np.arange(N + 1) * dt
    T = np.empty(N + 1)
    albedo_hist = np.empty(N + 1)
    C_hist = np.empty(N + 1)

    T[0] = T0
    lat_rad = np.radians(lat_deg)
    lat_idx = _lat_idx(lat_deg)
    lon_idx = _lon_idx(lon_deg)

    # --- Initialisation (k=0) ---
    jour_init = 1
    mois_init = 1
    albedo_hist[0] = monthly_albedo[mois_init - 1, lat_idx, lon_idx]
    c_massique_j_init = capacite_thermique_massique(albedo_hist[0]) * 1000.0
    C_hist[0] = c_massique_j_init * MASSE_SURFACIQUE_ACTIVE

    for k in range(N):
        t_sec = k * dt
        jour = int(t_sec // 86400) + 1
        jour_dans_annee = (jour - 1) % 365
        mois = int(jour_dans_annee / 30.4) + 1
        mois = min(max(mois, 1), 12)

        albedo = monthly_albedo[mois - 1, lat_idx, lon_idx]
        c_massique_j = capacite_thermique_massique(albedo) * 1000.0
        C = c_massique_j * MASSE_SURFACIQUE_ACTIVE

        albedo_hist[k + 1] = albedo
        C_hist[k + 1] = C

        # Newton-Raphson pour Backward Euler
        X = T[k]
        for _ in range(8):
            F = X - T[k] - dt * f_rhs_mod(X, lat_rad, np.radians(lon_deg), t_sec, albedo, C)
            dF = 1.0 - dt * (-4.0 * SIGMA * X**3 / C)  # dérivée de P_em_surf_thermal par rapport à T
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[k + 1] = X

    return times, T, albedo_hist, C_hist
