"""Simulation ponctuelle du moteur thermique CREPES.

Base: integrateur Backward Euler du groupe Carcajous modele 4.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

import fonctions as f
import bibliotheque as lib
from physique import convection


def f_rhs(T, phinet, C, q_latent, q_convection=0.0):
    """Bilan energetique de surface sous forme dT/dt."""
    return (
        phinet
        - q_latent
        - q_convection
        + lib.P_em_atm_thermal(lib.Tatm)
        - lib.P_em_surf_thermal(T)
    ) / C


def _vent_forcee(index, config):
    """Retourne le vent utilise par la convection forcee."""
    vent = config.get("vent")
    if vent is not None:
        return vent
    vents = config.get("vents_journaliers", [2.5])
    jour = min(index // int(24 * 3600 / lib.dt), len(vents) - 1)
    return vents[jour]


def flux_convection(T_surface, index, sim_params):
    """Retourne le flux convectif, positif si la surface perd de la chaleur."""
    config = sim_params.get("convection") or {}
    mode = config.get("mode", "aucune")
    if mode == "aucune":
        return 0.0

    temperature_air = config.get("temperature_air", 288.0)
    flux = 0.0
    if mode in ("forcee", "toutes"):
        vent = _vent_forcee(index, config)
        flux += convection.convection_forced_flux(T_surface, temperature_air, vent)

    if mode in ("naturelle", "toutes"):
        flux += convection.convection_natural_flux(T_surface, temperature_air)
        return flux

    if mode == "forcee":
        return flux

    raise ValueError(f"Mode de convection inconnu: {mode}")


def backward_euler(days, T0, dt, lat_rad, lon_deg, sim_params):
    """Integre la temperature par la methode implicite Backward Euler."""
    step_count = int(days * 24 * 3600 / dt)
    T = np.empty(step_count + 1)
    T[0] = T0

    C_const = sim_params["C"]
    q_base = sim_params["q_base"]
    alb_sol_daily = sim_params["albedo_sol_daily"]
    alb_nuages_daily = sim_params["albedo_nuages_daily"]
    q_latent_smoothed = sim_params["q_latent_smoothed"]

    albedo_sol_hist = np.empty(step_count + 1)
    albedo_nuages_hist = np.empty(step_count + 1)
    C_hist = np.empty(step_count + 1)
    q_latent_hist = np.empty(step_count + 1)
    q_latent_step_hist = np.empty(step_count + 1)
    q_convection_hist = np.empty(step_count + 1)

    albedo_sol_hist[0] = alb_sol_daily[0]
    albedo_nuages_hist[0] = alb_nuages_daily[0]
    C_hist[0] = C_const
    q_latent_hist[0] = q_base
    q_latent_step_hist[0] = q_latent_smoothed[0]
    q_convection_hist[0] = flux_convection(T[0], 0, sim_params)

    for index in range(step_count):
        t_sec = index * dt
        day_of_year, heure_solaire = f.get_time_variables(t_sec, lon_deg)
        albedo_sol = alb_sol_daily[day_of_year]
        albedo_nuages = alb_nuages_daily[day_of_year]
        q_latent_step = q_latent_smoothed[index]

        phi_n = lib.P_inc_solar(
            lat_rad,
            day_of_year + 1,
            heure_solaire,
            albedo_sol,
            albedo_nuages,
        )

        X = T[index]
        for _ in range(8):
            q_conv = flux_convection(X, index, sim_params)
            F = X - T[index] - dt * f_rhs(
                X, phi_n, C_const, q_latent_step, q_conv
            )
            if sim_params.get("convection", {}).get("mode", "aucune") == "aucune":
                dF = 1.0 - dt * (-4.0 * lib.sigma * X**3 / C_const)
            else:
                eps = max(1e-4, abs(X) * 1e-6)
                q_plus = flux_convection(X + eps, index, sim_params)
                q_minus = flux_convection(X - eps, index, sim_params)
                rhs_plus = f_rhs(X + eps, phi_n, C_const, q_latent_step, q_plus)
                rhs_minus = f_rhs(X - eps, phi_n, C_const, q_latent_step, q_minus)
                dF = 1.0 - dt * ((rhs_plus - rhs_minus) / (2 * eps))
            if abs(dF) < 1e-12:
                break
            X -= F / dF
            if abs(F) < 1e-6:
                break
        T[index + 1] = X

        albedo_sol_hist[index + 1] = albedo_sol
        albedo_nuages_hist[index + 1] = albedo_nuages
        C_hist[index + 1] = C_const
        q_latent_hist[index + 1] = q_base
        q_latent_step_hist[index + 1] = q_latent_step
        q_convection_hist[index + 1] = flux_convection(T[index + 1], index, sim_params)

    return (
        T,
        albedo_sol_hist,
        albedo_nuages_hist,
        C_hist,
        q_latent_hist,
        q_latent_step_hist,
        q_convection_hist,
    )


def tracer_comparaison(
    times,
    T,
    albedo_sol_hist,
    albedo_nuages_hist,
    C_hist,
    q_latent_step_hist,
    titre,
    jour_a_afficher,
    sigma_plot=3.0,
):
    """Trace temperature, albedos, flux latent et capacite."""
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True, height_ratios=[3, 2, 2]
    )
    days_axis = times / 86400
    steps_per_day = int(24 * 3600 / lib.dt)
    start_idx = max((jour_a_afficher - 1) * steps_per_day, 0)
    end_idx = min(jour_a_afficher * steps_per_day, len(days_axis) - 1)

    axes[0].plot(days_axis, T - 273.15, lw=1.0, color="gray", alpha=0.8)
    axes[0].plot(
        days_axis[start_idx : end_idx + 1],
        T[start_idx : end_idx + 1] - 273.15,
        lw=2.5,
        color="firebrick",
        label=f"Jour ndeg {jour_a_afficher}",
    )
    axes[0].set_ylabel("Température de surface (°C)")
    axes[0].set_title(titre)
    axes[0].grid(ls=":")
    axes[0].legend()

    axes[1].plot(days_axis, albedo_sol_hist, color="tab:blue", lw=2.0, label="Sol")
    axes[1].plot(
        days_axis,
        albedo_nuages_hist,
        color="cyan",
        lw=2.0,
        ls=":",
        label="Nuages",
    )
    axes[1].set_ylabel("Albedo")
    axes[1].grid(ls=":")
    axes[1].legend()

    q_plot = f.gaussian_filter1d(q_latent_step_hist, sigma=sigma_plot, mode="wrap")
    axes[2].plot(days_axis, q_plot, color="tab:green", lw=1.5, label="Flux latent")
    axes[2].set_ylabel("Flux latent (W m⁻²)")
    axes[2].grid(ls=":")
    axes[2].legend(loc="upper left")

    ax_capacity = axes[2].twinx()
    ax_capacity.plot(days_axis, C_hist, color="tab:red", lw=2.0, ls="--")
    ax_capacity.set_ylabel("Capacité (J m⁻² K⁻¹)", color="tab:red")
    ax_capacity.tick_params(axis="y", labelcolor="tab:red")

    axes[2].set_xlabel("Jour de simulation")
    fig.tight_layout()
    plt.show()


def run_point_simulation(
    lat_sim=48.5,
    lon_sim=2.3,
    days=365 * 2,
    T_initial=288.0,
    dt=lib.dt,
    mode_convection="toutes",
    temperature_air=288.0,
    vent=2.5,
):
    """Prepare les entrees et lance la simulation ponctuelle."""
    sim_params = f.prepare_simulation_inputs(
        lat_deg=lat_sim,
        lon_deg=lon_sim,
        total_days=days,
        dt=dt,
    )
    if mode_convection != "aucune":
        config = {
            "mode": mode_convection,
            "temperature_air": temperature_air,
            "vent": vent,
        }
        if mode_convection in ("forcee", "toutes") and vent is None:
            config["vents_journaliers"] = convection.get_daily_wind_speed(
                lat_sim, lon_sim
            )
        sim_params["convection"] = config

    results = backward_euler(
        days,
        T_initial,
        dt,
        np.radians(lat_sim),
        lon_sim,
        sim_params,
    )
    times = np.arange(len(results[0])) * dt
    return {
        "times": times,
        "temperature": results[0],
        "albedo_sol": results[1],
        "albedo_nuages": results[2],
        "capacity": results[3],
        "q_latent": results[4],
        "q_latent_step": results[5],
        "q_convection": results[6],
        "sim_params": sim_params,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Simulation ponctuelle CREPES")
    parser.add_argument("--lat", type=float, default=48.5)
    parser.add_argument("--lon", type=float, default=2.3)
    parser.add_argument("--days", type=int, default=365 * 2)
    parser.add_argument("--jour-affiche", type=int, default=182)
    parser.add_argument("--no-plot", action="store_true")
    convection_group = parser.add_mutually_exclusive_group()
    convection_group.add_argument(
        "--convection",
        choices=["toutes", "aucune", "forcee", "naturelle"],
        default="toutes",
        help="Choisit les convections actives. Par defaut: toutes.",
    )
    convection_group.add_argument(
        "--sans-convection",
        action="store_true",
        help="Desactive les deux convections, equivalent a --convection aucune.",
    )
    parser.add_argument(
        "--temperature-air",
        type=float,
        default=288.0,
        help="Temperature d'air en K utilisee par la convection.",
    )
    wind_group = parser.add_mutually_exclusive_group()
    wind_group.add_argument(
        "--vent",
        type=float,
        default=2.5,
        help="Vent constant en m/s pour la convection forcee. Defaut: 2.5.",
    )
    wind_group.add_argument(
        "--vent-api",
        action="store_true",
        help="Utilise le vent journalier NASA/cache pour la convection forcee.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    mode_convection = "aucune" if args.sans_convection else args.convection
    vent = None if args.vent_api else args.vent
    print(f"Lancement simulation Lat={args.lat}, Lon={args.lon}, jours={args.days}")
    output = run_point_simulation(
        lat_sim=args.lat,
        lon_sim=args.lon,
        days=args.days,
        mode_convection=mode_convection,
        temperature_air=args.temperature_air,
        vent=vent,
    )

    steps_per_year = int(365 * 24 * 3600 / lib.dt)
    if len(output["temperature"]) > steps_per_year:
        slicer = slice(steps_per_year, None)
    else:
        slicer = slice(None)

    if not args.no_plot:
        tracer_comparaison(
            output["times"][slicer] - output["times"][slicer][0],
            output["temperature"][slicer],
            output["albedo_sol"][slicer],
            output["albedo_nuages"][slicer],
            output["capacity"][slicer],
            output["q_latent_step"][slicer],
            f"Simulation Lat={args.lat}, Lon={args.lon}",
            args.jour_affiche,
        )
    print(f"Température finale: {output['temperature'][-1] - 273.15:.2f} °C")
