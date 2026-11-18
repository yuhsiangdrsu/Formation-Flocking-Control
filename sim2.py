"""
generate_flocking_data.py

Runs two Monte Carlo studies and saves all data needed for later plotting.

Studies:
1) Sweep R_values = [1.0, 1.5, 2.0, 2.5, 3.0] with fixed sigma = 0.5
2) Sweep sigma_values = [0.5, 1.0, 5.0, 10.0] with fixed R = 2.0

Also stores one full diagnostic example for:
    R = 2.0, sigma = 5.0

Output:
    results/flocking_sweep_data.npz
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_continuous_are


# ============================================================
# Global simulation parameters
# ============================================================
N_f = 20
N = N_f + 1
leader_idx = 0
followers = np.arange(1, N)

T = 20.0
dt = 0.01
steps = int(T / dt)
time = np.arange(steps) * dt

n_trials = 50
n_pinned = 10
pin_update_every = 1

# Leader constant velocity
p_L0 = np.array([0.0, 0.0])
v_L_const = np.array([0.2, 0.1])

# Required parameter sweeps
R_values = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
sigma_values = np.array([0.5, 1.0, 5.0, 10.0], dtype=float)

fixed_sigma_for_R_sweep = 5
fixed_R_for_sigma_sweep = 2.0

# Example case for trajectory plot
example_R = 2.0
example_sigma = 5.0
example_seed = 1000

# Flocking-time thresholds
eps_v = 0.15
eps_c = 3.0
dwell_time = 2.0

# Collision avoidance parameters
d_min = 0.5
d_act = 2.0
k_phi = 10.0
eps_phi = 1e-6

# Acceleration saturation
a_max = 6.0

# Initial condition region
init_low = -7.5
init_high = 7.5
initial_min_distance = 1.2

# Weight normalization
normalize_follower_weights = True
eps_w = 1e-8

# Output location
output_dir = Path("results")
output_file = output_dir / "flocking_sweep_data_2.npz"


# ============================================================
# LQR follower controller design
# ============================================================
A = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
)

B = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)

Q_follower = np.diag([10.0, 10.0, 50.0, 50.0])
R_follower_lqr = np.diag([1.0, 1.0])


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


K_follower, P_follower = lqr_gain(A, B, Q_follower, R_follower_lqr)


# ============================================================
# Utility functions
# ============================================================
def sat_vec(vec: np.ndarray, vmax: float) -> np.ndarray:
    """Saturate a 2D vector by Euclidean norm."""
    n = np.linalg.norm(vec)
    if n <= vmax or n < 1e-12:
        return vec
    return (vmax / n) * vec


def initialize_positions_with_min_distance(
    n_points: int,
    low: float,
    high: float,
    min_distance: float,
    rng: np.random.Generator,
    max_tries: int = 100000,
) -> np.ndarray:
    """Sample points uniformly in [low, high]^2 with pairwise distance above min_distance."""
    p = np.empty((n_points, 2), dtype=float)

    for i in range(n_points):
        placed = False
        for _ in range(max_tries):
            candidate = rng.uniform(low, high, size=2)
            if i == 0:
                p[i] = candidate
                placed = True
                break

            dists = np.linalg.norm(p[:i] - candidate, axis=1)
            if np.all(dists > min_distance):
                p[i] = candidate
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place point {i}. Increase the initial region "
                f"or reduce min_distance={min_distance}."
            )

    return p


def build_follower_neighbors(p: np.ndarray, followers: np.ndarray, R_interaction: float) -> dict[int, list[int]]:
    """Radius-based follower graph."""
    follower_neighbors = {int(i): [] for i in followers}

    for i in followers:
        neigh_i = []
        for j in followers:
            if i == j:
                continue
            dist_ij = np.linalg.norm(p[i] - p[j])
            if dist_ij <= R_interaction:
                neigh_i.append(int(j))
        follower_neighbors[int(i)] = neigh_i

    return follower_neighbors


def compute_potential_avoidance(
    i: int,
    p: np.ndarray,
    alive: np.ndarray,
    d_min: float,
    d_act: float,
    k_phi: float,
    eps_phi: float = 1e-6,
) -> np.ndarray:
    """Repulsive potential-based collision avoidance."""
    u_avoid = np.zeros(2)
    c_act = 1.0 / max(d_act - d_min, eps_phi)

    for j in range(len(p)):
        if j == i or not alive[j]:
            continue

        z = p[i] - p[j]
        r = np.linalg.norm(z)

        if r >= d_act or r < 1e-12:
            continue

        gap = max(r - d_min, eps_phi)
        z_hat = z / r

        s = (1.0 / gap) - c_act
        f_mag = k_phi * s / (gap**2)
        u_avoid += f_mag * z_hat

    return u_avoid


def select_adaptive_pinned_followers(
    p: np.ndarray,
    v: np.ndarray,
    p_L: np.ndarray,
    v_L: np.ndarray,
    followers: np.ndarray,
    n_pinned: int,
) -> tuple[np.ndarray, list[int]]:
    """
    Select exactly n_pinned followers with largest leader-tracking position error.

    To include velocity error too, replace:
        score = pos_err
    with:
        score = pos_err + vel_err
    """
    scores = []

    for i in followers:
        pos_err = np.linalg.norm(p[i] - p_L)
        vel_err = np.linalg.norm(v[i] - v_L)
        score = pos_err
        scores.append((score, int(i)))

    scores.sort(key=lambda x: x[0], reverse=True)
    selected = [i for _, i in scores[:n_pinned]]

    c = np.zeros(len(p))
    for i in selected:
        c[i] = 5.0

    return c, selected


def compute_follower_control(
    i: int,
    p: np.ndarray,
    v: np.ndarray,
    p_L: np.ndarray,
    v_L: np.ndarray,
    follower_neighbors: dict[int, list[int]],
    c: np.ndarray,
    sigma_f: float,
    alive: np.ndarray,
) -> np.ndarray:
    """Follower controller."""
    x_i = np.hstack((p[i], v[i]))
    x_L = np.hstack((p_L, v_L))

    e_pin = c[i] * (x_i - x_L)
    e_graph = np.zeros(4)

    neigh_i = [j for j in follower_neighbors.get(int(i), []) if alive[j]]

    if len(neigh_i) > 0:
        weights = []
        for j in neigh_i:
            dist_ij = np.linalg.norm(p[i] - p[j])
            a_ij = np.exp(-(dist_ij**2) / (sigma_f**2))
            weights.append(a_ij)

        weights = np.array(weights, dtype=float)

        if normalize_follower_weights:
            weights = weights / (np.sum(weights) + eps_w)

        for w_ij, j in zip(weights, neigh_i):
            x_j = np.hstack((p[j], v[j]))
            e_graph += w_ij * (x_i - x_j)

    u_avoid = compute_potential_avoidance(
        i=i,
        p=p,
        alive=alive,
        d_min=d_min,
        d_act=d_act,
        k_phi=k_phi,
        eps_phi=eps_phi,
    )

    u_i = -K_follower @ (e_pin + e_graph) + u_avoid
    return sat_vec(u_i, a_max)


# ============================================================
# Performance metric functions
# ============================================================
def compute_velocity_error(v_hist: np.ndarray, followers: np.ndarray, leader_idx: int) -> np.ndarray:
    """Mean follower-leader velocity error over time."""
    v_L = v_hist[:, leader_idx, :]
    follower_errors = np.linalg.norm(v_hist[:, followers, :] - v_L[:, None, :], axis=2)
    return np.mean(follower_errors, axis=1)


def compute_cohesion_radius(p_hist: np.ndarray, followers: np.ndarray) -> np.ndarray:
    """Follower cohesion radius over time."""
    pf = p_hist[:, followers, :]
    centroid = np.mean(pf, axis=1)
    return np.max(np.linalg.norm(pf - centroid[:, None, :], axis=2), axis=1)


def compute_min_inter_agent_distance(p_hist: np.ndarray) -> np.ndarray:
    """Minimum inter-agent distance at each time step."""
    n_steps, n_agents, _ = p_hist.shape
    d_min_time = np.empty(n_steps)

    for s in range(n_steps):
        min_dist = np.inf
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dij = np.linalg.norm(p_hist[s, i] - p_hist[s, j])
                if dij < min_dist:
                    min_dist = dij
        d_min_time[s] = min_dist

    return d_min_time


def compute_flocking_time_with_dwell(
    p_hist: np.ndarray,
    v_hist: np.ndarray,
    followers: np.ndarray,
    leader_idx: int,
    time: np.ndarray,
    eps_v: float,
    eps_c: float,
    dwell_time: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Flocking settling time with continuous dwell-time condition."""
    dt_local = time[1] - time[0]
    dwell_steps = int(np.ceil(dwell_time / dt_local))

    E_v = compute_velocity_error(v_hist, followers, leader_idx)
    R_c = compute_cohesion_radius(p_hist, followers)
    condition = (E_v <= eps_v) & (R_c <= eps_c)

    T_flock = np.nan
    for s in range(len(time) - dwell_steps):
        if np.all(condition[s : s + dwell_steps]):
            T_flock = time[s]
            break

    return T_flock, E_v, R_c


def summarize_last_window(y: np.ndarray, window_seconds: float = 2.0) -> float:
    """Mean value over the final window_seconds of a time series."""
    window_steps = max(1, int(round(window_seconds / dt)))
    return float(np.mean(y[-window_steps:]))


# ============================================================
# Single simulation run
# ============================================================
def run_one_trial(
    R_interaction: float,
    sigma_f: float,
    seed: int,
    store_diagnostics: bool = False,
) -> dict:
    """Run one Monte Carlo trial."""
    rng = np.random.default_rng(seed)

    p = np.zeros((N, 2))
    v = np.zeros((N, 2))

    p[leader_idx] = p_L0
    v[leader_idx] = v_L_const

    p_followers = initialize_positions_with_min_distance(
        n_points=N_f,
        low=init_low,
        high=init_high,
        min_distance=initial_min_distance,
        rng=rng,
    )

    p[followers] = p_followers
    v[followers] = rng.uniform(-0.5, 0.5, size=(N_f, 2))

    alive = np.ones(N, dtype=bool)

    p_hist = np.zeros((steps, N, 2), dtype=np.float32)
    v_hist = np.zeros((steps, N, 2), dtype=np.float32)
    u_hist = np.zeros((steps, N, 2), dtype=np.float32)
    pinned_hist = np.zeros((steps, N), dtype=np.float32)
    avg_degree_hist = np.zeros(steps, dtype=np.float32)

    c = np.zeros(N)

    for s, t in enumerate(time):
        p_L = p_L0 + v_L_const * t
        v_L = v_L_const

        p[leader_idx] = p_L
        v[leader_idx] = v_L

        follower_neighbors = build_follower_neighbors(
            p=p,
            followers=followers,
            R_interaction=R_interaction,
        )

        avg_degree_hist[s] = np.mean([len(follower_neighbors[int(i)]) for i in followers])

        if s == 0 or s % pin_update_every == 0:
            c, _ = select_adaptive_pinned_followers(
                p=p,
                v=v,
                p_L=p_L,
                v_L=v_L,
                followers=followers,
                n_pinned=n_pinned,
            )

        u = np.zeros((N, 2))

        for i in followers:
            u[i] = compute_follower_control(
                i=int(i),
                p=p,
                v=v,
                p_L=p_L,
                v_L=v_L,
                follower_neighbors=follower_neighbors,
                c=c,
                sigma_f=sigma_f,
                alive=alive,
            )

        v[followers] = v[followers] + dt * u[followers]
        p[followers] = p[followers] + dt * v[followers]

        p_hist[s] = p
        v_hist[s] = v
        u_hist[s] = u
        pinned_hist[s] = c

    T_flock, E_v, R_c = compute_flocking_time_with_dwell(
        p_hist=p_hist,
        v_hist=v_hist,
        followers=followers,
        leader_idx=leader_idx,
        time=time,
        eps_v=eps_v,
        eps_c=eps_c,
        dwell_time=dwell_time,
    )

    d_min_time = compute_min_inter_agent_distance(p_hist)

    output = {
        "T_flock": float(T_flock),
        "E_v": E_v.astype(np.float32),
        "R_c": R_c.astype(np.float32),
        "d_min_time": d_min_time.astype(np.float32),
        "E_v_final_window": summarize_last_window(E_v),
        "R_c_final_window": summarize_last_window(R_c),
        "d_min_over_time": float(np.min(d_min_time)),
        "avg_degree_final_window": summarize_last_window(avg_degree_hist),
    }

    if store_diagnostics:
        output.update(
            {
                "p_hist": p_hist,
                "v_hist": v_hist,
                "u_hist": u_hist,
                "pinned_hist": pinned_hist,
                "avg_degree_hist": avg_degree_hist,
            }
        )

    return output


# ============================================================
# Monte Carlo sweep helpers
# ============================================================
def empty_sweep_storage(param_values: np.ndarray) -> dict:
    shape_scalar = (len(param_values), n_trials)
    shape_time = (len(param_values), n_trials, steps)

    return {
        "T_flock": np.full(shape_scalar, np.nan, dtype=np.float32),
        "E_v_final_window": np.full(shape_scalar, np.nan, dtype=np.float32),
        "R_c_final_window": np.full(shape_scalar, np.nan, dtype=np.float32),
        "d_min_over_time": np.full(shape_scalar, np.nan, dtype=np.float32),
        "avg_degree_final_window": np.full(shape_scalar, np.nan, dtype=np.float32),
        "E_v_time": np.full(shape_time, np.nan, dtype=np.float32),
        "R_c_time": np.full(shape_time, np.nan, dtype=np.float32),
        "d_min_time": np.full(shape_time, np.nan, dtype=np.float32),
    }


def run_sweep(
    sweep_name: str,
    param_values: np.ndarray,
    fixed_R: float | None = None,
    fixed_sigma: float | None = None,
) -> dict:
    """Run either R sweep or sigma sweep."""
    storage = empty_sweep_storage(param_values)

    for p_idx, param_val in enumerate(param_values):
        if sweep_name == "R_sweep":
            R_interaction = float(param_val)
            sigma_f = float(fixed_sigma)
        elif sweep_name == "sigma_sweep":
            R_interaction = float(fixed_R)
            sigma_f = float(param_val)
        else:
            raise ValueError(f"Unknown sweep_name: {sweep_name}")

        print(f"\n{sweep_name}: R={R_interaction:.2f}, sigma={sigma_f:.2f}")

        for trial in range(n_trials):
            seed = 1000 + trial

            out = run_one_trial(
                R_interaction=R_interaction,
                sigma_f=sigma_f,
                seed=seed,
                store_diagnostics=False,
            )

            storage["T_flock"][p_idx, trial] = out["T_flock"]
            storage["E_v_final_window"][p_idx, trial] = out["E_v_final_window"]
            storage["R_c_final_window"][p_idx, trial] = out["R_c_final_window"]
            storage["d_min_over_time"][p_idx, trial] = out["d_min_over_time"]
            storage["avg_degree_final_window"][p_idx, trial] = out["avg_degree_final_window"]
            storage["E_v_time"][p_idx, trial] = out["E_v"]
            storage["R_c_time"][p_idx, trial] = out["R_c"]
            storage["d_min_time"][p_idx, trial] = out["d_min_time"]

            print(f"  trial {trial + 1:02d}/{n_trials}: T_flock={out['T_flock']}")

        T_vals = storage["T_flock"][p_idx]
        valid = ~np.isnan(T_vals)
        print(
            f"  summary: mean T={np.nanmean(T_vals):.3f}, "
            f"std T={np.nanstd(T_vals):.3f}, "
            f"success={np.sum(valid)}/{n_trials}"
        )

    return storage


def main() -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Follower LQR gain K_follower:")
    print(K_follower)

    print("\nRunning R sweep with fixed sigma = 0.5")
    R_sweep = run_sweep(
        sweep_name="R_sweep",
        param_values=R_values,
        fixed_sigma=fixed_sigma_for_R_sweep,
    )

    print("\nRunning sigma sweep with fixed R = 2.0")
    sigma_sweep = run_sweep(
        sweep_name="sigma_sweep",
        param_values=sigma_values,
        fixed_R=fixed_R_for_sigma_sweep,
    )

    print("\nRunning full diagnostic example for trajectory plot")
    example = run_one_trial(
        R_interaction=example_R,
        sigma_f=example_sigma,
        seed=example_seed,
        store_diagnostics=True,
    )

    metadata = {
        "description": "Monte Carlo follower-flocking sweeps.",
        "R_sweep": {
            "R_values": R_values.tolist(),
            "fixed_sigma": fixed_sigma_for_R_sweep,
        },
        "sigma_sweep": {
            "sigma_values": sigma_values.tolist(),
            "fixed_R": fixed_R_for_sigma_sweep,
        },
        "example_case": {
            "R": example_R,
            "sigma": example_sigma,
            "seed": example_seed,
        },
        "n_trials": n_trials,
        "T": T,
        "dt": dt,
        "N_f": N_f,
        "n_pinned": n_pinned,
        "eps_v": eps_v,
        "eps_c": eps_c,
        "dwell_time": dwell_time,
        "d_min": d_min,
        "d_act": d_act,
        "a_max": a_max,
        "init_low": init_low,
        "init_high": init_high,
        "initial_min_distance": initial_min_distance,
        "normalize_follower_weights": normalize_follower_weights,
    }

    np.savez_compressed(
        output_file,
        metadata_json=json.dumps(metadata, indent=2),
        time=time.astype(np.float32),
        followers=followers,
        leader_idx=np.array(leader_idx),
        R_values=R_values,
        sigma_values=sigma_values,
        K_follower=K_follower,

        # R sweep arrays
        R_sweep_T_flock=R_sweep["T_flock"],
        R_sweep_E_v_final_window=R_sweep["E_v_final_window"],
        R_sweep_R_c_final_window=R_sweep["R_c_final_window"],
        R_sweep_d_min_over_time=R_sweep["d_min_over_time"],
        R_sweep_avg_degree_final_window=R_sweep["avg_degree_final_window"],
        R_sweep_E_v_time=R_sweep["E_v_time"],
        R_sweep_R_c_time=R_sweep["R_c_time"],
        R_sweep_d_min_time=R_sweep["d_min_time"],

        # Sigma sweep arrays
        sigma_sweep_T_flock=sigma_sweep["T_flock"],
        sigma_sweep_E_v_final_window=sigma_sweep["E_v_final_window"],
        sigma_sweep_R_c_final_window=sigma_sweep["R_c_final_window"],
        sigma_sweep_d_min_over_time=sigma_sweep["d_min_over_time"],
        sigma_sweep_avg_degree_final_window=sigma_sweep["avg_degree_final_window"],
        sigma_sweep_E_v_time=sigma_sweep["E_v_time"],
        sigma_sweep_R_c_time=sigma_sweep["R_c_time"],
        sigma_sweep_d_min_time=sigma_sweep["d_min_time"],

        # Example full diagnostic arrays
        example_T_flock=np.array(example["T_flock"], dtype=np.float32),
        example_E_v=example["E_v"],
        example_R_c=example["R_c"],
        example_d_min_time=example["d_min_time"],
        example_p_hist=example["p_hist"],
        example_v_hist=example["v_hist"],
        example_u_hist=example["u_hist"],
        example_pinned_hist=example["pinned_hist"],
        example_avg_degree_hist=example["avg_degree_hist"],
    )

    print(f"\nSaved all data to: {output_file}")


if __name__ == "__main__":
    main()
