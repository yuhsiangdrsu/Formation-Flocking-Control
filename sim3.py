"""
generate_formation_flocking_case_study.py

Hierarchical formation-flocking simulation case study.

Saves the required data for later plotting:
    results/formation_flocking_case_study_data.npz

Saved data includes:
    - agent trajectories
    - agent velocities
    - acceleration commands
    - target trajectory
    - energy histories
    - alive/dead histories
    - leader histories
    - follower assignment histories
    - follower allocation counts
    - leader pinning histories
    - follower pinning histories
    - subgroup formation error
    - subgroup velocity tracking error
    - subgroup cohesion radius
    - subgroup velocity alignment error
    - global and subgroup minimum inter-agent distance
    - global cohesion radius
    - global velocity alignment error
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are
from matplotlib import cm, colors


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(10)


# ============================================================
# Simulation parameters
# ============================================================

N = 150
dt = 0.01
T = 30.0
steps = int(T / dt)

n_pin_leaders = 5
n_pin_followers = 10


# ============================================================
# Mission / formation parameters
# ============================================================

R_f = 10.0
d_max = 15.0
theta0 = 0.0

M = int(np.ceil(2 * np.pi * R_f / d_max))
M = max(3, M)
M = min(M, N - 1)


# ============================================================
# Time-varying formation parameters
# ============================================================

R_breath_amp = 0
R_breath_freq = 0.25
formation_spin_rate = 0.12


# ============================================================
# LQR design
# ============================================================

A = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])

B = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

Q_leader = np.diag([5.0, 5.0, 1.0, 1.0])
R_leader = np.diag([1.0, 1.0])

Q_follower = np.diag([10.0, 10.0, 50.0, 50.0])
R_follower = np.diag([1.0, 1.0])


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


K_leader, P_leader = lqr_gain(A, B, Q_leader, R_leader)
K_follower, P_follower = lqr_gain(A, B, Q_follower, R_follower)


# ============================================================
# Potential-based repulsive avoidance, followers only
# ============================================================

d_min = 0.5
d_act = 2.0
k_phi = 10.0
eps_phi = 1e-6


# ============================================================
# Saturation
# ============================================================

a_max = 60.0


# ============================================================
# Assignment parameters
# ============================================================

alpha = 1.0
beta = 50.0
reassign_every = 1000


# ============================================================
# Radius-based follower graph parameters
# ============================================================

R_follower_interaction = 3.0


# ============================================================
# Weighted flocking parameters
# ============================================================

sigma_f = 5.0
normalize_follower_weights = True
eps_w = 1e-8


# ============================================================
# Energy-aware leader selection weights
# ============================================================

w_energy = 5.0
w_dist = 1.0


# ============================================================
# Energy depletion parameters
# ============================================================

leader_decay_base = 5.0
follower_decay_base = 1.0
motion_decay_gain_v = 0.02
motion_decay_gain_u = 0.01


# ============================================================
# Energy colormap parameters
# ============================================================

energy_max = 100.0
cmap_energy = plt.cm.viridis
norm_energy = colors.Normalize(vmin=0, vmax=energy_max)


# ============================================================
# Initial states
# ============================================================

def initialize_positions_with_min_distance(
    N: int,
    low: float,
    high: float,
    d_act: float,
    max_tries: int = 100000,
) -> np.ndarray:
    """
    Sample N agent positions uniformly in [low, high]^2 such that
    every pairwise initial distance is strictly greater than d_act.
    """

    p = np.empty((N, 2), dtype=float)

    for i in range(N):
        placed = False

        for _ in range(max_tries):
            candidate = np.random.uniform(low, high, size=2)

            if i == 0:
                p[i] = candidate
                placed = True
                break

            dists = np.linalg.norm(p[:i] - candidate, axis=1)

            if np.all(dists > d_act):
                p[i] = candidate
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Could not place agent {i} with all pairwise distances > d_act={d_act}. "
                f"Try increasing the sampling region or reducing N."
            )

    return p


p = initialize_positions_with_min_distance(
    N=N,
    low=-15,
    high=15,
    d_act=d_act,
)

v = np.zeros((N, 2))
p_t = np.array([0.0, 0.0])

energy = np.random.uniform(80, 100, size=N)
alive = np.ones(N, dtype=bool)


# ============================================================
# Target motion: constant circular velocity
# ============================================================

# Circular target trajectory:
#     p_t(t) = p_center + R_target [cos(omega_t t), sin(omega_t t)]
#
# Velocity:
#     v_t(t) = R_target * omega_t [-sin(omega_t t), cos(omega_t t)]
#
# Acceleration:
#     a_t(t) = -R_target * omega_t^2 [cos(omega_t t), sin(omega_t t)]
#
# Speed is constant:
#     ||v_t(t)|| = R_target * omega_t

p_target_center = np.array([0.0, 0.0])
R_target = 8.0
omega_target = 0.08

p_t = p_target_center + R_target * np.array([
    np.cos(0.0),
    np.sin(0.0),
])


def target_position(t: float) -> np.ndarray:
    return p_target_center + R_target * np.array([
        np.cos(omega_target * t),
        np.sin(omega_target * t),
    ])


def target_velocity(t: float) -> np.ndarray:
    return R_target * omega_target * np.array([
        -np.sin(omega_target * t),
         np.cos(omega_target * t),
    ])


def target_acceleration(t: float) -> np.ndarray:
    return -R_target * omega_target**2 * np.array([
        np.cos(omega_target * t),
        np.sin(omega_target * t),
    ])


# ============================================================
# Time-varying formation geometry helpers
# ============================================================

def formation_profile(t: float):
    """
    Time-varying formation profile.

    Returns:
        R, R_dot, R_ddot, theta_c, theta_dot, theta_ddot
    """

    R = R_f + R_breath_amp * np.sin(R_breath_freq * t)
    R_dot = R_breath_amp * R_breath_freq * np.cos(R_breath_freq * t)
    R_ddot = -R_breath_amp * R_breath_freq**2 * np.sin(R_breath_freq * t)

    theta_c = theta0 + formation_spin_rate * t
    theta_dot = formation_spin_rate
    theta_ddot = 0.0

    return R, R_dot, R_ddot, theta_c, theta_dot, theta_ddot


def desired_leader_kinematics(M: int, t: float):
    """
    Desired time-varying leader offsets and their derivatives.

    Returns:
        offsets       r_k*(t)
        offsets_dot   dot{r}_k*(t)
        offsets_ddot  ddot{r}_k*(t)
        thetas
        R
    """

    R, R_dot, R_ddot, theta_c, theta_dot, theta_ddot = formation_profile(t)

    thetas = theta_c + 2 * np.pi * np.arange(M) / M

    e = np.stack([np.cos(thetas), np.sin(thetas)], axis=1)
    e_perp = np.stack([-np.sin(thetas), np.cos(thetas)], axis=1)

    offsets = R * e
    offsets_dot = R_dot * e + R * theta_dot * e_perp
    offsets_ddot = (
        (R_ddot - R * theta_dot**2) * e
        + (2.0 * R_dot * theta_dot + R * theta_ddot) * e_perp
    )

    return offsets, offsets_dot, offsets_ddot, thetas, R


def desired_leader_offsets(
    M: int,
    R_f_unused: float | None = None,
    theta0_unused: float = 0.0,
    t: float = 0.0,
):
    offsets, _, _, thetas, _ = desired_leader_kinematics(M, t)
    return offsets, thetas


def desired_leader_points(
    p_t: np.ndarray,
    M: int,
    R_f_unused: float | None = None,
    theta0_unused: float = 0.0,
    t: float = 0.0,
):
    offsets, _, _, _, _ = desired_leader_kinematics(M, t)
    return p_t[None, :] + offsets


# ============================================================
# Leader selection
# ============================================================

def select_leaders(
    p: np.ndarray,
    p_t: np.ndarray,
    M: int,
    R_f: float,
    theta0: float = 0.0,
    energy: np.ndarray | None = None,
    alive: np.ndarray | None = None,
    w_energy: float = 5.0,
    w_dist: float = 1.0,
    t: float = 0.0,
) -> np.ndarray:
    desired_points = desired_leader_points(p_t, M, R_f, theta0, t=t)

    if alive is None:
        remaining_agents = list(range(len(p)))
    else:
        remaining_agents = [i for i in range(len(p)) if alive[i]]

    leaders = []

    for k in range(M):
        if len(remaining_agents) == 0:
            break

        scores = []
        for i in remaining_agents:
            dist = np.linalg.norm(p[i] - desired_points[k])
            if energy is None:
                score = -dist
            else:
                score = w_energy * (energy[i] / 100.0) - w_dist * dist
            scores.append(score)

        chosen_local = int(np.argmax(scores))
        chosen_agent = remaining_agents.pop(chosen_local)
        leaders.append(chosen_agent)

    while len(leaders) < M:
        leaders.append(-1)

    return np.array(leaders, dtype=int)


leaders = select_leaders(
    p,
    p_t,
    M,
    R_f,
    theta0,
    energy=energy,
    alive=alive,
    w_energy=w_energy,
    w_dist=w_dist,
    t=0.0,
)


def get_followers(N: int, leaders: np.ndarray, alive: np.ndarray) -> np.ndarray:
    leader_set = set([x for x in leaders if x >= 0])
    return np.array(
        [i for i in range(N) if alive[i] and i not in leader_set],
        dtype=int,
    )


followers = get_followers(N, leaders, alive)


# ============================================================
# Utility functions
# ============================================================

def sat_vec(vec: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n <= vmax or n < 1e-12:
        return vec
    return (vmax / n) * vec


def compute_potential_avoidance(
    i: int,
    p: np.ndarray,
    alive: np.ndarray,
    d_min: float,
    d_act: float,
    k_phi: float,
    eps_phi: float = 1e-6,
) -> np.ndarray:
    """
    Potential-based repulsive avoidance for followers only.

    phi(r) = 0.5 * k_phi * (1/(r-d_min) - 1/(d_act-d_min))^2,
             d_min < r < d_act
           = 0,
             r >= d_act

    u_avoid_i = -grad_{p_i} Phi(p)
    """

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
        f_mag = k_phi * s / (gap ** 2)

        u_avoid += f_mag * z_hat

    return u_avoid


def build_leader_neighbors(M: int) -> dict[int, list[int]]:
    neigh = {}
    for k in range(M):
        neigh[k] = [((k - 1) % M), ((k + 1) % M)]
    return neigh


leader_neighbors = build_leader_neighbors(M)


def choose_pinned_leader_slots_geometry_based(
    leaders: np.ndarray,
    alive: np.ndarray,
    p: np.ndarray,
    p_t: np.ndarray,
    M: int,
    R_f: float,
    theta0: float,
    leader_neighbors: dict[int, list[int]],
    n_pin_leaders: int,
    t: float,
) -> list[int]:
    if n_pin_leaders <= 0:
        return []

    offsets, _ = desired_leader_offsets(M, R_f, theta0, t=t)
    scores = []

    for k in range(M):
        idx_k = leaders[k]
        if idx_k < 0 or not alive[idx_k]:
            continue

        delta_k = 0.0
        for j in leader_neighbors[k]:
            idx_j = leaders[j]
            if idx_j < 0 or not alive[idx_j]:
                continue

            actual_rel = p[idx_k] - p[idx_j]
            desired_rel = offsets[k] - offsets[j]
            delta_k += np.linalg.norm(actual_rel - desired_rel)

        scores.append((delta_k, k))

    if len(scores) == 0:
        return []

    scores.sort(key=lambda x: x[0], reverse=True)
    pinned = [k for _, k in scores[:min(n_pin_leaders, len(scores))]]
    return pinned


def compute_follower_pins(
    p: np.ndarray,
    leaders: np.ndarray,
    g: np.ndarray,
    alive: np.ndarray,
    M: int,
    n_pin_followers: int,
) -> np.ndarray:
    c = np.zeros(len(p), dtype=float)

    for k in range(M):
        leader_idx = leaders[k]
        if leader_idx < 0 or not alive[leader_idx]:
            continue

        subgroup = [i for i in range(len(p)) if alive[i] and g[i] == k]
        if len(subgroup) == 0:
            continue

        subgroup_sorted = sorted(
            subgroup,
            key=lambda i: np.linalg.norm(p[i] - p[leader_idx]),
        )

        n_pick = min(n_pin_followers, len(subgroup_sorted))
        selected = subgroup_sorted[-n_pick:]

        for i_pin in selected:
            c[i_pin] = 5.0

    return c


def build_follower_neighbor_graph(
    p: np.ndarray,
    followers: np.ndarray,
    g: np.ndarray,
    alive: np.ndarray,
    M: int,
    R_follower_interaction: float,
) -> dict[int, list[int]]:
    """
    Radius-based intra-subgroup follower graph.

    For follower i:
        N_i^f = {
            j in F :
            g(j) = g(i),
            j != i,
            ||p_i - p_j|| <= R_follower_interaction
        }
    """

    follower_neighbors = {int(i): [] for i in followers}

    for k in range(M):
        subgroup = [int(i) for i in followers if alive[i] and g[i] == k]

        if len(subgroup) <= 1:
            continue

        for i in subgroup:
            neigh_i = []

            for j in subgroup:
                if j == i:
                    continue

                dist_ij = np.linalg.norm(p[i] - p[j])

                if dist_ij <= R_follower_interaction:
                    neigh_i.append(j)

            follower_neighbors[i] = neigh_i

    return follower_neighbors


def compute_weighted_follower_graph_error(
    i: int,
    p: np.ndarray,
    v: np.ndarray,
    follower_neighbors: dict[int, list[int]],
    alive: np.ndarray,
    sigma_f: float,
    normalize_weights: bool = True,
    eps_w: float = 1e-8,
) -> np.ndarray:
    x_i = np.hstack((p[i], v[i]))
    neigh_i = [j for j in follower_neighbors.get(i, []) if alive[j]]

    if len(neigh_i) == 0:
        return np.zeros(4)

    weights = []
    for j in neigh_i:
        dist_ij = np.linalg.norm(p[i] - p[j])
        a_ij = np.exp(-(dist_ij ** 2) / (sigma_f ** 2))
        weights.append(a_ij)

    weights = np.array(weights, dtype=float)

    if normalize_weights:
        weights = weights / (np.sum(weights) + eps_w)

    e_graph = np.zeros(4)
    for w_ij, j in zip(weights, neigh_i):
        x_j = np.hstack((p[j], v[j]))
        e_graph += w_ij * (x_i - x_j)

    return e_graph


# ============================================================
# Follower assignment
# ============================================================

g = -np.ones(N, dtype=int)


def assign_followers(
    p: np.ndarray,
    leaders: np.ndarray,
    followers: np.ndarray,
    alive: np.ndarray,
    g_prev: np.ndarray | None = None,
) -> np.ndarray:
    g_new = -np.ones(N, dtype=int)
    n_k = np.zeros(M, dtype=int)

    valid_leader_slots = [k for k in range(M) if leaders[k] >= 0 and alive[leaders[k]]]

    if len(valid_leader_slots) == 0:
        return g_new

    for i in followers:
        if not alive[i]:
            continue

        costs = []
        slots = []
        for k in valid_leader_slots:
            Lk = leaders[k]
            dist2 = np.sum((p[i] - p[Lk]) ** 2)
            Jik = alpha * dist2 + beta * n_k[k]
            costs.append(Jik)
            slots.append(k)

        k_best_local = int(np.argmin(costs))
        k_best = slots[k_best_local]
        g_new[i] = k_best
        n_k[k_best] += 1

    return g_new


g = assign_followers(p, leaders, followers, alive)


# ============================================================
# Replacement leader selection
# ============================================================

def select_replacement_leader_for_slot(
    slot_k: int,
    subgroup_followers: list[int],
    p: np.ndarray,
    p_t: np.ndarray,
    energy: np.ndarray,
    alive: np.ndarray,
    M: int,
    R_f: float,
    theta0: float,
    w_energy: float,
    w_dist: float,
    t: float,
) -> int:
    candidates = [i for i in subgroup_followers if alive[i]]
    if len(candidates) == 0:
        return -1

    desired_points = desired_leader_points(p_t, M, R_f, theta0, t=t)
    desired_k = desired_points[slot_k]

    scores = []
    for i in candidates:
        dist = np.linalg.norm(p[i] - desired_k)
        score = w_energy * (energy[i] / 100.0) - w_dist * dist
        scores.append(score)

    best_local = int(np.argmax(scores))
    return candidates[best_local]


# ============================================================
# Diagnostics
# ============================================================

print("Leader LQR gain K_leader:")
print(K_leader)
print("\nFollower LQR gain K_follower:")
print(K_follower)
print("\nPotential-avoidance parameters:")
print(f"d_min = {d_min}, d_act = {d_act}, k_phi = {k_phi}")
print("\nTime-varying formation parameters:")
print(f"R_f = {R_f}, R_breath_amp = {R_breath_amp}, R_breath_freq = {R_breath_freq}")
print(f"formation_spin_rate = {formation_spin_rate}")
print("\nRadius-based follower interaction parameters:")
print(f"R_follower_interaction = {R_follower_interaction}, sigma_f = {sigma_f}")


# ============================================================
# Optional initial diagnostic plots
# ============================================================

SHOW_INITIAL_DIAGNOSTIC_PLOTS = False

if SHOW_INITIAL_DIAGNOSTIC_PLOTS:
    desired_points0 = desired_leader_points(p_t, M, R_f, theta0, t=0.0)
    remaining_agents_dbg = [i for i in range(N) if alive[i]]

    for k in range(M):
        if leaders[k] < 0:
            continue

        desired_k = desired_points0[k]
        candidate_ids = np.array(remaining_agents_dbg, dtype=int)
        candidate_dists = np.array([np.linalg.norm(p[i] - desired_k) for i in candidate_ids])
        candidate_energies = energy[candidate_ids]
        candidate_scores = w_energy * (candidate_energies / 100.0) - w_dist * candidate_dists

        sorted_idx = np.argsort(-candidate_scores)
        ranks = np.empty(len(candidate_scores), dtype=int)
        ranks[sorted_idx] = np.arange(1, len(candidate_scores) + 1)

        chosen_agent = leaders[k]
        chosen_local = int(np.where(candidate_ids == chosen_agent)[0][0])
        chosen_score = float(candidate_scores[chosen_local])

        fig_k, ax_k = plt.subplots(figsize=(14, 5))
        bars = ax_k.bar(np.arange(len(candidate_ids)), candidate_scores)

        bars[chosen_local].set_edgecolor("red")
        bars[chosen_local].set_linewidth(2.5)
        bars[chosen_local].set_alpha(1.0)

        for b_idx, b in enumerate(bars):
            if b_idx != chosen_local:
                b.set_alpha(0.6)

        ax_k.set_title(f"Leader allocation step k={k}: score of remaining agents for leader slot {k}")
        ax_k.set_xlabel("Remaining candidate agents")
        ax_k.set_ylabel("Score = w_energy*(energy/100) - w_dist*distance")
        ax_k.set_xticks(np.arange(len(candidate_ids)))
        ax_k.set_xticklabels([f"agent {i}" for i in candidate_ids], rotation=90)
        ax_k.grid(True, axis="y", alpha=0.3)

        for idx in range(len(candidate_ids)):
            ax_k.text(
                idx,
                candidate_scores[idx] + 0.03,
                f"#{ranks[idx]}\nE={candidate_energies[idx]:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        ax_k.text(
            chosen_local,
            chosen_score + 0.18,
            f"selected: agent {chosen_agent}\nrank #1\nenergy={energy[chosen_agent]:.1f}%\nscore={chosen_score:.2f}",
            ha="center",
            va="bottom",
            color="red",
            fontsize=9,
        )

        plt.tight_layout()
        plt.show()

        remaining_agents_dbg.remove(chosen_agent)

    followers_per_leader = np.zeros(M, dtype=int)
    for i in followers:
        if g[i] >= 0:
            followers_per_leader[g[i]] += 1

    fig_follower_count, ax_follower_count = plt.subplots(figsize=(10, 5))
    bars2 = ax_follower_count.bar(np.arange(M), followers_per_leader)
    ax_follower_count.set_title("Number of Assigned Followers for Each Leader")
    ax_follower_count.set_xlabel("Leader slot k")
    ax_follower_count.set_ylabel("Number of followers")
    ax_follower_count.set_xticks(np.arange(M))
    ax_follower_count.set_xticklabels([
        f"k={k}\nagent {leaders[k]}" if leaders[k] >= 0 else f"k={k}\nempty"
        for k in range(M)
    ])
    ax_follower_count.grid(True, axis="y", alpha=0.3)

    for k, b in enumerate(bars2):
        ax_follower_count.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{followers_per_leader[k]}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()

    fig_energy, ax_energy = plt.subplots(figsize=(12, 5))
    bars_energy = ax_energy.bar(np.arange(N), energy)

    leader_set = set([x for x in leaders if x >= 0])
    for i in range(N):
        if i not in leader_set:
            bars_energy[i].set_alpha(0.6)

    for k, leader_idx in enumerate(leaders):
        if leader_idx >= 0:
            bars_energy[leader_idx].set_edgecolor("red")
            bars_energy[leader_idx].set_linewidth(2.0)
            bars_energy[leader_idx].set_alpha(1.0)
            ax_energy.text(
                leader_idx,
                energy[leader_idx] + 1.0,
                f"L{k}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="red",
            )

    ax_energy.set_title("Initial Energy Level of All Agents (Selected Leaders Highlighted)")
    ax_energy.set_xlabel("Agent index")
    ax_energy.set_ylabel("Energy (%)")
    ax_energy.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# History arrays
# ============================================================

time = np.arange(steps) * dt

p_hist = np.full((steps, N, 2), np.nan)
v_hist = np.full((steps, N, 2), np.nan)
u_hist = np.full((steps, N, 2), np.nan)

pt_hist = np.zeros((steps, 2))
vt_hist = np.zeros((steps, 2))
at_hist = np.zeros((steps, 2))
R_form_hist = np.zeros(steps)

energy_hist = np.full((steps, N), np.nan)
alive_hist = np.zeros((steps, N), dtype=bool)

leaders_hist = np.full((steps, M), -1, dtype=int)
g_hist = np.full((steps, N), -1, dtype=int)

follower_pin_hist = np.zeros((steps, N), dtype=float)
leader_pin_hist = np.zeros((steps, M), dtype=float)

# Note: variable-length Python dicts cannot be stored with allow_pickle=False.
# We keep this for optional animation inside this script, but we do not save it in the npz.
follower_neighbors_hist = []

leader_form_error_hist = np.full((steps, M), np.nan)
avg_follower_degree_hist = np.zeros(steps)


# ============================================================
# Additional case-study metrics for later plotting
# ============================================================

# Leader/subgroup formation tracking error:
# e_p,k(t) = || p_Lk(t) - (p_t(t) + r_k^*(t)) ||
subgroup_formation_error_hist = np.full((steps, M), np.nan)

# Leader/subgroup velocity tracking error:
# e_v,k(t) = || v_Lk(t) - (v_t(t) + dot{r}_k^*(t)) ||
subgroup_velocity_error_hist = np.full((steps, M), np.nan)

# Cohesion radius of each subgroup:
# R_c,k(t) = max_{i in subgroup k} || p_i(t) - p_Lk(t) ||
subgroup_cohesion_radius_hist = np.full((steps, M), np.nan)

# Velocity alignment error of each subgroup:
# E_v,k(t) = mean_{i in followers of subgroup k} || v_i(t) - v_Lk(t) ||
subgroup_velocity_alignment_error_hist = np.full((steps, M), np.nan)

# Minimum inter-agent distance over all alive agents:
# d_min_global(t) = min_{i<j, alive} ||p_i - p_j||
min_inter_agent_distance_hist = np.full(steps, np.nan)

# Minimum inter-agent distance inside each subgroup:
# d_min,k(t) = min_{i<j in subgroup k} ||p_i - p_j||
subgroup_min_inter_agent_distance_hist = np.full((steps, M), np.nan)

# Number of followers allocated to each leader slot over time.
followers_per_leader_hist = np.zeros((steps, M), dtype=int)

# Global velocity alignment error over all alive agents:
# E_v_global(t) = mean_i ||v_i - mean(v_alive)||
global_velocity_alignment_error_hist = np.full(steps, np.nan)

# Global cohesion radius around target:
# R_c_global(t) = max_i ||p_i - p_t||
global_cohesion_radius_hist = np.full(steps, np.nan)


# ============================================================
# Case-study metric helper
# ============================================================

def compute_case_study_metrics(
    s: int,
    p: np.ndarray,
    v: np.ndarray,
    p_t: np.ndarray,
    v_t: np.ndarray,
    offsets: np.ndarray,
    offsets_dot: np.ndarray,
    leaders: np.ndarray,
    g: np.ndarray,
    alive: np.ndarray,
    M: int,
) -> None:
    """
    Compute and store case-study metrics at time index s.

    Subgroup k consists of:
        - leader assigned to slot k
        - followers with g[i] == k

    Saved metrics:
        subgroup formation error
        subgroup velocity tracking error
        subgroup cohesion radius
        subgroup velocity alignment error
        subgroup minimum inter-agent distance
        global minimum inter-agent distance
        follower allocation counts
        global velocity alignment error
        global cohesion radius
    """

    alive_ids = np.where(alive)[0]

    # --------------------------------------------------------
    # Global minimum inter-agent distance
    # --------------------------------------------------------
    if len(alive_ids) >= 2:
        dvals = []

        for a in range(len(alive_ids)):
            i = alive_ids[a]

            for b in range(a + 1, len(alive_ids)):
                j = alive_ids[b]
                dvals.append(np.linalg.norm(p[i] - p[j]))

        if len(dvals) > 0:
            min_inter_agent_distance_hist[s] = np.min(dvals)

    # --------------------------------------------------------
    # Global velocity alignment error
    # --------------------------------------------------------
    if len(alive_ids) > 0:
        v_alive = v[alive_ids]
        v_mean = np.mean(v_alive, axis=0)

        global_velocity_alignment_error_hist[s] = np.mean(
            np.linalg.norm(v_alive - v_mean, axis=1)
        )

        global_cohesion_radius_hist[s] = np.max(
            np.linalg.norm(p[alive_ids] - p_t, axis=1)
        )

    # --------------------------------------------------------
    # Per-subgroup metrics
    # --------------------------------------------------------
    for k in range(M):
        leader_idx = leaders[k]

        if leader_idx < 0 or not alive[leader_idx]:
            continue

        subgroup_followers = [
            i for i in range(len(p))
            if alive[i] and g[i] == k
        ]

        subgroup_agents = [leader_idx] + subgroup_followers

        followers_per_leader_hist[s, k] = len(subgroup_followers)

        # Desired leader position and velocity for slot k.
        p_des_k = p_t + offsets[k]
        v_des_k = v_t + offsets_dot[k]

        # Formation error for leader/subgroup slot k.
        subgroup_formation_error_hist[s, k] = np.linalg.norm(
            p[leader_idx] - p_des_k
        )

        # Velocity tracking error for leader/subgroup slot k.
        subgroup_velocity_error_hist[s, k] = np.linalg.norm(
            v[leader_idx] - v_des_k
        )

        # Cohesion radius of subgroup k around its leader.
        if len(subgroup_agents) > 0:
            subgroup_positions = p[subgroup_agents]

            subgroup_cohesion_radius_hist[s, k] = np.max(
                np.linalg.norm(subgroup_positions - p[leader_idx], axis=1)
            )

        # Follower velocity alignment error relative to leader.
        if len(subgroup_followers) > 0:
            follower_velocities = v[subgroup_followers]

            subgroup_velocity_alignment_error_hist[s, k] = np.mean(
                np.linalg.norm(follower_velocities - v[leader_idx], axis=1)
            )

        # Minimum inter-agent distance inside subgroup k.
        if len(subgroup_agents) >= 2:
            dvals_k = []

            for a in range(len(subgroup_agents)):
                i = subgroup_agents[a]

                for b in range(a + 1, len(subgroup_agents)):
                    j = subgroup_agents[b]
                    dvals_k.append(np.linalg.norm(p[i] - p[j]))

            if len(dvals_k) > 0:
                subgroup_min_inter_agent_distance_hist[s, k] = np.min(dvals_k)


# ============================================================
# Main simulation
# ============================================================

for s in range(steps):
    t = s * dt

    followers = get_followers(N, leaders, alive)

    p_t = target_position(t)
    v_t = target_velocity(t)
    a_t = target_acceleration(t)

    # Current time-varying desired formation kinematics.
    offsets, offsets_dot, offsets_ddot, _, R_now = desired_leader_kinematics(M, t)

    pinned_leader_slots = choose_pinned_leader_slots_geometry_based(
        leaders=leaders,
        alive=alive,
        p=p,
        p_t=p_t,
        M=M,
        R_f=R_f,
        theta0=theta0,
        leader_neighbors=leader_neighbors,
        n_pin_leaders=n_pin_leaders,
        t=t,
    )

    follower_pins = compute_follower_pins(
        p,
        leaders,
        g,
        alive,
        M,
        n_pin_followers=n_pin_followers,
    )

    follower_neighbors = build_follower_neighbor_graph(
        p=p,
        followers=followers,
        g=g,
        alive=alive,
        M=M,
        R_follower_interaction=R_follower_interaction,
    )

    if len(followers) > 0:
        avg_follower_degree_hist[s] = np.mean([
            len(follower_neighbors.get(int(i), [])) for i in followers
        ])
    else:
        avg_follower_degree_hist[s] = 0.0

    u = np.zeros((N, 2))

    # --------------------------------------------------------
    # Leaders
    # --------------------------------------------------------
    for k in range(M):
        agent_idx = leaders[k]
        if agent_idx < 0 or not alive[agent_idx]:
            continue

        x_i = np.hstack((p[agent_idx], v[agent_idx]))

        r_k_star = offsets[k]
        r_k_dot_star = offsets_dot[k]
        r_k_ddot_star = offsets_ddot[k]

        x_star = np.hstack((p_t + r_k_star, v_t + r_k_dot_star))

        b_k = 1.0 if k in pinned_leader_slots else 0.0
        e_pin = b_k * (x_i - x_star)

        e_graph = np.zeros(4)
        for j in leader_neighbors[k]:
            j_idx = leaders[j]
            if j_idx < 0 or not alive[j_idx]:
                continue

            x_j = np.hstack((p[j_idx], v[j_idx]))

            r_j_star = offsets[j]
            r_j_dot_star = offsets_dot[j]
            x_j_star = np.hstack((p_t + r_j_star, v_t + r_j_dot_star))

            e_graph += (x_i - x_j) - (x_star - x_j_star)

        e_total = e_pin + e_graph

        # Time-varying formation feedforward acceleration included here.
        u[agent_idx] = b_k * (a_t + r_k_ddot_star) - K_leader @ e_total

    # --------------------------------------------------------
    # Followers
    # --------------------------------------------------------
    for i in followers:
        if not alive[i]:
            continue

        k = g[i]
        if k < 0:
            continue

        Lk = leaders[k]
        if Lk < 0 or not alive[Lk]:
            continue

        x_i = np.hstack((p[i], v[i]))
        x_Lk = np.hstack((p[Lk], v[Lk]))

        c_i = follower_pins[i]
        e_pin = c_i * (x_i - x_Lk)

        e_graph = compute_weighted_follower_graph_error(
            i=int(i),
            p=p,
            v=v,
            follower_neighbors=follower_neighbors,
            alive=alive,
            sigma_f=sigma_f,
            normalize_weights=normalize_follower_weights,
            eps_w=eps_w,
        )

        e_total = e_pin + e_graph

        u_avoid = compute_potential_avoidance(
            int(i),
            p,
            alive,
            d_min=d_min,
            d_act=d_act,
            k_phi=k_phi,
            eps_phi=eps_phi,
        )

        u[i] = -K_follower @ e_total + u_avoid

    # --------------------------------------------------------
    # Saturate acceleration
    # --------------------------------------------------------
    for i in range(N):
        if alive[i]:
            u[i] = sat_vec(u[i], a_max)
        else:
            u[i] = np.zeros(2)

    # --------------------------------------------------------
    # Integrate dynamics
    # --------------------------------------------------------
    v = v + dt * u
    p = p + dt * v

    # --------------------------------------------------------
    # Energy depletion
    # --------------------------------------------------------
    current_leader_set = set([x for x in leaders if x >= 0 and alive[x]])

    for i in range(N):
        if not alive[i]:
            continue

        motion_cost = (
            motion_decay_gain_v * np.linalg.norm(v[i])
            + motion_decay_gain_u * np.linalg.norm(u[i])
        )

        if i in current_leader_set:
            energy[i] -= dt * (leader_decay_base + motion_cost)
        else:
            energy[i] -= dt * (follower_decay_base + motion_cost)

        if energy[i] < 0.0:
            energy[i] = 0.0

    # --------------------------------------------------------
    # Follower depletion
    # --------------------------------------------------------
    for i in range(N):
        if alive[i] and energy[i] <= 0.0 and i not in current_leader_set:
            alive[i] = False
            g[i] = -1

    # --------------------------------------------------------
    # Leader depletion and replacement
    # --------------------------------------------------------
    for k in range(M):
        leader_idx = leaders[k]
        if leader_idx < 0:
            continue

        if alive[leader_idx] and energy[leader_idx] > 0.0:
            continue

        alive[leader_idx] = False
        leaders[k] = -1

        subgroup_followers = [i for i in range(N) if alive[i] and g[i] == k]

        replacement = select_replacement_leader_for_slot(
            slot_k=k,
            subgroup_followers=subgroup_followers,
            p=p,
            p_t=p_t,
            energy=energy,
            alive=alive,
            M=M,
            R_f=R_f,
            theta0=theta0,
            w_energy=w_energy,
            w_dist=w_dist,
            t=t,
        )

        if replacement >= 0:
            leaders[k] = replacement
            g[replacement] = -1

    followers = get_followers(N, leaders, alive)

    # Optional periodic reassignment, now using the current leader set.
    # Uncomment if you want follower groups to adapt during the run.
    # if s > 0 and s % reassign_every == 0:
    #     g = assign_followers(p, leaders, followers, alive, g_prev=g)

    # --------------------------------------------------------
    # Save state histories
    # --------------------------------------------------------
    p_hist[s, alive, :] = p[alive]
    v_hist[s, alive, :] = v[alive]
    u_hist[s, alive, :] = u[alive]

    pt_hist[s] = p_t
    vt_hist[s] = v_t
    at_hist[s] = a_t
    R_form_hist[s] = R_now

    energy_hist[s] = energy.copy()
    alive_hist[s] = alive
    leaders_hist[s] = leaders
    g_hist[s] = g.copy()

    follower_pin_hist[s] = follower_pins.copy()
    follower_neighbors_hist.append({
        int(i): list(neigh) for i, neigh in follower_neighbors.items()
    })

    # --------------------------------------------------------
    # Existing local leader formation edge error
    # --------------------------------------------------------
    for k in range(M):
        idx_k = leaders[k]
        if idx_k < 0 or not alive[idx_k]:
            continue

        err_k = 0.0
        valid_neighbors = 0

        for j in leader_neighbors[k]:
            idx_j = leaders[j]
            if idx_j < 0 or not alive[idx_j]:
                continue

            actual_rel = p[idx_k] - p[idx_j]
            desired_rel = offsets[k] - offsets[j]
            err_k += np.linalg.norm(actual_rel - desired_rel)
            valid_neighbors += 1

        if valid_neighbors > 0:
            leader_form_error_hist[s, k] = err_k

    # --------------------------------------------------------
    # New case-study metrics for later plotting
    # --------------------------------------------------------
    compute_case_study_metrics(
        s=s,
        p=p,
        v=v,
        p_t=p_t,
        v_t=v_t,
        offsets=offsets,
        offsets_dot=offsets_dot,
        leaders=leaders,
        g=g,
        alive=alive,
        M=M,
    )

    # --------------------------------------------------------
    # Leader pin history
    # --------------------------------------------------------
    for k in pinned_leader_slots:
        if 0 <= k < M:
            leader_pin_hist[s, k] = 1.0


# # ============================================================
# # Save case-study data for later plotting
# # ============================================================
#
# results_dir = Path("results")
# results_dir.mkdir(parents=True, exist_ok=True)
#
# case_study_file = results_dir / "formation_flocking_case_study_data.npz"
#
# metadata = {
#     "N": N,
#     "M": M,
#     "dt": dt,
#     "T": T,
#     "steps": steps,
#     "R_f": R_f,
#     "d_max": d_max,
#     "theta0": theta0,
#     "R_breath_amp": R_breath_amp,
#     "R_breath_freq": R_breath_freq,
#     "formation_spin_rate": formation_spin_rate,
#     "n_pin_leaders": n_pin_leaders,
#     "n_pin_followers": n_pin_followers,
#     "d_min": d_min,
#     "d_act": d_act,
#     "k_phi": k_phi,
#     "a_max": a_max,
#     "alpha": alpha,
#     "beta": beta,
#     "R_follower_interaction": R_follower_interaction,
#     "sigma_f": sigma_f,
#     "normalize_follower_weights": normalize_follower_weights,
#     "w_energy": w_energy,
#     "w_dist": w_dist,
#     "leader_decay_base": leader_decay_base,
#     "follower_decay_base": follower_decay_base,
#     "motion_decay_gain_v": motion_decay_gain_v,
#     "motion_decay_gain_u": motion_decay_gain_u,
#     "energy_max": energy_max,
#     "target_motion": "constant circular velocity",
#     "p_target_center": p_target_center.tolist(),
#     "R_target": R_target,
#     "omega_target": omega_target,
#     "target_speed": R_target * omega_target,
#     "metric_definitions": {
#         "leader_form_error_hist": "sum over neighboring leader slots j of ||(p_Lk - p_Lj) - (r_k_star - r_j_star)||",
#         "subgroup_formation_error_hist": "||p_Lk - (p_t + r_k_star)||",
#         "subgroup_velocity_error_hist": "||v_Lk - (v_t + r_dot_k_star)||",
#         "subgroup_cohesion_radius_hist": "max_{i in subgroup k including leader} ||p_i - p_Lk||",
#         "subgroup_velocity_alignment_error_hist": "mean_{followers i in subgroup k} ||v_i - v_Lk||",
#         "min_inter_agent_distance_hist": "min_{alive i<j} ||p_i - p_j||",
#         "subgroup_min_inter_agent_distance_hist": "min_{i<j in subgroup k including leader} ||p_i - p_j||",
#         "global_velocity_alignment_error_hist": "mean_i ||v_i - mean(v_alive)||",
#         "global_cohesion_radius_hist": "max_i ||p_i - p_t||",
#     },
# }
#
# np.savez_compressed(
#     case_study_file,
#
#     # Metadata
#     metadata_json=json.dumps(metadata),
#
#     # Time
#     time=time,
#
#     # Main trajectories
#     p_hist=p_hist,
#     v_hist=v_hist,
#     u_hist=u_hist,
#
#     # Target trajectory
#     pt_hist=pt_hist,
#     vt_hist=vt_hist,
#     at_hist=at_hist,
#
#     # Time-varying formation radius
#     R_form_hist=R_form_hist,
#
#     # Energy and alive status
#     energy_hist=energy_hist,
#     alive_hist=alive_hist,
#
#     # Leader and follower assignment histories
#     leaders_hist=leaders_hist,
#     g_hist=g_hist,
#     follower_pin_hist=follower_pin_hist,
#     leader_pin_hist=leader_pin_hist,
#     followers_per_leader_hist=followers_per_leader_hist,
#
#     # Existing diagnostics
#     leader_form_error_hist=leader_form_error_hist,
#     avg_follower_degree_hist=avg_follower_degree_hist,
#
#     # New case-study metrics
#     subgroup_formation_error_hist=subgroup_formation_error_hist,
#     subgroup_velocity_error_hist=subgroup_velocity_error_hist,
#     subgroup_cohesion_radius_hist=subgroup_cohesion_radius_hist,
#     subgroup_velocity_alignment_error_hist=subgroup_velocity_alignment_error_hist,
#     min_inter_agent_distance_hist=min_inter_agent_distance_hist,
#     subgroup_min_inter_agent_distance_hist=subgroup_min_inter_agent_distance_hist,
#     global_velocity_alignment_error_hist=global_velocity_alignment_error_hist,
#     global_cohesion_radius_hist=global_cohesion_radius_hist,
#
#     # Initial/final useful values
#     initial_energy=energy_hist[0],
#     final_energy=energy_hist[-1],
#     final_alive=alive_hist[-1],
#     final_leaders=leaders_hist[-1],
#     final_assignment=g_hist[-1],
# )
#
# print(f"Saved case-study data to: {case_study_file}")


# ============================================================
# Optional quick-check plots
# ============================================================

SHOW_QUICK_CHECK_PLOTS = True
SHOW_ANIMATION = True

if SHOW_QUICK_CHECK_PLOTS:
    # --------------------------------------------------------
    # Final snapshot with energy color map
    # --------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    final_alive = alive_hist[-1]
    final_leaders = leaders_hist[-1]
    final_leader_set = [x for x in final_leaders if x >= 0 and final_alive[x]]
    final_followers = [i for i in range(N) if final_alive[i] and i not in final_leader_set]

    final_p = p.copy()
    final_v = v.copy()
    final_pt = pt_hist[-1]
    final_energy = energy.copy()

    if len(final_followers) > 0:
        ax1.scatter(
            final_p[final_followers, 0],
            final_p[final_followers, 1],
            s=60,
            c=final_energy[final_followers],
            cmap=cmap_energy,
            norm=norm_energy,
            label="Followers",
        )

    if len(final_leader_set) > 0:
        ax1.scatter(
            final_p[final_leader_set, 0],
            final_p[final_leader_set, 1],
            s=120,
            marker="s",
            c=final_energy[final_leader_set],
            cmap=cmap_energy,
            norm=norm_energy,
            edgecolors="black",
            linewidths=0.8,
            label="Leaders",
        )

    ax1.scatter(final_pt[0], final_pt[1], s=160, marker="*", c="red", label="Target")

    circle = plt.Circle(
        (final_pt[0], final_pt[1]),
        R_form_hist[-1],
        fill=False,
        linestyle="--",
        alpha=0.7,
    )
    ax1.add_patch(circle)

    # Communication links in final snapshot.
    final_follower_neighbors = follower_neighbors_hist[-1]
    final_follower_pins = follower_pin_hist[-1]

    # Follower-follower links.
    drawn_pairs = set()
    for i in final_followers:
        for j in final_follower_neighbors.get(i, []):
            if j in final_followers and final_alive[j]:
                pair = tuple(sorted((i, j)))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)

                ax1.plot(
                    [final_p[i, 0], final_p[j, 0]],
                    [final_p[i, 1], final_p[j, 1]],
                    color="gray",
                    alpha=0.18,
                    linewidth=0.8,
                )

    # Follower-leader pinned links.
    for i in final_followers:
        if final_follower_pins[i] > 0:
            k = g[i]
            if k >= 0 and final_leaders[k] >= 0 and final_alive[final_leaders[k]]:
                Lk = final_leaders[k]
                ax1.plot(
                    [final_p[i, 0], final_p[Lk, 0]],
                    [final_p[i, 1], final_p[Lk, 1]],
                    color="red",
                    alpha=0.35,
                    linewidth=1.0,
                )

    # Leader ring links.
    for k in range(M):
        i1 = final_leaders[k]
        i2 = final_leaders[(k + 1) % M]
        if i1 >= 0 and i2 >= 0 and final_alive[i1] and final_alive[i2]:
            ax1.plot(
                [final_p[i1, 0], final_p[i2, 0]],
                [final_p[i1, 1], final_p[i2, 1]],
                "k--",
                alpha=0.4,
            )

    alive_indices = np.where(final_alive)[0]
    if len(alive_indices) > 0:
        ax1.quiver(
            final_p[alive_indices, 0],
            final_p[alive_indices, 1],
            final_v[alive_indices, 0],
            final_v[alive_indices, 1],
            angles="xy",
            scale_units="xy",
            scale=3.0,
            width=0.003,
            color="black",
            alpha=0.7,
        )

    sm1 = cm.ScalarMappable(cmap=cmap_energy, norm=norm_energy)
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1)
    cbar1.set_label("Energy (%)")

    ax1.set_title("Final snapshot with radius-based follower interaction")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.axis("equal")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Time-varying formation radius
    # --------------------------------------------------------
    fig_r, ax_r = plt.subplots(figsize=(12, 5))
    ax_r.plot(time, R_form_hist, linewidth=2.0)
    ax_r.set_title("Time-varying formation radius")
    ax_r.set_xlabel("Time [s]")
    ax_r.set_ylabel("R(t) [m]")
    ax_r.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Subgroup formation error
    # --------------------------------------------------------
    fig_fe, ax_fe = plt.subplots(figsize=(12, 6))
    for k in range(M):
        ax_fe.plot(time, subgroup_formation_error_hist[:, k], label=f"Subgroup {k}")
    ax_fe.set_title("Subgroup formation tracking error")
    ax_fe.set_xlabel("Time [s]")
    ax_fe.set_ylabel("Formation error [m]")
    ax_fe.grid(True, alpha=0.3)
    ax_fe.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Subgroup velocity tracking error
    # --------------------------------------------------------
    fig_ve, ax_ve = plt.subplots(figsize=(12, 6))
    for k in range(M):
        ax_ve.plot(time, subgroup_velocity_error_hist[:, k], label=f"Subgroup {k}")
    ax_ve.set_title("Subgroup leader velocity tracking error")
    ax_ve.set_xlabel("Time [s]")
    ax_ve.set_ylabel("Velocity error [m/s]")
    ax_ve.grid(True, alpha=0.3)
    ax_ve.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Subgroup cohesion radius
    # --------------------------------------------------------
    fig_cr, ax_cr = plt.subplots(figsize=(12, 6))
    for k in range(M):
        ax_cr.plot(time, subgroup_cohesion_radius_hist[:, k], label=f"Subgroup {k}")
    ax_cr.set_title("Subgroup cohesion radius")
    ax_cr.set_xlabel("Time [s]")
    ax_cr.set_ylabel("Cohesion radius [m]")
    ax_cr.grid(True, alpha=0.3)
    ax_cr.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Subgroup velocity alignment error
    # --------------------------------------------------------
    fig_va, ax_va = plt.subplots(figsize=(12, 6))
    for k in range(M):
        ax_va.plot(time, subgroup_velocity_alignment_error_hist[:, k], label=f"Subgroup {k}")
    ax_va.set_title("Subgroup follower velocity alignment error")
    ax_va.set_xlabel("Time [s]")
    ax_va.set_ylabel("Velocity alignment error [m/s]")
    ax_va.grid(True, alpha=0.3)
    ax_va.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Global minimum inter-agent distance
    # --------------------------------------------------------
    fig_md, ax_md = plt.subplots(figsize=(12, 5))
    ax_md.plot(time, min_inter_agent_distance_hist, linewidth=2.0)
    ax_md.axhline(d_min, color="black", linestyle="--", linewidth=1.2, label=r"$d_{min}$")
    ax_md.axhline(d_act, color="black", linestyle=":", linewidth=1.2, label=r"$d_{act}$")
    ax_md.set_title("Minimum inter-agent distance")
    ax_md.set_xlabel("Time [s]")
    ax_md.set_ylabel("Distance [m]")
    ax_md.grid(True, alpha=0.3)
    ax_md.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Final follower allocation bar chart
    # --------------------------------------------------------
    fig_alloc, ax_alloc = plt.subplots(figsize=(10, 5))
    final_counts = followers_per_leader_hist[-1]
    bars = ax_alloc.bar(np.arange(M), final_counts)
    ax_alloc.set_title("Final follower allocation per leader slot")
    ax_alloc.set_xlabel("Leader slot")
    ax_alloc.set_ylabel("Number of followers")
    ax_alloc.set_xticks(np.arange(M))
    ax_alloc.set_xticklabels([
        f"k={k}\nagent {final_leaders[k]}" if final_leaders[k] >= 0 else f"k={k}\nempty"
        for k in range(M)
    ])
    ax_alloc.grid(True, axis="y", alpha=0.3)

    for k, b in enumerate(bars):
        ax_alloc.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{final_counts[k]}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


# ============================================================
# Optional animation
# ============================================================

if SHOW_ANIMATION:
    fig, ax = plt.subplots(figsize=(8, 8))

    follower_sc = ax.scatter(
        [np.nan],
        [np.nan],
        s=50,
        c=[0.0],
        cmap=cmap_energy,
        norm=norm_energy,
        label="Followers",
    )

    leader_sc = ax.scatter(
        [np.nan],
        [np.nan],
        s=120,
        marker="s",
        c=[0.0],
        cmap=cmap_energy,
        norm=norm_energy,
        edgecolors="black",
        linewidths=0.8,
        label="Leaders",
    )

    target_sc = ax.scatter([], [], s=160, marker="*", c="red", label="Target")

    leader_lines = [ax.plot([], [], "k--", alpha=0.4)[0] for _ in range(M)]

    max_comm_lines = N * (N - 1) // 2
    comm_lines = [
        ax.plot([], [], color="gray", alpha=0.18, linewidth=0.8)[0]
        for _ in range(max_comm_lines)
    ]
    pin_lines = [
        ax.plot([], [], color="red", alpha=0.35, linewidth=1.0)[0]
        for _ in range(N)
    ]

    vel_lines = [ax.plot([], [], "k-", lw=1.2, alpha=0.9)[0] for _ in range(N)]

    ring_patch = plt.Circle(
        (0, 0),
        R_form_hist[0],
        fill=False,
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(ring_patch)

    finite_x = p_hist[:, :, 0][~np.isnan(p_hist[:, :, 0])]
    finite_y = p_hist[:, :, 1][~np.isnan(p_hist[:, :, 1])]

    if len(finite_x) > 0 and len(finite_y) > 0:
        ax.set_xlim(np.min(finite_x) - 5, np.max(finite_x) + 5)
        ax.set_ylim(np.min(finite_y) - 5, np.max(finite_y) + 5)
    else:
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

    sm = cm.ScalarMappable(cmap=cmap_energy, norm=norm_energy)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Energy (%)")

    def init():
        follower_sc.set_offsets(np.array([[np.nan, np.nan]]))
        follower_sc.set_array(np.array([0.0]))

        leader_sc.set_offsets(np.array([[np.nan, np.nan]]))
        leader_sc.set_array(np.array([0.0]))

        target_sc.set_offsets(np.empty((0, 2)))

        ring_patch.center = (0.0, 0.0)
        ring_patch.set_radius(R_form_hist[0])

        for ln in leader_lines + comm_lines + pin_lines + vel_lines:
            ln.set_data([], [])

        return [follower_sc, leader_sc, target_sc, ring_patch] + leader_lines + comm_lines + pin_lines + vel_lines

    def update(frame):
        pp = p_hist[frame]
        vv = v_hist[frame]
        pt = pt_hist[frame]
        alive_now = alive_hist[frame]
        leaders_now = leaders_hist[frame]
        g_now = g_hist[frame]
        ee = energy_hist[frame]

        leader_set_now = [x for x in leaders_now if x >= 0 and alive_now[x]]
        follower_set_now = [i for i in range(N) if alive_now[i] and i not in leader_set_now]

        # Followers.
        if len(follower_set_now) > 0:
            pos_f = pp[follower_set_now]
            en_f = ee[follower_set_now]
            valid_f = (~np.isnan(pos_f).any(axis=1)) & (~np.isnan(en_f))
            pos_f = pos_f[valid_f]
            en_f = en_f[valid_f]

            if len(pos_f) > 0:
                follower_sc.set_offsets(pos_f)
                follower_sc.set_array(en_f)
            else:
                follower_sc.set_offsets(np.array([[np.nan, np.nan]]))
                follower_sc.set_array(np.array([0.0]))
        else:
            follower_sc.set_offsets(np.array([[np.nan, np.nan]]))
            follower_sc.set_array(np.array([0.0]))

        # Leaders.
        if len(leader_set_now) > 0:
            pos_l = pp[leader_set_now]
            en_l = ee[leader_set_now]
            valid_l = (~np.isnan(pos_l).any(axis=1)) & (~np.isnan(en_l))
            pos_l = pos_l[valid_l]
            en_l = en_l[valid_l]

            if len(pos_l) > 0:
                leader_sc.set_offsets(pos_l)
                leader_sc.set_array(en_l)
            else:
                leader_sc.set_offsets(np.array([[np.nan, np.nan]]))
                leader_sc.set_array(np.array([0.0]))
        else:
            leader_sc.set_offsets(np.array([[np.nan, np.nan]]))
            leader_sc.set_array(np.array([0.0]))

        # Target and time-varying ring.
        target_sc.set_offsets(pt.reshape(1, 2))
        ring_patch.center = (pt[0], pt[1])
        ring_patch.set_radius(R_form_hist[frame])

        # Leader connection lines.
        for k in range(M):
            i1 = leaders_now[k]
            i2 = leaders_now[(k + 1) % M]
            if (
                i1 >= 0 and i2 >= 0
                and alive_now[i1] and alive_now[i2]
                and not np.any(np.isnan(pp[i1]))
                and not np.any(np.isnan(pp[i2]))
            ):
                leader_lines[k].set_data(
                    [pp[i1, 0], pp[i2, 0]],
                    [pp[i1, 1], pp[i2, 1]],
                )
            else:
                leader_lines[k].set_data([], [])

        # Communication graph for this frame.
        follower_neighbors_now = follower_neighbors_hist[frame]
        follower_pins_now = follower_pin_hist[frame]

        for ln in comm_lines + pin_lines:
            ln.set_data([], [])

        # Follower-follower communication links.
        drawn_pairs = set()
        line_idx = 0
        for i in follower_set_now:
            for j in follower_neighbors_now.get(i, []):
                if j not in follower_set_now:
                    continue
                if not alive_now[j]:
                    continue
                if np.any(np.isnan(pp[i])) or np.any(np.isnan(pp[j])):
                    continue

                pair = tuple(sorted((i, j)))
                if pair in drawn_pairs:
                    continue
                drawn_pairs.add(pair)

                if line_idx < len(comm_lines):
                    comm_lines[line_idx].set_data(
                        [pp[i, 0], pp[j, 0]],
                        [pp[i, 1], pp[j, 1]],
                    )
                    line_idx += 1

        # Follower-leader pinned communication links.
        pin_idx = 0
        for i in follower_set_now:
            if follower_pins_now[i] <= 0:
                continue

            k = g_now[i]
            if k < 0 or k >= M:
                continue

            leader_i = leaders_now[k]
            if leader_i < 0 or not alive_now[leader_i]:
                continue

            if np.any(np.isnan(pp[i])) or np.any(np.isnan(pp[leader_i])):
                continue

            if pin_idx < len(pin_lines):
                pin_lines[pin_idx].set_data(
                    [pp[i, 0], pp[leader_i, 0]],
                    [pp[i, 1], pp[leader_i, 1]],
                )
                pin_idx += 1

        # Velocity direction lines.
        arrow_gain = 0.6
        for i in range(N):
            if alive_now[i] and not np.any(np.isnan(pp[i])) and not np.any(np.isnan(vv[i])):
                x0, y0 = pp[i]
                dx, dy = arrow_gain * vv[i]
                vel_lines[i].set_data([x0, x0 + dx], [y0, y0 + dy])
            else:
                vel_lines[i].set_data([], [])

        ax.set_title(
            f"t = {frame * dt:.2f} s, formation radius = {R_form_hist[frame]:.2f}, "
            f"avg follower degree = {avg_follower_degree_hist[frame]:.2f}"
        )

        return [follower_sc, leader_sc, target_sc, ring_patch] + leader_lines + comm_lines + pin_lines + vel_lines

    ani = FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        interval=30,
        blit=False,
    )

    plt.tight_layout()
    plt.show()
