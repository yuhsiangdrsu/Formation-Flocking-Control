import numpy as np
from scipy.linalg import solve_continuous_are
import pickle


# ============================================================
# 3D cubic formation control: data generation only
# ============================================================

np.random.seed(10)

# ============================================================
# Simulation parameters
# ============================================================
N = 20
dt = 0.01
T = 20.0
steps = int(T / dt)
time = np.arange(steps) * dt

MC_trials = 50

a_max = 10.0

pin_update_every = 20

k_pin = 1.0
k_graph = 1.0

threshold_abs = 1.0

adaptive_pin_numbers = [4, 6, 8,10,12,16,18]


# ============================================================
# Target motion
# ============================================================
target0 = np.array([0.0, 0.0, 0.0])
v_target = np.array([0.4, 0.25, 0.15])
a_target = np.array([0.0, 0.0, 0.0])


def target_position(t):
    return target0 + v_target * t


# ============================================================
# 3D double-integrator LQR design
# ============================================================
A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))]
])

B = np.block([
    [np.zeros((3, 3))],
    [np.eye(3)]
])

Q = np.diag([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
R = np.diag([1.0, 1.0, 1.0])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)


# ============================================================
# Desired 20-agent cubic formation
# ============================================================
def cubic_formation_offsets(half_size=4.0):
    a = half_size

    corners = np.array([
        [-a, -a, -a],
        [-a, -a,  a],
        [-a,  a, -a],
        [-a,  a,  a],
        [ a, -a, -a],
        [ a, -a,  a],
        [ a,  a, -a],
        [ a,  a,  a],
    ])

    edge_midpoints = np.array([
        [0, -a, -a],
        [0, -a,  a],
        [0,  a, -a],
        [0,  a,  a],

        [-a, 0, -a],
        [-a, 0,  a],
        [ a, 0, -a],
        [ a, 0,  a],

        [-a, -a, 0],
        [-a,  a, 0],
        [ a, -a, 0],
        [ a,  a, 0],
    ])

    return np.vstack((corners, edge_midpoints))


half_size = 4.0
r_star = cubic_formation_offsets(half_size=half_size)


# ============================================================
# Interaction graph
# ============================================================
def build_cube_graph(half_size=4.0):
    neighbors = {i: [] for i in range(N)}

    desired_p = r_star.copy()
    neighbor_distance = half_size
    tol = 1e-6

    for i in range(N):
        for j in range(i + 1, N):
            dij = np.linalg.norm(desired_p[i] - desired_p[j])

            if abs(dij - neighbor_distance) < tol:
                neighbors[i].append(j)
                neighbors[j].append(i)

    return neighbors


neighbors = build_cube_graph(half_size=half_size)


# ============================================================
# Fixed pinning setup
# ============================================================
fixed_pinned_agents = np.array([0, 1, 2, 3])


# ============================================================
# Helper functions
# ============================================================
def sat_vec(vec, vmax):
    n = np.linalg.norm(vec)

    if n <= vmax or n < 1e-12:
        return vec

    return vmax * vec / n


def local_formation_mismatch(i, p):
    delta_i = 0.0

    for j in neighbors[i]:
        actual_rel = p[i] - p[j]
        desired_rel = r_star[i] - r_star[j]

        delta_i += np.linalg.norm(actual_rel - desired_rel)

    return delta_i


def choose_adaptive_pins(p, n_pin):
    mismatches = np.array([
        local_formation_mismatch(i, p)
        for i in range(N)
    ])

    pinned = np.argsort(-mismatches)[:n_pin]

    return pinned, mismatches


def choose_fixed_pins():
    return fixed_pinned_agents.copy()


def formation_error_absolute(p, p_target):
    desired_p = p_target + r_star
    err = p - desired_p

    return np.sqrt(np.mean(np.sum(err ** 2, axis=1)))


def agent_position_errors(p, p_target):
    """
    Per-agent absolute formation position error.

    Output shape:
        (N,)

    e_i = || p_i - (p_target + r_i*) ||
    """

    desired_p = p_target + r_star
    err = p - desired_p

    return np.linalg.norm(err, axis=1)


def agent_velocity_errors(v, v_target_now):
    """
    Per-agent velocity tracking error.

    Output shape:
        (N,)

    ev_i = || v_i - v_target ||
    """

    err = v - v_target_now
    return np.linalg.norm(err, axis=1)


def generate_initial_condition():
    p_target0 = target_position(0.0)
    desired_p0 = p_target0 + r_star

    disturbance_scale = 10.0
    max_disturbance = 25.0

    disturbance = disturbance_scale * np.random.standard_t(
        df=2.0,
        size=(N, 3)
    )

    disturbance = np.clip(
        disturbance,
        -max_disturbance,
        max_disturbance
    )

    p0 = desired_p0 + disturbance

    return p0


def convergence_time(error, threshold, time):
    below = np.where(error <= threshold)[0]

    if len(below) == 0:
        return np.nan

    return time[below[0]]


# ============================================================
# One trial simulation
# ============================================================
def simulate_one_trial(p0, n_pin=4, pinning_mode="adaptive"):

    p = p0.copy()
    v = np.zeros((N, 3))

    abs_error_hist = np.zeros(steps)
    pinned_hist = np.zeros((steps, N))
    mismatch_hist = np.zeros((steps, N))

    p_hist = np.zeros((steps, N, 3))
    v_hist = np.zeros((steps, N, 3))
    target_hist = np.zeros((steps, 3))

    per_agent_pos_error_hist = np.zeros((steps, N))
    per_agent_vel_error_hist = np.zeros((steps, N))

    current_pinned_agents = None
    current_mismatches = np.zeros(N)

    for s in range(steps):
        t = s * dt

        p_target = target_position(t)
        v_target_now = v_target.copy()
        a_target_now = a_target.copy()

        # ----------------------------------------------------
        # Pin selection
        # ----------------------------------------------------
        if pinning_mode == "adaptive":

            if s == 0 or s % pin_update_every == 0:
                current_pinned_agents, current_mismatches = choose_adaptive_pins(
                    p,
                    n_pin
                )

            pinned_agents = current_pinned_agents

        elif pinning_mode == "fixed":

            pinned_agents = choose_fixed_pins()

            current_mismatches = np.array([
                local_formation_mismatch(i, p)
                for i in range(N)
            ])

        else:
            raise ValueError("pinning_mode must be 'adaptive' or 'fixed'.")

        pinned_set = set(pinned_agents.tolist())

        # ----------------------------------------------------
        # Formation controller
        # ----------------------------------------------------
        u = np.zeros((N, 3))

        for i in range(N):
            x_i = np.hstack((p[i], v[i]))
            x_i_star = np.hstack((p_target + r_star[i], v_target_now))

            b_i = 1.0 if i in pinned_set else 0.0

            e_pin = b_i * (x_i - x_i_star)

            e_graph = np.zeros(6)

            for j in neighbors[i]:
                x_j = np.hstack((p[j], v[j]))
                x_j_star = np.hstack((p_target + r_star[j], v_target_now))

                e_graph += (x_i - x_j) - (x_i_star - x_j_star)

            e_total = k_pin * e_pin + k_graph * e_graph

            u[i] = b_i * a_target_now - K @ e_total
            u[i] = sat_vec(u[i], a_max)

        # ----------------------------------------------------
        # Integrate dynamics
        # ----------------------------------------------------
        v = v + dt * u
        p = p + dt * v

        # ----------------------------------------------------
        # Record data
        # ----------------------------------------------------
        abs_error_hist[s] = formation_error_absolute(p, p_target)

        p_hist[s] = p
        v_hist[s] = v
        target_hist[s] = p_target

        mismatch_hist[s] = current_mismatches

        per_agent_pos_error_hist[s] = agent_position_errors(p, p_target)
        per_agent_vel_error_hist[s] = agent_velocity_errors(v, v_target_now)

        for i in pinned_agents:
            pinned_hist[s, i] = 1.0

    return {
        "abs_error": abs_error_hist,
        "pinned_hist": pinned_hist,
        "mismatch_hist": mismatch_hist,
        "p_hist": p_hist,
        "v_hist": v_hist,
        "target_hist": target_hist,
        "per_agent_pos_error": per_agent_pos_error_hist,
        "per_agent_vel_error": per_agent_vel_error_hist,
    }


# ============================================================
# Allocate storage
# ============================================================
adaptive_4_abs_errors = np.zeros((MC_trials, steps))
fixed_4_abs_errors = np.zeros((MC_trials, steps))

adaptive_multi_abs_errors = {
    n_pin: np.zeros((MC_trials, steps))
    for n_pin in adaptive_pin_numbers
}

adaptive_4_conv_times = np.zeros(MC_trials)
fixed_4_conv_times = np.zeros(MC_trials)

adaptive_multi_conv_times = {
    n_pin: np.zeros(MC_trials)
    for n_pin in adaptive_pin_numbers
}

example_adaptive_4 = None
example_fixed_4 = None

example_adaptive_multi = {
    n_pin: None
    for n_pin in adaptive_pin_numbers
}

initial_conditions = np.zeros((MC_trials, N, 3))


# ============================================================
# Monte Carlo simulation
# ============================================================
for trial in range(MC_trials):
    print(f"Running Monte Carlo trial {trial + 1}/{MC_trials}")

    p0 = generate_initial_condition()
    initial_conditions[trial] = p0

    # --------------------------------------------------------
    # Adaptive n_pin = 4
    # --------------------------------------------------------
    result_adaptive_4 = simulate_one_trial(
        p0,
        n_pin=4,
        pinning_mode="adaptive"
    )

    # --------------------------------------------------------
    # Fixed n_pin = 4
    # --------------------------------------------------------
    result_fixed_4 = simulate_one_trial(
        p0,
        n_pin=4,
        pinning_mode="fixed"
    )

    adaptive_4_abs_errors[trial] = result_adaptive_4["abs_error"]
    fixed_4_abs_errors[trial] = result_fixed_4["abs_error"]

    adaptive_4_conv_times[trial] = convergence_time(
        result_adaptive_4["abs_error"],
        threshold_abs,
        time
    )

    fixed_4_conv_times[trial] = convergence_time(
        result_fixed_4["abs_error"],
        threshold_abs,
        time
    )

    if trial == 0:
        example_adaptive_4 = result_adaptive_4
        example_fixed_4 = result_fixed_4

    # --------------------------------------------------------
    # Adaptive n_pin = 4, 8, 12
    # --------------------------------------------------------
    for n_pin in adaptive_pin_numbers:

        result_adaptive_n = simulate_one_trial(
            p0,
            n_pin=n_pin,
            pinning_mode="adaptive"
        )

        adaptive_multi_abs_errors[n_pin][trial] = result_adaptive_n["abs_error"]

        adaptive_multi_conv_times[n_pin][trial] = convergence_time(
            result_adaptive_n["abs_error"],
            threshold_abs,
            time
        )

        if trial == 0:
            example_adaptive_multi[n_pin] = result_adaptive_n


# ============================================================
# Compute statistics
# ============================================================
adaptive_4_mean = np.mean(adaptive_4_abs_errors, axis=0)
adaptive_4_std = np.std(adaptive_4_abs_errors, axis=0)

fixed_4_mean = np.mean(fixed_4_abs_errors, axis=0)
fixed_4_std = np.std(fixed_4_abs_errors, axis=0)

adaptive_multi_mean = {}
adaptive_multi_std = {}

for n_pin in adaptive_pin_numbers:
    adaptive_multi_mean[n_pin] = np.mean(
        adaptive_multi_abs_errors[n_pin],
        axis=0
    )

    adaptive_multi_std[n_pin] = np.std(
        adaptive_multi_abs_errors[n_pin],
        axis=0
    )


# ============================================================
# Save all data needed for plotting
# ============================================================
data = {
    "N": N,
    "dt": dt,
    "T": T,
    "steps": steps,
    "time": time,
    "MC_trials": MC_trials,
    "threshold_abs": threshold_abs,
    "adaptive_pin_numbers": adaptive_pin_numbers,

    "target0": target0,
    "v_target": v_target,
    "a_target": a_target,

    "half_size": half_size,
    "r_star": r_star,
    "neighbors": neighbors,
    "fixed_pinned_agents": fixed_pinned_agents,

    "initial_conditions": initial_conditions,

    "adaptive_4_abs_errors": adaptive_4_abs_errors,
    "fixed_4_abs_errors": fixed_4_abs_errors,

    "adaptive_4_mean": adaptive_4_mean,
    "adaptive_4_std": adaptive_4_std,
    "fixed_4_mean": fixed_4_mean,
    "fixed_4_std": fixed_4_std,

    "adaptive_4_conv_times": adaptive_4_conv_times,
    "fixed_4_conv_times": fixed_4_conv_times,

    "adaptive_multi_abs_errors": adaptive_multi_abs_errors,
    "adaptive_multi_mean": adaptive_multi_mean,
    "adaptive_multi_std": adaptive_multi_std,
    "adaptive_multi_conv_times": adaptive_multi_conv_times,

    "example_adaptive_4": example_adaptive_4,
    "example_fixed_4": example_fixed_4,
    "example_adaptive_multi": example_adaptive_multi,
}

output_file = "formation_control_data_2.pkl"

with open(output_file, "wb") as f:
    pickle.dump(data, f)

print("\nSaved data to:")
print(output_file)

print("\nFinished data generation.")