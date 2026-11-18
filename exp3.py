"""
Crazyflie + OptiTrack combined formation + flocking mission.

Updated version:
1. Keeps the two physical leader formation controller.
2. Keeps the associated follower flocking controller.
3. Uses the safer pre-takeoff workflow from the formation/flocking templates:
   - OptiTrack wait and stability checks
   - crash/tumble recovery
   - external position streaming
   - Kalman reset and estimator convergence wait
   - fixed-x/y position-setpoint takeoff
   - fixed-current-x/y position-setpoint landing
   - final stop and disarm

Mapping:
- cf0, cf1 are physical leaders.
- cf2, cf3, cf4 follow leader 0.
- cf5, cf6 follow leader 1.
"""

import time
import threading
import sys
import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.linalg import solve_continuous_are

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory, Swarm

# From NatNet SDK 4.4 PythonClient folder
from NatNetClient import NatNetClient


# ============================================================
# Crazyflie + OptiTrack two-leader associated-follower mission
# ------------------------------------------------------------
# 7 physical Crazyflies total:
#   - 2 physical leaders
#   - 5 physical followers
#
# Mapping:
#   cf0, cf1 = leaders
#   cf2, cf3, cf4 = followers of leader 0
#   cf5, cf6 = followers of leader 1
#
# Leaders:
#   - track desired rotating circular positions around a fixed target
#   - maintain 180-degree rotating leader formation
#   - are commanded with hover velocity commands in x-y and z
#
# Followers:
#   - flock with their associated physical leader
#   - avoid all other followers and both leaders in x-y
#   - use fixed hover height
#
# IMPORTANT FIRST-FLIGHT NOTES
#   1. Start with leader_radius <= 0.4 and MISSION_T <= 20.
#   2. Verify OptiTrack axes before flying.
#   3. Place all drones near their intended starting positions.
#   4. Keep a physical kill switch ready.
# ============================================================


# ============================================================
# USER SETTINGS
# ============================================================

SERVER_IP = "192.168.3.3"   # Motive / OptiTrack PC
CLIENT_IP = "192.168.3.5"   # Python / Crazyflie PC
USE_MULTICAST = True

# ------------------------------------------------------------
# Crazyflie radio URIs
# ------------------------------------------------------------
# IMPORTANT:
# Replace cf6 with the actual radio address of your extra drone.
# The values below follow the same numbering pattern as your original script,
# but you must confirm them with your Crazyradio/Crazyflie configuration.
# ------------------------------------------------------------
URIS = {
    "cf0": "radio://0/80/2M/E7E7E7E711",
    "cf1": "radio://0/80/2M/E7E7E7E712",
    "cf2": "radio://1/100/2M/E7E7E7E713",
    "cf3": "radio://1/100/2M/E7E7E7E714",
    "cf4": "radio://2/120/2M/E7E7E7E715",
    "cf5": "radio://2/120/2M/E7E7E7E716",
    "cf6": "radio://1/100/2M/E7E7E7E717",
}

CF_NAMES = list(URIS.keys())
URI_LIST = [URIS[name] for name in CF_NAMES]
URI_TO_NAME = {URIS[name]: name for name in CF_NAMES}
NAME_TO_INDEX = {name: i for i, name in enumerate(CF_NAMES)}

N_L = 2
N_F = 5
N_TOTAL = N_L + N_F

LEADER_AGENT_IDS = np.array([0, 1], dtype=int)          # cf0, cf1
FOLLOWER_AGENT_IDS = np.array([2, 3, 4, 5, 6], dtype=int)  # cf2..cf6

AGENT_TO_NAME = {i: CF_NAMES[i] for i in range(N_TOTAL)}

# ------------------------------------------------------------
# OptiTrack rigid body mapping
# Motive rigid body ID -> physical agent index
# ------------------------------------------------------------
# IMPORTANT:
# Make sure Motive rigid body ID 7 exists and corresponds to cf6.
# ------------------------------------------------------------
RIGID_BODY_TO_AGENT = {
    1: 0,   # cf0, leader 0
    2: 1,   # cf1, leader 1
    3: 2,   # cf2, follower 0 -> leader 0
    4: 3,   # cf3, follower 1 -> leader 0
    5: 4,   # cf4, follower 2 -> leader 0
    6: 5,   # cf5, follower 3 -> leader 1
    7: 6,   # cf6, follower 4 -> leader 1
}


# ============================================================
# Leader-follower assignment, matching the simulation
# ============================================================

leader_followers = {
    0: [0, 1, 2],
    1: [3, 4],
}

# Local follower index 0..5 -> local leader index 0..1.
# Followers 0,1,2 follow Leader 0.
# Followers 3,4 follow Leader 1.
follower_leader = np.array([
    0,
    0,
    0,
    1,
    1,
], dtype=int)


# ============================================================
# Timing parameters
# ============================================================

EXTPOSE_DT = 0.01       # 50 Hz external pose stream
SEND_DT = 0.02          # 50 Hz takeoff / landing position setpoints
CONTROL_DT = 0.05       # 100 Hz controller

TAKEOFF_TIME = 2.0
LANDING_TIME = 3.0
MISSION_T = 30.0        # Start conservative. Increase after testing.
TAKEOFF_HOLD_TIME = 1.0

# Estimator settings copied from the safer formation/flocking templates.
ESTIMATOR_TIMEOUT = 10.0
ESTIMATOR_LOG_PERIOD_MS = 100
ESTIMATOR_SPREAD_LIMIT = 0.001

# Position-setpoint conversion used during the combined mission.
K_VEL_TO_POS = 1.0
MAX_POSITION_STEP_XY = 0.1
POSITION_SETPOINT_YAW = 0.0

# ============================================================
# Data logging settings
# ============================================================

SAVE_FLIGHT_LOGS = True
LOG_DIR = "flight_logs"
LOG_FILE = "two_leader_mission_log.csv"


# ============================================================
# Flight and safety parameters
# ============================================================

takeoff_height = 0.7
landing_height = 0.05
leader_flight_z = 1
follower_flight_z = 0.5

workspace_min = np.array([0, -2.5, -0.01])
workspace_max = np.array([ 3,  0.5, 2.0])

lost_tracking_timeout = 0.25

# First-flight conservative velocity/acceleration limits.
a_max_leader = 100
a_max_follower = 100

v_max_leader_xy = 100
v_max_leader_z = 100
v_max_follower_xy = 100

hover_yawrate = 0.0


# ============================================================
# Two-leader rotating target formation
# ============================================================

# Keep this radius small for the real flight volume.
target_pos = np.array([1.26, -1.13, leader_flight_z])
target_vel = np.array([0.0, 0.0, 0.0])
target_acc = np.array([0.0, 0.0, 0.0])

leader_radius = 0.65      # Simulation used 0.8. Start smaller on hardware.
leader_omega = 0.2      # Simulation used 0.2. Start slower on hardware.


def desired_leader_formation(t):
    """
    Desired two-leader formation around fixed target.

    Leader 0 and Leader 1 stay 180 degrees apart.
    """

    p_star = np.zeros((N_L, 3))
    v_star = np.zeros((N_L, 3))
    a_star = np.zeros((N_L, 3))

    for i in range(N_L):
        theta = leader_omega * t + np.pi * i

        p_star[i] = target_pos + np.array([
            leader_radius * np.cos(theta),
            leader_radius * np.sin(theta),
            0.0,
        ])

        v_star[i] = target_vel + np.array([
            -leader_radius * leader_omega * np.sin(theta),
             leader_radius * leader_omega * np.cos(theta),
             0.0,
        ])

        a_star[i] = target_acc + np.array([
            -leader_radius * leader_omega**2 * np.cos(theta),
            -leader_radius * leader_omega**2 * np.sin(theta),
             0.0,
        ])

    return p_star, v_star, a_star


# ============================================================
# Controller parameters from simulation, made conservative
# ============================================================

# Leader controller
k_target = 400.0
k_form = 1.0

# Follower controller
R_interaction = 2.0
sigma_f = 5.0
pinning_gain = 40.0
normalize_follower_weights = True
eps_w = 1e-8

# Collision avoidance.
d_min = 0.25
d_act = 0.6
k_phi = 0.2
eps_phi = 1e-6


# ============================================================
# LQR design: leader 3D double integrator
# State: [x, y, z, vx, vy, vz]
# Input: [ax, ay, az]
# ============================================================

A_leader = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))],
])

B_leader = np.block([
    [np.zeros((3, 3))],
    [np.eye(3)],
])

Q_leader =  np.diag([1, 1, 1, 0.2, 0.2, 0.2])
R_leader = np.diag([0.01, 0.01, 0.01])

P_leader = solve_continuous_are(A_leader, B_leader, Q_leader, R_leader)
K_leader = np.linalg.solve(R_leader, B_leader.T @ P_leader)

print("Leader LQR gain K_leader:")
print(K_leader)


# ============================================================
# LQR design: follower x-y double integrator
# State: [x, y, vx, vy]
# Input: [ax, ay]
# ============================================================

A_follower = np.array([
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
])

B_follower = np.array([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

Q_follower = np.diag([0.4, 0.4, 0.1, 0.1])
R_follower_lqr = np.diag([0.01, 0.01])

P_follower = solve_continuous_are(A_follower, B_follower, Q_follower, R_follower_lqr)
K_follower = np.linalg.solve(R_follower_lqr, B_follower.T @ P_follower)

print("Follower LQR gain K_follower:")
print(K_follower)


# ============================================================
# General helpers
# ============================================================

def sat_vec(vec, vmax):
    n = np.linalg.norm(vec)
    if n <= vmax or n < 1e-12:
        return vec
    return vmax * vec / n


def clip_leader_velocity(v):
    v_out = v.copy()
    xy_norm = np.linalg.norm(v_out[:2])
    if xy_norm > v_max_leader_xy:
        v_out[:2] = v_out[:2] * v_max_leader_xy / xy_norm
    v_out[2] = np.clip(v_out[2], -v_max_leader_z, v_max_leader_z)
    return v_out


def clip_follower_v_xy(v_xy):
    out = v_xy.copy()
    for i in range(out.shape[0]):
        n = np.linalg.norm(out[i])
        if n > v_max_follower_xy:
            out[i] = out[i] * v_max_follower_xy / n
    return out


def inside_workspace(p):
    return np.all(p >= workspace_min) and np.all(p <= workspace_max)


def split_states(p_all, v_all):
    p_L = p_all[LEADER_AGENT_IDS]
    v_L = v_all[LEADER_AGENT_IDS]
    p_F = p_all[FOLLOWER_AGENT_IDS]
    v_F = v_all[FOLLOWER_AGENT_IDS]
    return p_L, v_L, p_F, v_F


def merge_velocity_commands(v_L_cmd, v_F_xy_cmd):
    """
    Build per-physical-agent hover velocity commands.

    Leaders use full [vx, vy, vz]. Followers use [vx, vy, 0].
    """

    v_cmd = np.zeros((N_TOTAL, 3))

    for local_leader_id, agent_id in enumerate(LEADER_AGENT_IDS):
        v_cmd[agent_id] = v_L_cmd[local_leader_id]

    for local_follower_id, agent_id in enumerate(FOLLOWER_AGENT_IDS):
        v_cmd[agent_id, :2] = v_F_xy_cmd[local_follower_id]
        v_cmd[agent_id, 2] = 0.0

    return v_cmd


# ============================================================
# Follower-follower distance graph
# ============================================================

def build_follower_neighbors(p_F):
    """
    Build follower-follower neighbors in x-y only.
    """

    follower_neighbors = {i: [] for i in range(N_F)}

    for i in range(N_F):
        for j in range(N_F):
            if i == j:
                continue

            dist_ij = np.linalg.norm(p_F[i, :2] - p_F[j, :2])
            if dist_ij <= R_interaction:
                follower_neighbors[i].append(j)

    return follower_neighbors


# ============================================================
# Collision avoidance
# ============================================================

def compute_potential_avoidance_xy(i, p_F, p_L):
    """
    Repulsive potential collision avoidance for follower i in x-y.

    Each follower avoids:
        1. all other followers
        2. both physical leaders
    """

    u_avoid = np.zeros(2)
    c_act = 1.0 / max(d_act - d_min, eps_phi)
    p_i_xy = p_F[i, :2]

    # Avoid all other followers.
    for j in range(N_F):
        if j == i:
            continue

        z = p_i_xy - p_F[j, :2]
        r = np.linalg.norm(z)

        if r >= d_act or r < 1e-12:
            continue

        gap = max(r - d_min, eps_phi)
        z_hat = z / r
        s = (1.0 / gap) - c_act
        f_mag = k_phi * s / (gap**2)
        u_avoid += f_mag * z_hat

    # Avoid both leaders.
    for ell in range(N_L):
        z = p_i_xy - p_L[ell, :2]
        r = np.linalg.norm(z)

        if r >= d_act or r < 1e-12:
            continue

        gap = max(r - d_min, eps_phi)
        z_hat = z / r
        s = (1.0 / gap) - c_act
        f_mag = k_phi * s / (gap**2)
        u_avoid += f_mag * z_hat

    return u_avoid


# ============================================================
# Leader controller
# ============================================================

def compute_leader_control(p_L, v_L, t):
    """
    Physical leader controller.

    u_L_i = a_i^* - K_leader [
                k_target (x_i - x_i^*)
                + k_form sum_j ((x_i - x_j) - (x_i^* - x_j^*))
            ]
    """

    p_star, v_star, a_star = desired_leader_formation(t)
    u_L = np.zeros((N_L, 3))

    for i in range(N_L):
        x_i = np.hstack((p_L[i], v_L[i]))
        x_i_star = np.hstack((p_star[i], v_star[i]))

        e_track = x_i - x_i_star
        e_form = np.zeros(6)

        for j in range(N_L):
            if j == i:
                continue

            x_j = np.hstack((p_L[j], v_L[j]))
            x_j_star = np.hstack((p_star[j], v_star[j]))
            e_form += (x_i - x_j) - (x_i_star - x_j_star)

        e_total = k_target * e_track + k_form * e_form
        u_L[i] = a_star[i] - K_leader @ e_total
        u_L[i] = sat_vec(u_L[i], a_max_leader)

    return u_L, p_star, v_star, a_star


# ============================================================
# Follower controller
# ============================================================

def compute_follower_control(p_F, v_F, p_L, v_L, u_L):
    """
    Physical follower controller in x-y.

    u_F_i_xy = a_ref_xy
               - K_follower [
                    pinning_gain (x_i_xy - x_ref_xy)
                    + sum_j w_ij (x_i_xy - x_j_xy)
                 ]
               + u_avoid_i_xy
    """

    u_F_xy = np.zeros((N_F, 2))
    follower_neighbors = build_follower_neighbors(p_F)

    for i in range(N_F):
        leader_id = follower_leader[i]

        p_ref = p_L[leader_id]
        v_ref = v_L[leader_id]
        a_ref = u_L[leader_id]

        x_i_xy = np.hstack((p_F[i, :2], v_F[i, :2]))
        x_ref_xy = np.hstack((p_ref[:2], v_ref[:2]))

        e_leader = x_i_xy - x_ref_xy
        e_graph = np.zeros(4)
        neigh_i = follower_neighbors[i]

        if len(neigh_i) > 0:
            weights = []

            for j in neigh_i:
                dist_ij = np.linalg.norm(p_F[i, :2] - p_F[j, :2])
                w_ij = np.exp(-(dist_ij**2) / (sigma_f**2))
                weights.append(w_ij)

            weights = np.array(weights, dtype=float)

            if normalize_follower_weights:
                weights = weights / (np.sum(weights) + eps_w)

            for w_ij, j in zip(weights, neigh_i):
                x_j_xy = np.hstack((p_F[j, :2], v_F[j, :2]))
                e_graph += w_ij * (x_i_xy - x_j_xy)

        u_avoid_xy = compute_potential_avoidance_xy(i, p_F, p_L)

        u_xy = (
            a_ref[:2]
            - K_follower @ (pinning_gain * e_leader + e_graph)
            + u_avoid_xy
        )

        u_xy = sat_vec(u_xy, a_max_follower)
        u_F_xy[i] = u_xy

    return u_F_xy


# ============================================================
# Metrics
# ============================================================

def compute_metrics(p_L, v_L, p_F, v_F, t):
    p_L_star, v_L_star, _ = desired_leader_formation(t)

    leader_pos_err = np.sqrt(np.mean(np.sum((p_L - p_L_star) ** 2, axis=1)))
    leader_vel_err = np.sqrt(np.mean(np.sum((v_L - v_L_star) ** 2, axis=1)))

    follower_leader_pos_err = np.zeros(N_F)
    follower_leader_vel_err = np.zeros(N_F)

    for i in range(N_F):
        leader_id = follower_leader[i]
        follower_leader_pos_err[i] = np.linalg.norm(p_F[i, :2] - p_L[leader_id, :2])
        follower_leader_vel_err[i] = np.linalg.norm(v_F[i, :2] - v_L[leader_id, :2])

    distances = []
    for i in range(N_F):
        for j in range(i + 1, N_F):
            distances.append(np.linalg.norm(p_F[i, :2] - p_F[j, :2]))

    min_follower_distance = float(np.min(distances)) if distances else float("inf")

    follower_leader_distances = []
    for i in range(N_F):
        for ell in range(N_L):
            follower_leader_distances.append(np.linalg.norm(p_F[i, :2] - p_L[ell, :2]))

    min_follower_leader_distance = float(np.min(follower_leader_distances))

    return (
        float(leader_pos_err),
        float(leader_vel_err),
        float(np.mean(follower_leader_pos_err)),
        float(np.mean(follower_leader_vel_err)),
        min_follower_distance,
        min_follower_leader_distance,
    )


# ============================================================
# OptiTrack shared state
# ============================================================

class MocapState:
    def __init__(self, n_agents):
        self.lock = threading.Lock()
        self.p = np.zeros((n_agents, 3))
        self.v = np.zeros((n_agents, 3))
        self.q = np.zeros((n_agents, 4))
        self.q[:, 3] = 1.0
        self.prev_p = np.zeros((n_agents, 3))
        self.prev_time = np.zeros(n_agents)
        self.valid = np.zeros(n_agents, dtype=bool)
        self.last_update = np.zeros(n_agents)

    def update_agent(self, agent_id, position, rotation, timestamp):
        position = np.array(position, dtype=float)
        rotation = np.array(rotation, dtype=float)

        with self.lock:
            if self.valid[agent_id]:
                dt = timestamp - self.prev_time[agent_id]
                if dt > 1e-4:
                    raw_v = (position - self.prev_p[agent_id]) / dt
                    alpha = 0.35
                    self.v[agent_id] = alpha * raw_v + (1.0 - alpha) * self.v[agent_id]

            self.p[agent_id] = position
            self.q[agent_id] = rotation
            self.prev_p[agent_id] = position
            self.prev_time[agent_id] = timestamp
            self.last_update[agent_id] = timestamp
            self.valid[agent_id] = True

    def get(self):
        with self.lock:
            return (
                self.p.copy(),
                self.v.copy(),
                self.q.copy(),
                self.valid.copy(),
                self.last_update.copy(),
            )

    def get_agent(self, agent_id):
        with self.lock:
            return {
                "p": self.p[agent_id].copy(),
                "v": self.v[agent_id].copy(),
                "q": self.q[agent_id].copy(),
                "valid": bool(self.valid[agent_id]),
                "last_update": float(self.last_update[agent_id]),
            }


mocap = MocapState(N_TOTAL)


# ============================================================
# NatNet callbacks and connection
# ============================================================

def receive_new_frame(data_dict):
    # Required by NatNetClient. We only use rigid body callbacks.
    pass


def receive_rigid_body_frame(rigid_body_id, position, rotation):
    if rigid_body_id not in RIGID_BODY_TO_AGENT:
        return

    agent_id = RIGID_BODY_TO_AGENT[rigid_body_id]
    mocap.update_agent(agent_id, position, rotation, time.time())


def start_natnet():
    client = NatNetClient()
    client.set_client_address(CLIENT_IP)
    client.set_server_address(SERVER_IP)
    client.set_use_multicast(USE_MULTICAST)
    client.set_print_level(0)
    client.new_frame_listener = receive_new_frame
    client.rigid_body_listener = receive_rigid_body_frame

    print("Starting NatNet client...")
    if not client.run("d"):
        print("ERROR: Could not start NatNet client.")
        sys.exit(1)

    time.sleep(1.0)

    try:
        if client.connected() is False:
            print("ERROR: Could not connect to Motive.")
            client.shutdown()
            sys.exit(2)
    except Exception:
        print("WARNING: Could not check NatNet connected() status.")

    print("NatNet connected.")
    return client


# ============================================================
# Pose waiting helpers
# ============================================================

def wait_for_all_poses(timeout=10.0):
    print("Waiting for all OptiTrack rigid bodies...")
    t0 = time.time()

    while time.time() - t0 < timeout:
        _, _, _, valid, _ = mocap.get()

        if np.all(valid):
            print("All OptiTrack poses received.")
            return

        print(f"Received {np.sum(valid)}/{N_TOTAL} rigid bodies...")
        time.sleep(0.5)

    raise RuntimeError("Not all OptiTrack rigid bodies were received.")


def wait_for_stable_poses(timeout=10.0, stable_time=1.0, max_age=0.2):
    print("Waiting for stable OptiTrack poses...")
    t0 = time.time()
    stable_start = None

    while time.time() - t0 < timeout:
        now = time.time()
        p, _, _, valid, last_update = mocap.get()
        fresh = np.all(now - last_update < max_age)

        if np.all(valid) and fresh:
            if stable_start is None:
                stable_start = now

            if now - stable_start >= stable_time:
                print("Stable OptiTrack poses confirmed.")
                for i, name in enumerate(CF_NAMES):
                    if i in LEADER_AGENT_IDS:
                        role = f"leader {np.where(LEADER_AGENT_IDS == i)[0][0]}"
                    else:
                        local_follower = int(np.where(FOLLOWER_AGENT_IDS == i)[0][0])
                        role = f"follower {local_follower} -> leader {follower_leader[local_follower]}"

                    print(
                        f"{name} ({role}): "
                        f"x={p[i, 0]:.3f}, y={p[i, 1]:.3f}, z={p[i, 2]:.3f}"
                    )
                return
        else:
            stable_start = None

        time.sleep(0.05)

    raise RuntimeError("Stable OptiTrack poses not confirmed.")


def check_initial_geometry():
    """
    Warn before takeoff if the physical layout is too close.
    This does not require exact simulation starting positions, but it catches unsafe spacing.
    """

    p_all, v_all, _, valid, _ = mocap.get()

    if not np.all(valid):
        raise RuntimeError("Cannot check initial geometry: missing OptiTrack poses.")

    p_L, _, p_F, _ = split_states(p_all, v_all)

    ff_distances = []
    for i in range(N_F):
        for j in range(i + 1, N_F):
            ff_distances.append(np.linalg.norm(p_F[i, :2] - p_F[j, :2]))

    fl_distances = []
    for i in range(N_F):
        for ell in range(N_L):
            fl_distances.append(np.linalg.norm(p_F[i, :2] - p_L[ell, :2]))

    min_ff = float(np.min(ff_distances)) if ff_distances else float("inf")
    min_fl = float(np.min(fl_distances)) if fl_distances else float("inf")

    print("\nInitial geometry check:")
    print(f"minimum follower-follower distance = {min_ff:.3f} m")
    print(f"minimum follower-leader distance   = {min_fl:.3f} m")
    print(f"d_min = {d_min:.3f} m")

    if min_ff <= d_min:
        raise RuntimeError("Initial follower-follower spacing is too small.")

    if min_fl <= d_min:
        raise RuntimeError("Initial follower-leader spacing is too small.")

    for i in range(N_TOTAL):
        if not inside_workspace(p_all[i]):
            raise RuntimeError(f"Agent {i} starts outside workspace: {p_all[i]}")



def check_tracking(context: str = "") -> None:
    now = time.time()
    _, _, _, valid, last_update = mocap.get()

    if not np.all(valid):
        raise RuntimeError(f"Not all agents have valid OptiTrack data {context}.")

    tracking_age = now - last_update
    if np.any(tracking_age > lost_tracking_timeout):
        raise RuntimeError(f"OptiTrack tracking timeout {context}.")


def capture_current_xy(label: str) -> np.ndarray:
    p_all, _, _, valid, _ = mocap.get()

    if not np.all(valid):
        raise RuntimeError(f"Cannot capture {label} x/y because not all poses are valid.")

    xy = p_all[:, 0:2].copy()
    print(f"{label} x/y positions captured:")
    for i, name in enumerate(CF_NAMES):
        print(f"{name}: x_{label}={xy[i, 0]:.3f}, y_{label}={xy[i, 1]:.3f}")

    return xy


# ============================================================
# Swarm setup functions
# ============================================================

def setup_estimator_and_reset(scf):
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Setting estimator and resetting Kalman...")
    cf.param.set_value("stabilizer.estimator", "2")
    time.sleep(2.0)

    # Low-level commander because this script sends setpoints directly.
    cf.param.set_value("commander.enHighLevel", "0")
    time.sleep(2.0)

    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(2.0)
    cf.param.set_value("kalman.resetEstimation", "0")
    time.sleep(2.0)

    print(f"{name}: Estimator reset command complete.")


def wait_for_position_estimator(scf, name: str, timeout: float = ESTIMATOR_TIMEOUT) -> None:
    cf = scf.cf
    log_config = LogConfig(name=f"{name}_kalman_var", period_in_ms=ESTIMATOR_LOG_PERIOD_MS)
    log_config.add_variable("kalman.varPX", "float")
    log_config.add_variable("kalman.varPY", "float")
    log_config.add_variable("kalman.varPZ", "float")

    var_x: List[float] = []
    var_y: List[float] = []
    var_z: List[float] = []

    def log_callback(timestamp, data, logconf):
        var_x.append(data["kalman.varPX"])
        var_y.append(data["kalman.varPY"])
        var_z.append(data["kalman.varPZ"])

    cf.log.add_config(log_config)
    log_config.data_received_cb.add_callback(log_callback)
    log_config.start()

    print(f"{name}: Waiting for estimator to converge...")
    t0 = time.time()
    ready = False

    try:
        while time.time() - t0 < timeout:
            if len(var_x) >= 10:
                dx = max(var_x[-10:]) - min(var_x[-10:])
                dy = max(var_y[-10:]) - min(var_y[-10:])
                dz = max(var_z[-10:]) - min(var_z[-10:])

                print(f"{name}: kalman variance spread dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")

                if dx < ESTIMATOR_SPREAD_LIMIT and dy < ESTIMATOR_SPREAD_LIMIT and dz < ESTIMATOR_SPREAD_LIMIT:
                    ready = True
                    break

            time.sleep(0.2)
    finally:
        log_config.stop()
        log_config.delete()

    if not ready:
        raise RuntimeError(f"{name}: estimator did not converge.")

    print(f"{name}: Estimator ready.")


def send_arming_request_compatible(cf, arm: bool) -> None:
    try:
        cf.supervisor.send_arming_request(arm)
    except AttributeError:
        cf.platform.send_arming_request(arm)


def recover_cf(scf) -> None:
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Sending crash recovery request...")

    try:
        cf.supervisor.send_crash_recovery_request()
    except AttributeError:
        cf.platform.send_crash_recovery_request()

    time.sleep(1.0)

    try:
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.2)
    except Exception:
        pass

    try:
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.05)
    except Exception:
        pass

    try:
        send_arming_request_compatible(cf, False)
        time.sleep(0.5)
    except Exception:
        pass

    print(f"{name}: Recovery request sent.")


def arm_cf(scf):
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Arming...")
    send_arming_request_compatible(cf, True)
    time.sleep(2.0)
    print(f"{name}: Arming request sent.")


# ============================================================
# External pose stream to Crazyflies
# ============================================================

extpose_running = False
extpose_threads = []
fixed_xy: Optional[np.ndarray] = None
landing_xy: Optional[np.ndarray] = None


def extpose_loop(scf):
    global extpose_running

    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]
    agent_id = NAME_TO_INDEX[name]

    print(f"{name}: External pose thread started.")

    while extpose_running:
        pose = mocap.get_agent(agent_id)
        now = time.time()

        if pose["valid"] and now - pose["last_update"] < 0.2:
            p = pose["p"]

            # Position-only external pose matches the safer templates and avoids
            # quaternion convention problems during first hardware tests.
            cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

            # If your estimator setup requires orientation, use:
            # q = pose["q"]
            # cf.extpos.send_extpose(
            #     float(p[0]), float(p[1]), float(p[2]),
            #     float(q[0]), float(q[1]), float(q[2]), float(q[3]),
            # )

        time.sleep(EXTPOSE_DT)

    print(f"{name}: External pose thread stopped.")


def start_extpose_thread(scf):
    t = threading.Thread(target=extpose_loop, args=(scf,), daemon=True)
    t.start()
    extpose_threads.append(t)


def stop_extpose_threads():
    global extpose_running

    extpose_running = False
    for t in extpose_threads:
        try:
            t.join(timeout=1.0)
        except Exception:
            pass


# ============================================================
# Position-setpoint takeoff and landing
# ============================================================

def send_position_for_duration(scf, x: float, y: float, z: float, duration: float, yaw: float = 0.0) -> None:
    cf = scf.cf
    steps = max(1, int(duration / SEND_DT))

    for _ in range(steps):
        cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw))
        time.sleep(SEND_DT)


def takeoff_one_cf(scf, name):
    if fixed_xy is None:
        raise RuntimeError("fixed_xy has not been captured before takeoff.")

    cf = scf.cf
    agent_id = NAME_TO_INDEX[name]
    pose = mocap.get_agent(agent_id)

    if not pose["valid"]:
        raise RuntimeError(f"{name}: No valid OptiTrack pose before takeoff.")

    x = fixed_xy[agent_id, 0]
    y = fixed_xy[agent_id, 1]
    start_z = max(0.0, pose["p"][2])
    steps = max(1, int(TAKEOFF_TIME / SEND_DT))

    print(f"{name}: Takeoff at fixed x/y x={x:.3f}, y={y:.3f}, z={start_z:.3f} -> {takeoff_height:.2f}")

    for k in range(steps):
        alpha = (k + 1) / steps
        z_cmd = start_z + alpha * (takeoff_height - start_z)
        cf.commander.send_position_setpoint(float(x), float(y), float(z_cmd), 0.0)
        time.sleep(SEND_DT)

    send_position_for_duration(scf, x, y, takeoff_height, duration=TAKEOFF_HOLD_TIME)
    print(f"{name}: Takeoff command sequence finished.")


def takeoff_sequence(swarm):
    print("Position-setpoint takeoff using fixed x/y spots...")
    swarm.parallel_safe(
        takeoff_one_cf,
        args_dict={URIS[name]: [name] for name in CF_NAMES},
    )
    print("All Crazyflies finished takeoff command sequence.")


def land_one_cf(scf, name):
    if landing_xy is None:
        raise RuntimeError("landing_xy has not been captured before landing.")

    cf = scf.cf
    agent_id = NAME_TO_INDEX[name]
    pose = mocap.get_agent(agent_id)

    if not pose["valid"]:
        raise RuntimeError(f"{name}: No valid OptiTrack pose before landing.")

    x = landing_xy[agent_id, 0]
    y = landing_xy[agent_id, 1]
    start_z = max(pose["p"][2], landing_height)
    steps = max(1, int(LANDING_TIME / SEND_DT))

    print(f"{name}: Landing at fixed current x/y x={x:.3f}, y={y:.3f}, z={start_z:.3f} -> {landing_height:.3f}")

    for k in range(steps):
        alpha = (k + 1) / steps
        z_cmd = start_z + alpha * (landing_height - start_z)
        cf.commander.send_position_setpoint(float(x), float(y), float(max(z_cmd, landing_height)), 0.0)
        time.sleep(SEND_DT)

    cf.commander.send_stop_setpoint()
    time.sleep(0.2)
    print(f"{name}: Landing command sequence finished.")


def landing_sequence(swarm):
    global landing_xy

    check_tracking(context="before landing")
    landing_xy = capture_current_xy("landing")

    print("Position-setpoint landing using fixed current x/y spots...")
    swarm.parallel_safe(
        land_one_cf,
        args_dict={URIS[name]: [name] for name in CF_NAMES},
    )
    print("All Crazyflies finished landing command sequence.")


# ============================================================
# Stop and disarm
# ============================================================

def stop_and_disarm(scf):
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Stopping and disarming...")

    try:
        cf.commander.send_notify_setpoint_stop()
        time.sleep(0.2)
    except Exception:
        pass

    try:
        for _ in range(10):
            cf.commander.send_stop_setpoint()
            time.sleep(0.05)
    except Exception:
        pass

    try:
        send_arming_request_compatible(cf, False)
        time.sleep(0.5)
    except Exception:
        pass

    print(f"{name}: Disarmed.")


def stop_all(swarm):
    swarm.parallel_safe(stop_and_disarm)



# ============================================================
# Data logging helpers
# ============================================================

def cohesion_radius_xy(points_xy: np.ndarray) -> float:
    """
    Cohesion radius in x-y:
        max distance from agents in the subgroup to that subgroup centroid.
    """
    if points_xy.shape[0] == 0:
        return float("nan")

    centroid = np.mean(points_xy, axis=0)
    return float(np.max(np.linalg.norm(points_xy - centroid, axis=1)))


def pairwise_distance_columns(p_all: np.ndarray) -> Dict[str, float]:
    """
    Return all unique inter-agent x-y and 3D distances.
    Example columns:
        dist_xy_cf0_cf1, dist_3d_cf0_cf1, ...
    """
    row = {}

    for i in range(N_TOTAL):
        for j in range(i + 1, N_TOTAL):
            name_i = CF_NAMES[i]
            name_j = CF_NAMES[j]

            row[f"dist_xy_{name_i}_{name_j}"] = float(
                np.linalg.norm(p_all[i, :2] - p_all[j, :2])
            )
            row[f"dist_3d_{name_i}_{name_j}"] = float(
                np.linalg.norm(p_all[i, :] - p_all[j, :])
            )

    return row


def save_two_leader_log(rows: List[dict]) -> None:
    """Save all recorded two-leader mission data to CSV."""
    if not SAVE_FLIGHT_LOGS:
        return

    if len(rows) == 0:
        print("No two-leader mission log rows to save.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, LOG_FILE)

    # Keep the header stable even if some rows get optional fields later.
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved two-leader mission log to: {csv_path}")


# ============================================================
# Crazyflie command helpers
# ============================================================

def send_position_to_cf(scf, position_command: np.ndarray) -> None:
    scf.cf.commander.send_position_setpoint(
        float(position_command[0]),
        float(position_command[1]),
        float(position_command[2]),
        float(POSITION_SETPOINT_YAW),
    )


def send_position_commands(swarm, position_cmd: np.ndarray) -> None:
    def _send(scf, position_command):
        send_position_to_cf(scf, position_command)

    swarm.parallel_safe(
        _send,
        args_dict={
            URIS[CF_NAMES[i]]: [position_cmd[i]]
            for i in range(N_TOTAL)
        },
    )


# Hover helpers are kept below for reference, but the mission now uses
# position setpoints like the safer formation/flocking scripts.

def send_hover_to_cf(scf, velocity_xyz):
    """
    Send Crazyflie hover command:
        vx, vy, yawrate, zDistance

    zDistance is absolute target height in hover mode.
    Leaders use fixed leader_flight_z.
    Followers use fixed follower_flight_z.
    """

    name = URI_TO_NAME[scf.cf.link_uri]
    agent_id = NAME_TO_INDEX[name]

    if agent_id in LEADER_AGENT_IDS:
        z_cmd = leader_flight_z
    else:
        z_cmd = follower_flight_z

    z_cmd = float(np.clip(z_cmd, workspace_min[2], workspace_max[2]))

    scf.cf.commander.send_hover_setpoint(
        float(velocity_xyz[0]),
        float(velocity_xyz[1]),
        float(hover_yawrate),
        z_cmd,
    )


def send_hover_commands(swarm, v_cmd_xyz):
    def _send(scf, velocity_xyz):
        send_hover_to_cf(scf, velocity_xyz)

    swarm.parallel_safe(
        _send,
        args_dict={
            URIS[CF_NAMES[i]]: [v_cmd_xyz[i]]
            for i in range(N_TOTAL)
        },
    )


# ============================================================
# Main two-leader associated-follower mission
# ============================================================

def run_two_leader_mission(swarm):
    print("Starting two-leader associated-follower Crazyflie mission...")

    mission_start = time.time()
    loop_counter = 0
    log_rows: List[dict] = []

    while True:
        now = time.time()
        t = now - mission_start

        if t >= MISSION_T:
            break

        p_all, v_all, _, valid, last_update = mocap.get()

        # ----------------------------------------------------
        # Safety checks
        # ----------------------------------------------------
        if not np.all(valid):
            raise RuntimeError("Not all agents have valid OptiTrack data.")

        tracking_age = now - last_update
        if np.any(tracking_age > lost_tracking_timeout):
            raise RuntimeError("OptiTrack tracking timeout.")

        for i in range(N_TOTAL):
            if not inside_workspace(p_all[i]):
                raise RuntimeError(f"Agent {i} left workspace: {p_all[i]}")

        p_L, v_L, p_F, v_F = split_states(p_all, v_all)

        p_star, _, _ = desired_leader_formation(t)
        for ell in range(N_L):
            if not inside_workspace(p_star[ell]):
                raise RuntimeError(f"Desired leader {ell} outside workspace: {p_star[ell]}")

        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------
        u_L, p_L_star, v_L_star, a_L_star = compute_leader_control(
            p_L=p_L,
            v_L=v_L,
            t=t,
        )

        u_F_xy = compute_follower_control(
            p_F=p_F,
            v_F=v_F,
            p_L=p_L,
            v_L=v_L,
            u_L=u_L,
        )

        # ----------------------------------------------------
        # Convert acceleration commands to position setpoints
        # ----------------------------------------------------
        v_L_cmd = np.zeros((N_L, 3))
        for i in range(N_L):
            v_L_cmd[i] = v_L[i] + CONTROL_DT * u_L[i]
            v_L_cmd[i] = clip_leader_velocity(v_L_cmd[i])

        v_F_cmd_xy = v_F[:, :2] + CONTROL_DT * u_F_xy
        v_F_cmd_xy = clip_follower_v_xy(v_F_cmd_xy)

        v_cmd_xyz = merge_velocity_commands(v_L_cmd, v_F_cmd_xy)

        position_cmd = np.zeros((N_TOTAL, 3))
        for i in range(N_TOTAL):
            delta_xy = K_VEL_TO_POS * CONTROL_DT * v_cmd_xyz[i, :2]
            delta_norm = np.linalg.norm(delta_xy)

            if delta_norm > MAX_POSITION_STEP_XY:
                delta_xy = delta_xy * MAX_POSITION_STEP_XY / delta_norm

            position_cmd[i, :2] = p_all[i, :2] + delta_xy

            if i in LEADER_AGENT_IDS:
                # Leaders fly at fixed leader height.
                z_cmd = leader_flight_z
            else:
                # Followers fly at fixed follower height.
                z_cmd = follower_flight_z

            position_cmd[i, 2] = float(np.clip(z_cmd, workspace_min[2], workspace_max[2]))

            if not inside_workspace(position_cmd[i]):
                raise RuntimeError(
                    f"{CF_NAMES[i]} position setpoint outside workspace: "
                    f"current={p_all[i]}, command={position_cmd[i]}"
                )

        send_position_commands(swarm, position_cmd)

        # ----------------------------------------------------
        # Metrics and row-by-row data logging
        # ----------------------------------------------------
        (
            leader_pos_err_mean,
            leader_vel_err_mean,
            follower_pos_err_mean,
            follower_vel_err_mean,
            min_ff,
            min_fl,
        ) = compute_metrics(p_L, v_L, p_F, v_F, t)

        # Leader formation error: actual leader-to-leader vector minus desired
        # leader-to-leader vector. With two leaders, this is the formation shape error.
        actual_leader_rel = p_L[0] - p_L[1]
        desired_leader_rel = p_L_star[0] - p_L_star[1]
        leader_pair_formation_error = float(np.linalg.norm(actual_leader_rel - desired_leader_rel))

        # Individual leader tracking errors.
        leader_pos_err_each = np.linalg.norm(p_L - p_L_star, axis=1)
        leader_vel_err_each = np.linalg.norm(v_L - v_L_star, axis=1)

        # Cohesion radius for each follower subgroup.
        # leader_followers stores local follower IDs, so index p_F directly.
        cohesion_by_subgroup = {}
        for leader_id, local_follower_ids in leader_followers.items():
            subgroup_xy = p_F[local_follower_ids, :2]
            cohesion_by_subgroup[f"cohesion_radius_xy_followers_of_leader{leader_id}"] = cohesion_radius_xy(subgroup_xy)

        row = {
            "time": float(t),
            "loop_counter": int(loop_counter),

            # Fixed target position and velocity.
            "target_x": float(target_pos[0]),
            "target_y": float(target_pos[1]),
            "target_z": float(target_pos[2]),
            "target_vx": float(target_vel[0]),
            "target_vy": float(target_vel[1]),
            "target_vz": float(target_vel[2]),

            # Summary metrics.
            "leader_pos_error_mean": float(leader_pos_err_mean),
            "leader_vel_error_mean": float(leader_vel_err_mean),
            "leader_pair_formation_error": float(leader_pair_formation_error),
            "follower_to_associated_leader_pos_error_mean": float(follower_pos_err_mean),
            "follower_to_associated_leader_vel_error_mean": float(follower_vel_err_mean),
            "min_follower_follower_distance_xy": float(min_ff),
            "min_follower_leader_distance_xy": float(min_fl),
        }

        row.update(cohesion_by_subgroup)
        row.update(pairwise_distance_columns(p_all))

        # Save all actual trajectories, measured velocities, desired leader
        # trajectories, commanded velocities, control inputs, and position setpoints.
        for agent_id, name in enumerate(CF_NAMES):
            role = "leader" if agent_id in LEADER_AGENT_IDS else "follower"

            row[f"{name}_role"] = role
            row[f"{name}_x"] = float(p_all[agent_id, 0])
            row[f"{name}_y"] = float(p_all[agent_id, 1])
            row[f"{name}_z"] = float(p_all[agent_id, 2])
            row[f"{name}_vx"] = float(v_all[agent_id, 0])
            row[f"{name}_vy"] = float(v_all[agent_id, 1])
            row[f"{name}_vz"] = float(v_all[agent_id, 2])
            row[f"{name}_cmd_vx"] = float(v_cmd_xyz[agent_id, 0])
            row[f"{name}_cmd_vy"] = float(v_cmd_xyz[agent_id, 1])
            row[f"{name}_cmd_vz"] = float(v_cmd_xyz[agent_id, 2])
            row[f"{name}_cmd_x"] = float(position_cmd[agent_id, 0])
            row[f"{name}_cmd_y"] = float(position_cmd[agent_id, 1])
            row[f"{name}_cmd_z"] = float(position_cmd[agent_id, 2])

            if agent_id in LEADER_AGENT_IDS:
                local_leader_id = int(np.where(LEADER_AGENT_IDS == agent_id)[0][0])

                row[f"{name}_desired_x"] = float(p_L_star[local_leader_id, 0])
                row[f"{name}_desired_y"] = float(p_L_star[local_leader_id, 1])
                row[f"{name}_desired_z"] = float(p_L_star[local_leader_id, 2])
                row[f"{name}_desired_vx"] = float(v_L_star[local_leader_id, 0])
                row[f"{name}_desired_vy"] = float(v_L_star[local_leader_id, 1])
                row[f"{name}_desired_vz"] = float(v_L_star[local_leader_id, 2])
                row[f"{name}_desired_ax"] = float(a_L_star[local_leader_id, 0])
                row[f"{name}_desired_ay"] = float(a_L_star[local_leader_id, 1])
                row[f"{name}_desired_az"] = float(a_L_star[local_leader_id, 2])
                row[f"{name}_leader_pos_error"] = float(leader_pos_err_each[local_leader_id])
                row[f"{name}_leader_vel_error"] = float(leader_vel_err_each[local_leader_id])
                row[f"{name}_ux"] = float(u_L[local_leader_id, 0])
                row[f"{name}_uy"] = float(u_L[local_leader_id, 1])
                row[f"{name}_uz"] = float(u_L[local_leader_id, 2])

            else:
                local_follower_id = int(np.where(FOLLOWER_AGENT_IDS == agent_id)[0][0])
                associated_leader_id = int(follower_leader[local_follower_id])
                associated_leader_agent_id = int(LEADER_AGENT_IDS[associated_leader_id])

                row[f"{name}_local_follower_id"] = local_follower_id
                row[f"{name}_associated_leader"] = CF_NAMES[associated_leader_agent_id]
                row[f"{name}_associated_leader_id"] = associated_leader_id
                row[f"{name}_ux"] = float(u_F_xy[local_follower_id, 0])
                row[f"{name}_uy"] = float(u_F_xy[local_follower_id, 1])
                row[f"{name}_distance_to_associated_leader_xy"] = float(
                    np.linalg.norm(p_F[local_follower_id, :2] - p_L[associated_leader_id, :2])
                )

        log_rows.append(row)

        # ----------------------------------------------------
        # Console printout
        # ----------------------------------------------------
        if loop_counter % 50 == 0:
            print(
                f"t = {t:5.2f} s | "
                f"L_pos_err = {leader_pos_err_mean:.3f} m | "
                f"L_vel_err = {leader_vel_err_mean:.3f} m/s | "
                f"L_form_err = {leader_pair_formation_error:.3f} m | "
                f"F_pos_dist = {follower_pos_err_mean:.3f} m | "
                f"F_vel_err = {follower_vel_err_mean:.3f} m/s | "
                f"min_FF = {min_ff:.3f} m | "
                f"min_FL = {min_fl:.3f} m | "
                f"cohesion0 = {cohesion_by_subgroup['cohesion_radius_xy_followers_of_leader0']:.3f} m | "
                f"cohesion1 = {cohesion_by_subgroup['cohesion_radius_xy_followers_of_leader1']:.3f} m"
            )

        loop_counter += 1

        elapsed = time.time() - now
        sleep_time = max(0.0, CONTROL_DT - elapsed)
        time.sleep(sleep_time)

    # Briefly hold current positions before the landing routine takes over.
    p_all, _, _, valid, _ = mocap.get()
    if np.all(valid):
        hold_cmd = p_all.copy()
        for i in range(N_TOTAL):
            hold_cmd[i, 2] = float(np.clip(hold_cmd[i, 2], workspace_min[2], workspace_max[2]))
        for _ in range(10):
            send_position_commands(swarm, hold_cmd)
            time.sleep(0.05)

    save_two_leader_log(log_rows)
    print("Two-leader associated-follower mission complete.")

# ============================================================
# Main
# ============================================================

def main():
    global extpose_running, fixed_xy

    natnet_client = start_natnet()
    airborne = False

    try:
        wait_for_all_poses()
        wait_for_stable_poses()
        check_initial_geometry()

        print("Initializing Crazyflie drivers...")
        cflib.crtp.init_drivers()

        factory = CachedCfFactory(rw_cache="./cache")

        print("Connecting to Crazyflie swarm...")
        print("URI_LIST:")
        for uri in URI_LIST:
            print("  ", uri)

        with Swarm(URI_LIST, factory=factory) as swarm:
            print("Connected.")

            try:
                print("Recovering any Crazyflies in crash mode...")
                swarm.parallel_safe(recover_cf)
                time.sleep(2.0)

                print("Starting external pose streams...")
                extpose_running = True
                swarm.parallel_safe(start_extpose_thread)
                time.sleep(2.0)

                print("Setting estimator=2 and resetting Kalman...")
                swarm.parallel_safe(setup_estimator_and_reset)
                print("Estimators configured and reset command sent.")

                time.sleep(2.0)

                print("Waiting for all estimators to converge...")
                swarm.parallel_safe(
                    wait_for_position_estimator,
                    args_dict={URIS[name]: [name] for name in CF_NAMES},
                )
                print("All estimators ready.")

                check_tracking(context="before arming")
                fixed_xy = capture_current_xy("takeoff")

                print("Arming all Crazyflies...")
                swarm.parallel_safe(arm_cf)
                time.sleep(2.0)

                check_tracking(context="before takeoff")
                takeoff_sequence(swarm)
                airborne = True

                check_tracking(context="before combined formation/flocking mission")
                run_two_leader_mission(swarm)

            except KeyboardInterrupt:
                print("Stopped by user. Landing...")

            except Exception as e:
                print("Emergency condition:")
                print(e)
                print("Landing...")

            finally:
                if airborne:
                    try:
                        landing_sequence(swarm)
                    except Exception as e:
                        print("Landing failed or interrupted:")
                        print(e)

                print("Final stop and disarm...")
                stop_all(swarm)
                airborne = False

    finally:
        stop_extpose_threads()

        try:
            natnet_client.shutdown()
        except Exception:
            pass

        print("Finished.")


if __name__ == "__main__":
    main()