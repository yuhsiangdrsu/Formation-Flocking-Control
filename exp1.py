"""
Crazyflie + OptiTrack adaptive-pinning formation mission.

This version keeps the safer workflow from the fixed-position takeoff / hover / land
script and inserts the formation flight mission between takeoff and landing.

Workflow:
1. Connect to OptiTrack / Motive through NatNet.
2. Wait for all rigid bodies and stable tracking.
3. Connect to the Crazyflie swarm.
4. Recover any Crazyflies that are in crash / tumble mode.
5. Start external position streaming to each Crazyflie.
6. Configure and reset Kalman estimators.
7. Wait until all estimators converge.
8. Capture current x/y positions for fixed-position takeoff.
9. Arm, take off, run formation mission, land, stop, and disarm.

Important:
- The OptiTrack rigid body IDs must match the Crazyflie names below.
- The coordinate frame is assumed to be: OptiTrack x/y/z -> Crazyflie world x/y/z.
- The formation mission uses position setpoints after takeoff.
"""

import sys
import time
import threading
import os
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import solve_continuous_are

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory, Swarm

# From NatNet SDK 4.4 PythonClient folder
from NatNetClient import NatNetClient


# ============================================================
# User configuration
# ============================================================

SERVER_IP = "192.168.3.3"   # Motive / OptiTrack PC
CLIENT_IP = "192.168.3.5"   # Python / Crazyflie PC
USE_MULTICAST = True

URIS: Dict[str, str] = {
    "cf0": "radio://0/80/2M/E7E7E7E711",
    "cf1": "radio://0/80/2M/E7E7E7E712",
    "cf2": "radio://1/100/2M/E7E7E7E713",
    "cf3": "radio://1/100/2M/E7E7E7E714",
    "cf4": "radio://2/120/2M/E7E7E7E715",
    "cf5": "radio://2/120/2M/E7E7E7E716",
}

# Motive rigid body ID -> Crazyflie name.
# Check this carefully if a drone receives commands but the wrong body moves.
RIGID_BODY_TO_NAME: Dict[int, str] = {
    1: "cf0",
    2: "cf1",
    3: "cf2",
    4: "cf3",
    5: "cf4",
    6: "cf5",
}

# Timing.
EXTPOSE_DT = 0.01       # 50 Hz external pose stream
SEND_DT = 0.1           # 10 Hz position-setpoint takeoff / landing, like the stable template
CONTROL_DT = 0.05        # 10 Hz formation controller, matching the adaptive-pinning simulation DT

TAKEOFF_TIME = 2.0
LANDING_TIME = 3.0
MISSION_T = 30.0
TAKEOFF_HOLD_TIME = 1.0

# Flight settings.
TAKEOFF_HEIGHT = 0.7
LANDING_HEIGHT = 0.05
LOST_TRACKING_TIMEOUT = 0.25

# Workspace safety bounds.
WORKSPACE_MIN = np.array([0, -2.5, 0.01])
WORKSPACE_MAX= np.array([ 3,  0.5, 2.0])

# Estimator settings.
ESTIMATOR_TIMEOUT = 10.0
ESTIMATOR_LOG_PERIOD_MS = 100
ESTIMATOR_SPREAD_LIMIT = 0.001

# Formation controller settings.
N_PIN = 4
PIN_UPDATE_EVERY = 10

# Data logging settings.
SAVE_FLIGHT_LOGS = True
LOG_DIR = "flight_logs"

K_PIN = 40
K_GRAPH = 1.0

# Conservative real-flight limits. Increase only after testing in a net/cage.
A_MAX = 100.0
V_MAX_XY = 100
V_MAX_Z = 0.25  # Used only if you switch back to velocity-world z control.
MAX_POSITION_STEP_XY = 0.1

# Position-setpoint conversion for the formation mission.
K_VEL_TO_POS = 1.0
POSITION_SETPOINT_YAW = 0.0


# ============================================================
# Derived configuration
# ============================================================

CF_NAMES: List[str] = list(URIS.keys())
URI_LIST: List[str] = [URIS[name] for name in CF_NAMES]
URI_TO_NAME: Dict[str, str] = {uri: name for name, uri in URIS.items()}
NAME_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CF_NAMES)}
RIGID_BODY_TO_AGENT: Dict[int, int] = {
    rigid_body_id: NAME_TO_INDEX[name]
    for rigid_body_id, name in RIGID_BODY_TO_NAME.items()
}
N_AGENTS = len(CF_NAMES)


# ============================================================
# Shared state
# ============================================================

extpose_running = False
extpose_threads: List[threading.Thread] = []
fixed_xy: np.ndarray | None = None
landing_xy: np.ndarray | None = None


@dataclass
class AgentPose:
    p: np.ndarray
    v: np.ndarray
    q: np.ndarray
    valid: bool
    last_update: float


class MocapState:
    def __init__(self, n_agents: int):
        self.lock = threading.Lock()
        self.p = np.zeros((n_agents, 3))
        self.v = np.zeros((n_agents, 3))
        self.q = np.zeros((n_agents, 4))
        self.q[:, 3] = 1.0
        self.prev_p = np.zeros((n_agents, 3))
        self.prev_time = np.zeros(n_agents)
        self.valid = np.zeros(n_agents, dtype=bool)
        self.last_update = np.zeros(n_agents)

    def update_agent(self, agent_id: int, position, rotation, timestamp: float) -> None:
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)

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

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            return (
                self.p.copy(),
                self.v.copy(),
                self.q.copy(),
                self.valid.copy(),
                self.last_update.copy(),
            )

    def get_agent(self, agent_id: int) -> AgentPose:
        with self.lock:
            return AgentPose(
                p=self.p[agent_id].copy(),
                v=self.v[agent_id].copy(),
                q=self.q[agent_id].copy(),
                valid=bool(self.valid[agent_id]),
                last_update=float(self.last_update[agent_id]),
            )


mocap = MocapState(N_AGENTS)


# ============================================================
# Target motion
# ============================================================

target0 = np.array([1.26, -1.13, 0.0])
v_target = np.array([0.0, 0.0, 0.0])
a_target = np.array([0.0, 0.0, 0.0])


def target_position(t: float) -> np.ndarray:
    return target0 + v_target * t


# ============================================================
# Desired 3D two-layer circular formation
# ============================================================

lower_radius = 0.7
upper_radius = 0.4

lower_z = 0.5
upper_z = 1

omega_lower = 0.15
omega_upper = -0.15

def formation_offsets(t: float) -> np.ndarray:
    r = np.zeros((N_AGENTS, 3))

    for i in range(3):
        theta = omega_lower * t + 2.0 * np.pi * i / 3.0
        r[i, 0] = lower_radius * np.cos(theta)
        r[i, 1] = lower_radius * np.sin(theta)
        r[i, 2] = lower_z

    for i in range(3):
        theta = omega_upper * t + 2.0 * np.pi * i / 3.0 + np.pi / 3.0
        idx = i + 3
        r[idx, 0] = upper_radius * np.cos(theta)
        r[idx, 1] = upper_radius * np.sin(theta)
        r[idx, 2] = upper_z

    return r


def formation_offset_velocities(t: float) -> np.ndarray:
    r_dot = np.zeros((N_AGENTS, 3))

    for i in range(3):
        theta = omega_lower * t + 2.0 * np.pi * i / 3.0
        r_dot[i, 0] = -lower_radius * omega_lower * np.sin(theta)
        r_dot[i, 1] =  lower_radius * omega_lower * np.cos(theta)

    for i in range(3):
        theta = omega_upper * t + 2.0 * np.pi * i / 3.0 + np.pi / 3.0
        idx = i + 3
        r_dot[idx, 0] = -upper_radius * omega_upper * np.sin(theta)
        r_dot[idx, 1] =  upper_radius * omega_upper * np.cos(theta)

    return r_dot


def formation_offset_accelerations(t: float) -> np.ndarray:
    r_ddot = np.zeros((N_AGENTS, 3))

    for i in range(3):
        theta = omega_lower * t + 2.0 * np.pi * i / 3.0
        r_ddot[i, 0] = -lower_radius * omega_lower**2 * np.cos(theta)
        r_ddot[i, 1] = -lower_radius * omega_lower**2 * np.sin(theta)

    for i in range(3):
        theta = omega_upper * t + 2.0 * np.pi * i / 3.0 + np.pi / 3.0
        idx = i + 3
        r_ddot[idx, 0] = -upper_radius * omega_upper**2 * np.cos(theta)
        r_ddot[idx, 1] = -upper_radius * omega_upper**2 * np.sin(theta)

    return r_ddot


# ============================================================
# LQR design
# ============================================================

A = np.block([
    [np.zeros((3, 3)), np.eye(3)],
    [np.zeros((3, 3)), np.zeros((3, 3))],
])

B = np.block([
    [np.zeros((3, 3))],
    [np.eye(3)],
])

# Same LQR weights as the adaptive-pinning simulation controller.
Q = np.diag([0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
R = np.diag([0.01, 0.01, 0.01])

P = solve_continuous_are(A, B, Q, R)
K = np.linalg.solve(R, B.T @ P)


# ============================================================
# Interaction graph
# ============================================================

def build_two_layer_graph() -> Dict[int, List[int]]:
    neighbors = {i: [] for i in range(N_AGENTS)}

    edges = [
        # lower triangle
        (0, 1), (1, 2), (2, 0),

        # upper triangle
        (3, 4), (4, 5), (5, 3),

        # vertical inter-layer links
        (0, 3), (1, 4), (2, 5),

        # extra cross-layer links
        (0, 4), (0, 5),
        (1, 3), (1, 5),
        (2, 3), (2, 4),
    ]

    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    return neighbors


neighbors = build_two_layer_graph()


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
# Mocap checks
# ============================================================

def wait_for_all_poses(timeout: float = 10.0) -> None:
    print("Waiting for all OptiTrack rigid bodies...")
    t0 = time.time()

    while time.time() - t0 < timeout:
        _, _, _, valid, _ = mocap.get()
        if np.all(valid):
            print("All OptiTrack poses received.")
            return

        print(f"Received {np.sum(valid)}/{N_AGENTS} rigid bodies...")
        time.sleep(0.5)

    raise RuntimeError("Not all OptiTrack rigid bodies were received.")


def wait_for_stable_poses(timeout: float = 10.0, stable_time: float = 1.0, max_age: float = 0.2) -> None:
    print("Waiting for stable OptiTrack poses...")
    t0 = time.time()
    stable_start = None

    while time.time() - t0 < timeout:
        now = time.time()
        p, _, _, valid, last_update = mocap.get()
        fresh = np.all(now - last_update < max_age)

        if np.all(valid) and fresh:
            stable_start = stable_start or now
            if now - stable_start >= stable_time:
                print("Stable OptiTrack poses confirmed.")
                for i, name in enumerate(CF_NAMES):
                    print(f"{name}: x={p[i, 0]:.3f}, y={p[i, 1]:.3f}, z={p[i, 2]:.3f}")
                return
        else:
            stable_start = None

        time.sleep(0.05)

    raise RuntimeError("Stable OptiTrack poses not confirmed.")


def check_tracking(context: str = "") -> None:
    now = time.time()
    _, _, _, valid, last_update = mocap.get()

    if not np.all(valid):
        raise RuntimeError(f"Not all agents have valid OptiTrack data {context}.")

    tracking_age = now - last_update
    if np.any(tracking_age > LOST_TRACKING_TIMEOUT):
        raise RuntimeError(f"OptiTrack tracking timeout {context}.")


def capture_current_xy(label: str) -> np.ndarray:
    p, _, _, valid, _ = mocap.get()

    if not np.all(valid):
        raise RuntimeError(f"Cannot capture {label} x/y because not all poses are valid.")

    xy = p[:, 0:2].copy()
    print(f"{label} x/y positions captured:")
    for i, name in enumerate(CF_NAMES):
        print(f"{name}: x_{label}={xy[i, 0]:.3f}, y_{label}={xy[i, 1]:.3f}")

    return xy


# ============================================================
# General helpers
# ============================================================

def sat_vec(vec: np.ndarray, vmax: float) -> np.ndarray:
    n = np.linalg.norm(vec)

    if n <= vmax or n < 1e-12:
        return vec

    return vmax * vec / n


def clip_velocity(v_cmd: np.ndarray) -> np.ndarray:
    v_cmd = v_cmd.copy()

    xy_norm = np.linalg.norm(v_cmd[:2])

    if xy_norm > V_MAX_XY:
        v_cmd[:2] = v_cmd[:2] * V_MAX_XY / xy_norm

    v_cmd[2] = np.clip(v_cmd[2], -V_MAX_Z, V_MAX_Z)

    return v_cmd


def inside_workspace(p: np.ndarray) -> bool:
    return bool(np.all(p >= WORKSPACE_MIN) and np.all(p <= WORKSPACE_MAX))


# ============================================================
# Crazyflie setup
# ============================================================

def setup_estimator_and_reset(scf) -> None:
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Setting estimator and resetting Kalman...")
    cf.param.set_value("stabilizer.estimator", "2")
    time.sleep(2)

    # Low-level commander because this script sends setpoints directly.
    cf.param.set_value("commander.enHighLevel", "0")
    time.sleep(2)

    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(2)
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
    """
    Send an arming/disarming request using the newer supervisor API when
    available, with a fallback for older cflib versions.
    """
    try:
        cf.supervisor.send_arming_request(arm)
    except AttributeError:
        cf.platform.send_arming_request(arm)


def recover_cf(scf) -> None:
    """
    Clear Crazyflie crash/tumble mode if it is active.

    This is equivalent to pressing Recover in the Crazyflie Client. It is safe
    to call at startup before arming; drones that are not in crash mode should
    simply continue normally.
    """
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


def arm_cf(scf) -> None:
    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]

    print(f"{name}: Arming...")
    send_arming_request_compatible(cf, True)
    time.sleep(2.0)
    print(f"{name}: Arming request sent.")


# ============================================================
# External pose streaming
# ============================================================

def extpose_loop(scf) -> None:
    global extpose_running

    cf = scf.cf
    name = URI_TO_NAME[cf.link_uri]
    agent_id = NAME_TO_INDEX[name]

    print(f"{name}: External pose thread started.")

    while extpose_running:
        pose = mocap.get_agent(agent_id)
        now = time.time()

        if pose.valid and now - pose.last_update < 0.2:
            # Use position-only external pose, matching the stable template.
            # If your firmware/estimator setup requires orientation, replace this
            # with cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw).
            cf.extpos.send_extpos(float(pose.p[0]), float(pose.p[1]), float(pose.p[2]))

        time.sleep(EXTPOSE_DT)

    print(f"{name}: External pose thread stopped.")


def start_extpose_thread(scf) -> None:
    thread = threading.Thread(target=extpose_loop, args=(scf,), daemon=True)
    thread.start()
    extpose_threads.append(thread)


def stop_extpose_threads() -> None:
    global extpose_running

    extpose_running = False
    for thread in extpose_threads:
        thread.join(timeout=1.0)


# ============================================================
# Position-setpoint takeoff and landing
# ============================================================

def send_position_for_duration(scf, x: float, y: float, z: float, duration: float, yaw: float = 0.0) -> None:
    cf = scf.cf
    steps = max(1, int(duration / SEND_DT))

    for _ in range(steps):
        cf.commander.send_position_setpoint(float(x), float(y), float(z), float(yaw))
        time.sleep(SEND_DT)


def takeoff_one_cf(scf, name: str) -> None:
    if fixed_xy is None:
        raise RuntimeError("fixed_xy has not been captured before takeoff.")

    cf = scf.cf
    agent_id = NAME_TO_INDEX[name]
    pose = mocap.get_agent(agent_id)

    if not pose.valid:
        raise RuntimeError(f"{name}: No valid OptiTrack pose before takeoff.")

    x = fixed_xy[agent_id, 0]
    y = fixed_xy[agent_id, 1]
    start_z = max(0.0, pose.p[2])
    steps = max(1, int(TAKEOFF_TIME / SEND_DT))

    print(f"{name}: Takeoff at fixed x/y x={x:.3f}, y={y:.3f}, z={start_z:.3f} -> {TAKEOFF_HEIGHT:.2f}")

    for k in range(steps):
        alpha = (k + 1) / steps
        z_cmd = start_z + alpha * (TAKEOFF_HEIGHT - start_z)
        cf.commander.send_position_setpoint(float(x), float(y), float(z_cmd), 0.0)
        time.sleep(SEND_DT)

    send_position_for_duration(scf, x, y, TAKEOFF_HEIGHT, duration=TAKEOFF_HOLD_TIME)
    print(f"{name}: Takeoff command sequence finished.")


def takeoff_sequence(swarm) -> None:
    print("Position-setpoint takeoff using fixed x/y spots...")
    swarm.parallel_safe(takeoff_one_cf, args_dict={URIS[name]: [name] for name in CF_NAMES})
    print("All Crazyflies finished takeoff command sequence.")


def land_one_cf(scf, name: str) -> None:
    if landing_xy is None:
        raise RuntimeError("landing_xy has not been captured before landing.")

    cf = scf.cf
    agent_id = NAME_TO_INDEX[name]
    pose = mocap.get_agent(agent_id)

    if not pose.valid:
        raise RuntimeError(f"{name}: No valid OptiTrack pose before landing.")

    x = landing_xy[agent_id, 0]
    y = landing_xy[agent_id, 1]
    start_z = max(pose.p[2], LANDING_HEIGHT)
    steps = max(1, int(LANDING_TIME / SEND_DT))

    print(f"{name}: Landing at fixed current x/y x={x:.3f}, y={y:.3f}, z={start_z:.3f} -> {LANDING_HEIGHT:.3f}")

    for k in range(steps):
        alpha = (k + 1) / steps
        z_cmd = start_z + alpha * (LANDING_HEIGHT - start_z)
        cf.commander.send_position_setpoint(float(x), float(y), float(max(z_cmd, LANDING_HEIGHT)), 0.0)
        time.sleep(SEND_DT)

    cf.commander.send_stop_setpoint()
    time.sleep(0.2)
    print(f"{name}: Landing command sequence finished.")


def landing_sequence(swarm) -> None:
    global landing_xy

    check_tracking(context="before landing")
    landing_xy = capture_current_xy("landing")

    print("Position-setpoint landing using fixed current x/y spots...")
    swarm.parallel_safe(land_one_cf, args_dict={URIS[name]: [name] for name in CF_NAMES})
    print("All Crazyflies finished landing command sequence.")


# ============================================================
# Stop and disarm
# ============================================================

def stop_and_disarm(scf) -> None:
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

    print(f"{name}: Disarm request sent.")


def stop_all(swarm) -> None:
    swarm.parallel_safe(stop_and_disarm)


# ============================================================
# Adaptive pinning controller
# ============================================================

def local_formation_mismatch(i: int, p: np.ndarray, r_now: np.ndarray) -> float:
    delta_i = 0.0

    for j in neighbors[i]:
        actual_rel = p[i] - p[j]
        desired_rel = r_now[i] - r_now[j]
        delta_i += np.linalg.norm(actual_rel - desired_rel)

    return float(delta_i)


def choose_adaptive_pins(p: np.ndarray, r_now: np.ndarray, n_pin: int) -> Tuple[np.ndarray, np.ndarray]:
    mismatches = np.array([
        local_formation_mismatch(i, p, r_now)
        for i in range(N_AGENTS)
    ])

    pinned = np.argsort(-mismatches)[:n_pin]

    return pinned, mismatches


def compute_control(p: np.ndarray, v: np.ndarray, t: float, pinned_agents: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_target = target_position(t)

    r_now = formation_offsets(t)
    r_dot_now = formation_offset_velocities(t)
    r_ddot_now = formation_offset_accelerations(t)

    desired_p = p_target + r_now
    desired_v = v_target + r_dot_now
    desired_a = a_target + r_ddot_now

    pinned_set = set(pinned_agents.tolist())

    u = np.zeros((N_AGENTS, 3))

    for i in range(N_AGENTS):
        x_i = np.hstack((p[i], v[i]))
        x_i_star = np.hstack((desired_p[i], desired_v[i]))

        b_i = 1.0 if i in pinned_set else 0.0
        e_pin = b_i * (x_i - x_i_star)

        e_graph = np.zeros(6)

        for j in neighbors[i]:
            x_j = np.hstack((p[j], v[j]))
            x_j_star = np.hstack((desired_p[j], desired_v[j]))
            e_graph += (x_i - x_j) - (x_i_star - x_j_star)

        e_total = K_PIN * e_pin + K_GRAPH * e_graph

        # Feedforward desired acceleration is applied to all agents.
        u[i] = (desired_a[i] - K @ e_total)
        u[i] = sat_vec(u[i], A_MAX)

    return u, desired_p, desired_v


def per_agent_formation_error(p: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute formation error for each agent.

    Returns:
        err_per_agent: shape (N_AGENTS,), Euclidean position error per agent.
        e_vec: shape (N_AGENTS, 3), vector error p - desired_p.
    """
    desired_p = target_position(t) + formation_offsets(t)
    e_vec = p - desired_p
    err_per_agent = np.linalg.norm(e_vec, axis=1)
    return err_per_agent, e_vec


def formation_error(p: np.ndarray, t: float) -> float:
    desired_p = target_position(t) + formation_offsets(t)
    e = p - desired_p

    return float(np.sqrt(np.mean(np.sum(e**2, axis=1))))


def save_formation_log(rows: List[dict]) -> None:
    """Save all recorded trajectory, velocity, command, and error data to CSV."""
    if not SAVE_FLIGHT_LOGS:
        return

    if len(rows) == 0:
        print("No formation log rows to save.")
        return

    os.makedirs(LOG_DIR, exist_ok=True)


    csv_path = os.path.join(LOG_DIR, f"formation_log_3.csv")

    fieldnames = list(rows[0].keys())

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved formation log to: {csv_path}")



# ============================================================
# Crazyflie command helpers
# ============================================================

def send_position_to_cf(scf, position_command: np.ndarray) -> None:
    """
    Send position setpoint to one Crazyflie.

    position_command = [x_world, y_world, z_world]
    """
    scf.cf.commander.send_position_setpoint(
        float(position_command[0]),
        float(position_command[1]),
        float(position_command[2]),
        float(POSITION_SETPOINT_YAW),
    )


def run_formation_command(scf, position_command: np.ndarray) -> None:
    """
    Per-Crazyflie command function used by swarm.parallel_safe().

    This preserves the same pattern as the original formation code:
    controller computes one command per URI, then Swarm dispatches those
    commands to the matching Crazyflie.
    """
    send_position_to_cf(scf, position_command)


def build_position_args(position_cmd: np.ndarray) -> Dict[str, List[np.ndarray]]:
    """
    Convert the N x 3 position command array into the URI-keyed args_dict
    required by swarm.parallel_safe().

    Each row is [x_world, y_world, z_world].
    """
    return {
        URIS[CF_NAMES[i]]: [position_cmd[i]]
        for i in range(N_AGENTS)
    }


def send_position_commands(swarm, position_cmd: np.ndarray) -> None:
    swarm.parallel_safe(
        run_formation_command,
        args_dict=build_position_args(position_cmd),
    )


# ============================================================
# Main formation mission
# ============================================================

def run_formation_mission(swarm) -> None:
    print("Starting adaptive-pinning formation mission...")

    mission_start = time.time()
    loop_counter = 0
    log_rows: List[dict] = []
    pinned_agents = np.array([0, 1, 2, 3, 4])
    mismatches = np.zeros(N_AGENTS)

    while True:
        now = time.time()
        t = now - mission_start

        if t >= MISSION_T:
            break

        p, v, _, valid, last_update = mocap.get()

        # Safety checks.
        if not np.all(valid):
            raise RuntimeError("Not all agents have valid OptiTrack data during formation mission.")

        tracking_age = now - last_update
        if np.any(tracking_age > LOST_TRACKING_TIMEOUT):
            raise RuntimeError("OptiTrack tracking timeout during formation mission.")

        for i in range(N_AGENTS):
            if not inside_workspace(p[i]):
                raise RuntimeError(f"{CF_NAMES[i]} left workspace: {p[i]}")

        # Adaptive pinning.
        r_now = formation_offsets(t)

        if loop_counter == 0 or loop_counter % PIN_UPDATE_EVERY == 0:
            pinned_agents, mismatches = choose_adaptive_pins(p, r_now, N_PIN)


        # Formation controller.
        u, desired_p, desired_v = compute_control(p, v, t, pinned_agents)

        # Convert acceleration command to position command.
        #
        # XY: acceleration command -> velocity command -> short-horizon position setpoint.
        # Z: send desired z directly as a position setpoint.
        position_cmd = np.zeros((N_AGENTS, 3))
        v_xy_cmd_all = np.zeros((N_AGENTS, 2))

        for i in range(N_AGENTS):
            v_xy_cmd = v[i, 0:2] + CONTROL_DT * u[i, 0:2]

            xy_norm = np.linalg.norm(v_xy_cmd)
            if xy_norm > V_MAX_XY:
                v_xy_cmd = v_xy_cmd * V_MAX_XY / xy_norm

            delta_xy = K_VEL_TO_POS * CONTROL_DT * v_xy_cmd

            # Extra guard: do not let the commanded position jump too far in
            # one control cycle, even if K_VEL_TO_POS is increased.
            delta_norm = np.linalg.norm(delta_xy)
            if delta_norm > MAX_POSITION_STEP_XY:
                delta_xy = delta_xy * MAX_POSITION_STEP_XY / delta_norm

            z_des = desired_p[i, 2]

            position_cmd[i, 0:2] = p[i, 0:2] + delta_xy
            position_cmd[i, 2] = z_des
            v_xy_cmd_all[i] = v_xy_cmd

            if not inside_workspace(position_cmd[i]):
                raise RuntimeError(
                    f"{CF_NAMES[i]} position setpoint outside workspace: "
                    f"current={p[i]}, position_cmd=[x={position_cmd[i,0]:.3f}, "
                    f"y={position_cmd[i,1]:.3f}, z={position_cmd[i,2]:.3f}], "
                    f"v_xy_cmd=[vx={v_xy_cmd[0]:.3f}, vy={v_xy_cmd[1]:.3f}]"
                )


        # Log trajectory, velocity, desired values, command, and formation error.
        err_each, err_vec = per_agent_formation_error(p, t)
        err_rms = formation_error(p, t)

        if SAVE_FLIGHT_LOGS:
            pinned_set_for_log = set(pinned_agents.tolist())

            for i, name in enumerate(CF_NAMES):
                log_rows.append({
                    "time_s": float(t),
                    "loop": int(loop_counter),
                    "agent_id": int(i),
                    "agent_name": name,

                    "x": float(p[i, 0]),
                    "y": float(p[i, 1]),
                    "z": float(p[i, 2]),

                    "vx": float(v[i, 0]),
                    "vy": float(v[i, 1]),
                    "vz": float(v[i, 2]),

                    "desired_x": float(desired_p[i, 0]),
                    "desired_y": float(desired_p[i, 1]),
                    "desired_z": float(desired_p[i, 2]),

                    "desired_vx": float(desired_v[i, 0]),
                    "desired_vy": float(desired_v[i, 1]),
                    "desired_vz": float(desired_v[i, 2]),

                    "error_x": float(err_vec[i, 0]),
                    "error_y": float(err_vec[i, 1]),
                    "error_z": float(err_vec[i, 2]),
                    "formation_error_agent": float(err_each[i]),
                    "formation_error_rms": float(err_rms),

                    "ux": float(u[i, 0]),
                    "uy": float(u[i, 1]),
                    "uz": float(u[i, 2]),

                    "cmd_vx": float(v_xy_cmd_all[i, 0]),
                    "cmd_vy": float(v_xy_cmd_all[i, 1]),
                    "cmd_x": float(position_cmd[i, 0]),
                    "cmd_y": float(position_cmd[i, 1]),
                    "cmd_z": float(position_cmd[i, 2]),

                    "is_pinned": int(i in pinned_set_for_log),
                    "pinned_agent_ids": ",".join(str(int(j)) for j in pinned_agents),
                    "pinned_agent_names": ",".join(CF_NAMES[int(j)] for j in pinned_agents),
                    "local_mismatch": float(mismatches[i]),
                })

        send_position_commands(swarm, position_cmd)

        if loop_counter % 50 == 0:
            print(
                f"t = {t:5.2f} s | "
                f"error = {err_rms:.3f} m | "
                f"pinned = {pinned_agents.tolist()} | "
                f"mismatch = {[round(float(x), 3) for x in mismatches]}"
            )

        loop_counter += 1

        elapsed = time.time() - now
        sleep_time = max(0.0, CONTROL_DT - elapsed)
        time.sleep(sleep_time)

    def _hold_before_landing(scf):
        name = URI_TO_NAME[scf.cf.link_uri]
        agent_id = NAME_TO_INDEX[name]
        pose = mocap.get_agent(agent_id)

        if pose.valid:
            x_hold = float(pose.p[0])
            y_hold = float(pose.p[1])
            z_hold = float(pose.p[2])
        else:
            x_hold = float(landing_xy[agent_id, 0]) if landing_xy is not None else 0.0
            y_hold = float(landing_xy[agent_id, 1]) if landing_xy is not None else 0.0
            z_hold = TAKEOFF_HEIGHT

        for _ in range(10):
            scf.cf.commander.send_position_setpoint(x_hold, y_hold, z_hold, 0.0)
            time.sleep(0.05)

    swarm.parallel_safe(_hold_before_landing)
    save_formation_log(log_rows)
    print("Formation mission complete.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    global extpose_running, fixed_xy

    natnet_client = start_natnet()
    airborne = False

    try:
        wait_for_all_poses()
        wait_for_stable_poses()

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

                check_tracking(context="before formation mission")
                run_formation_mission(swarm)

            except KeyboardInterrupt:
                print("Stopped by user. Landing...")

            except Exception as exc:
                print("Emergency condition:")
                print(exc)
                print("Landing...")

            finally:
                if airborne:
                    try:
                        landing_sequence(swarm)
                    except Exception as exc:
                        print("Landing failed or interrupted:")
                        print(exc)

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
