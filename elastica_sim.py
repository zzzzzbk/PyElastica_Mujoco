"""Headless PyElastica simulation for the GUI.

Trimmed adaptation of ``run_buckling_test.ArmEnvironment`` (no COOM muscles, no
drag), updated for the installed pyelastica 1.0.0 API:

  * boundary condition class is ``OneEndFixedBC`` (not ``OneEndFixedRod``);
  * time stepping uses ``PositionVerlet().step(collection, t, dt)`` directly
    (the legacy ``extend_stepper_interface`` / ``do_step`` path is broken in
    this release).

The simulation records ``time`` / ``position`` / ``radius`` / ``directors``
into a ``defaultdict(list)`` with the same shape produced by
``utils.VisualizerDictCallBack``, and ``save_dat`` writes the same pickle
layout as ``run_buckling_test.save_data`` so results remain loadable through
``utils.load_data``.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from elastica import (
    BaseSystemCollection,
    Constraints,
    Forcing,
    CallBacks,
    Damping,
    Contact,
    CosseratRod,
    OneEndFixedBC,
    FixedConstraint,
    GeneralConstraint,
    GravityForces,
    NoForces,
    Plane,
    RodPlaneContact,
    CallBackBaseClass,
    AnalyticalLinearDamper,
)
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica._calculus import _isnan_check

from rod_config import RodConfig, SimConfig


class _Simulator(
    BaseSystemCollection, Constraints, Forcing, CallBacks, Damping, Contact
):
    pass


# --- custom external forces --------------------------------------------------

class TimedPointForce(NoForces):
    """Apply a constant force to a single node during a time window."""

    def __init__(self, node_id, force, start_time=0.0, end_time=np.inf):
        super().__init__()
        self.node_id = int(node_id)
        self.force = np.asarray(force, dtype=np.float64)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)):
        if self.start_time <= time <= self.end_time:
            n = system.external_forces.shape[1]
            idx = max(0, min(self.node_id, n - 1))
            system.external_forces[:, idx] += self.force


class TimedEndpointForce(NoForces):
    """Apply a constant force to the last node during a time window."""

    def __init__(self, force, start_time=0.0, end_time=np.inf):
        super().__init__()
        self.force = np.asarray(force, dtype=np.float64)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)):
        if self.start_time <= time <= self.end_time:
            system.external_forces[:, -1] += self.force


class TimedUniformForce(NoForces):
    """Apply a per-node force to every node during a time window."""

    def __init__(self, force, start_time=0.0, end_time=np.inf):
        super().__init__()
        self.force = np.asarray(force, dtype=np.float64).reshape(3, 1)
        self.start_time = float(start_time)
        self.end_time = float(end_time)

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)):
        if self.start_time <= time <= self.end_time:
            system.external_forces[:] += self.force


class _RecordCallBack(CallBackBaseClass):
    """Record rod history into a defaultdict(list) at a fixed stride."""

    def __init__(self, step_skip: int, params: dict):
        super().__init__()
        self.every = max(1, int(step_skip))
        self.params = params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.params["time"].append(time)
            self.params["position"].append(system.position_collection.copy())
            self.params["radius"].append(system.radius.copy())
            self.params["directors"].append(system.director_collection.copy())


# --- rod / simulator construction --------------------------------------------

def _direction_vectors(direction: str):
    if direction == "up":
        return np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    if direction == "down":
        return np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0])
    if direction == "horizontal":
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    raise ValueError(f"Unknown direction: {direction}")


def rod_geometry(cfg: RodConfig):
    """Return (positions Nx3+? , radii) of the *undeformed* rod for preview.

    positions: (3, n_elements + 1) node coordinates
    radii:     (n_elements,) element radii
    """
    n = cfg.n_elements
    radius = np.linspace(cfg.radius_base, cfg.radius_tip, n + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2.0
    direction, _ = _direction_vectors(cfg.direction)
    s = np.linspace(0.0, cfg.length, n + 1)
    positions = np.outer(direction, s)  # (3, n+1)
    return positions, radius_mean


def build_rod(cfg: RodConfig) -> CosseratRod:
    n = cfg.n_elements
    radius = np.linspace(cfg.radius_base, cfg.radius_tip, n + 1)
    radius_mean = (radius[:-1] + radius[1:]) / 2.0
    direction, normal = _direction_vectors(cfg.direction)
    rod = CosseratRod.straight_rod(
        n_elements=n,
        start=np.zeros(3),
        direction=direction,
        normal=normal,
        base_length=cfg.length,
        base_radius=radius_mean.copy(),
        density=cfg.density,
        youngs_modulus=cfg.youngs_modulus,
        shear_modulus=cfg.shear_modulus,
    )
    return rod


def _resolve_node(location: str, node_id: int, n_nodes: int) -> int:
    """Map a (location, node_id) pair to a concrete node index."""
    if location == "start":
        return 0
    if location == "end":
        return n_nodes - 1
    return int(max(0, min(node_id, n_nodes - 1)))


def _add_load(sim: _Simulator, rod: CosseratRod, load, n_nodes: int):
    force = np.asarray(load.force, dtype=np.float64)
    if load.kind == "endpoint":
        sim.add_forcing_to(rod).using(
            TimedEndpointForce, force=force,
            start_time=load.start_time, end_time=load.end_time,
        )
    elif load.kind == "point":
        node = _resolve_node(load.location, load.node_id, n_nodes)
        sim.add_forcing_to(rod).using(
            TimedPointForce, node_id=node, force=force,
            start_time=load.start_time, end_time=load.end_time,
        )
    elif load.kind == "uniform":
        sim.add_forcing_to(rod).using(
            TimedUniformForce, force=force,
            start_time=load.start_time, end_time=load.end_time,
        )
    else:
        raise ValueError(f"Unknown load kind: {load.kind}")


def _add_fixture(sim: _Simulator, rod: CosseratRod, fx, n_nodes: int):
    node = _resolve_node(fx.location, fx.node_id, n_nodes)
    # Directors are indexed by element; clamp the last node to the last element.
    dir_idx = min(node, n_nodes - 2)
    if fx.kind == "pinned":
        # Position only: constrain the node's translation, leave orientation free.
        sim.constrain(rod).using(
            GeneralConstraint,
            constrained_position_idx=(node,),
            constrained_director_idx=(),
            translational_constraint_selector=np.array([True, True, True]),
        )
    else:  # "fixed": position + orientation
        sim.constrain(rod).using(
            FixedConstraint,
            constrained_position_idx=(node,),
            constrained_director_idx=(dir_idx,),
        )


def build_simulator(rod_cfg: RodConfig, sim_cfg: SimConfig):
    """Assemble a finalized simulator and return (sim, rod, params, step_skip)."""
    sim = _Simulator()
    rod = build_rod(rod_cfg)
    sim.append(rod)
    n_nodes = rod_cfg.n_elements + 1
    env = sim_cfg.env

    # Damping
    sim.dampen(rod).using(
        AnalyticalLinearDamper,
        damping_constant=sim_cfg.damping_constant,
        time_step=sim_cfg.time_step,
    )

    # Gravity
    if env.gravity_on:
        sim.add_forcing_to(rod).using(
            GravityForces,
            acc_gravity=np.array([0.0, 0.0, env.gravity_g]),
        )

    # Loads (external forces)
    for load in sim_cfg.loads:
        _add_load(sim, rod, load, n_nodes)

    # Fixtures (boundary conditions)
    for fx in sim_cfg.fixtures:
        _add_fixture(sim, rod, fx, n_nodes)

    # Ground plane + optional collision
    if env.plane_on and env.plane_collision:
        plane = Plane(
            plane_origin=np.array([0.0, 0.0, env.plane_z]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )
        sim.append(plane)
        sim.detect_contact_between(rod, plane).using(
            RodPlaneContact, k=env.collision_k, nu=env.collision_nu
        )

    # Recording callback
    step_skip = max(1, int(1.0 / (sim_cfg.recording_fps * sim_cfg.time_step)))
    params: dict = defaultdict(list)
    sim.collect_diagnostics(rod).using(
        _RecordCallBack, step_skip=step_skip, params=params
    )

    sim.finalize()
    return sim, rod, params, step_skip


def run(
    rod_cfg: RodConfig,
    sim_cfg: SimConfig,
    progress_cb: Optional[Callable[[float], bool]] = None,
) -> dict:
    """Run the simulation and return the recorded history dict.

    ``progress_cb(fraction)`` is called periodically; if it returns True the run
    stops early (used for cancellation from the GUI).
    """
    sim, rod, params, step_skip = build_simulator(rod_cfg, sim_cfg)
    stepper = PositionVerlet()

    total_steps = int(sim_cfg.final_time / sim_cfg.time_step)
    dt = np.float64(sim_cfg.time_step)
    time = np.float64(0.0)

    report_every = max(1, total_steps // 200)
    for k in range(total_steps):
        time = stepper.step(sim, time, dt)

        if _isnan_check(rod.position_collection):
            print("NaN detected in the simulation — stopping early.")
            break

        if progress_cb is not None and (k % report_every == 0):
            if progress_cb(k / total_steps):
                break

    if progress_cb is not None:
        progress_cb(1.0)

    # Convert to plain dict of arrays for downstream use / pickling.
    history = {key: list(val) for key, val in params.items()}
    return history


# --- persistence (compatible with run_buckling_test / utils.load_data) -------

def save_dat(history: dict, recording_fps: int, path: str | Path, **extra) -> None:
    """Save history as pickle in the run_buckling_test.save_data layout."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".dat")
    data = dict(recording_fps=recording_fps, systems=[history], **extra)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_dat(path: str | Path) -> tuple[dict, int]:
    """Load a .dat file; return (history_dict, recording_fps)."""
    with open(Path(path), "rb") as f:
        data = pickle.load(f)
    history = data["systems"][0]
    fps = int(data.get("recording_fps", 30))
    return history, fps


def history_to_arrays(history: dict):
    """Return (positions T x3x(N+1), radii (N,), times (T,)) from a history dict."""
    positions = np.asarray(history["position"])  # (T, 3, N+1)
    radii = np.asarray(history["radius"][0])      # (N,)
    times = np.asarray(history["time"])           # (T,)
    return positions, radii, times
