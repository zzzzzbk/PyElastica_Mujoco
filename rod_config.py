"""Configuration model and XML (de)serialization for the Elastica GUI.

The setup is described as small dataclasses:

  * ``RodConfig``  - rod geometry / material.
  * ``EnvConfig``  - gravity + ground plane (visualization / collision).
  * ``Fixture``    - a boundary condition entry (fixes a node's DOFs).
  * ``Load``       - an external-force entry over a time window.
  * ``SimConfig``  - timing + lists of ``Fixture`` and ``Load`` (a small
                     "model tree", SolidWorks-style) + the ``EnvConfig``.

A compact ``<elastica_config>`` XML format lets a full setup be saved and
reloaded from the GUI.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple
import xml.etree.ElementTree as ET


# --- valid enumerations (kept here so the GUI and sim agree) -----------------

DIRECTIONS = ("up", "down", "horizontal")

# Where a fixture / point-load attaches along the rod.
LOCATIONS = ("start", "end", "node")

# What a fixture constrains.
FIXTURE_KINDS = ("fixed", "pinned")  # fixed = position+orientation, pinned = position only

# Load force distribution.
LOAD_KINDS = ("endpoint", "point", "uniform")


@dataclass
class RodConfig:
    """Geometric and material parameters of the rod."""

    length: float = 0.2
    radius_base: float = 0.012
    radius_tip: float = 0.0012
    n_elements: int = 100
    density: float = 1050.0
    youngs_modulus: float = 10_000.0
    poisson_ratio: float = 0.5
    direction: str = "up"  # one of DIRECTIONS

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def taper_ratio(self) -> float:
        """Fractional reduction from base to tip radius (0 = no taper)."""
        if self.radius_base <= 0:
            return 0.0
        return 1.0 - (self.radius_tip / self.radius_base)


@dataclass
class EnvConfig:
    """Environment: gravity and an optional ground plane."""

    gravity_on: bool = True
    gravity_g: float = -9.80665
    plane_on: bool = False           # show a ground plane
    plane_z: float = 0.0             # plane height (world z)
    plane_collision: bool = False    # enable rod<->plane contact in the sim
    collision_k: float = 50.0        # contact stiffness
    collision_nu: float = 10.0       # contact damping


@dataclass
class Fixture:
    """A boundary condition attached to a node.

    ``location``: "start" (node 0), "end" (last node), or "node" (``node_id``).
    ``kind``: "fixed" (position + orientation) or "pinned" (position only).
    """

    kind: str = "fixed"
    location: str = "start"
    node_id: int = 0

    def label(self) -> str:
        where = {"start": "start", "end": "tip"}.get(self.location, f"node {self.node_id}")
        return f"{self.kind.capitalize()} @ {where}"


@dataclass
class Load:
    """An external force applied over a time window.

    ``kind``:
        - "endpoint": force at the last node
        - "point":    force at ``node_id`` (or start/end per ``location``)
        - "uniform":  same force on every node
    """

    kind: str = "endpoint"
    force: Tuple[float, float, float] = (0.01, 0.0, 0.0)
    location: str = "end"
    node_id: int = 50
    start_time: float = 0.0
    end_time: float = 1.0

    def label(self) -> str:
        fx, fy, fz = self.force
        if self.kind == "uniform":
            where = "all nodes"
        elif self.kind == "endpoint":
            where = "tip"
        else:
            where = {"start": "start", "end": "tip"}.get(self.location, f"node {self.node_id}")
        return f"{self.kind.capitalize()} ({fx:g},{fy:g},{fz:g}) @ {where}"


@dataclass
class SimConfig:
    """Simulation-level parameters plus the fixture / load model tree."""

    damping_constant: float = 0.25
    final_time: float = 2.0
    time_step: float = 1.0e-5
    recording_fps: int = 30
    env: EnvConfig = field(default_factory=EnvConfig)
    fixtures: List[Fixture] = field(default_factory=lambda: [Fixture()])
    loads: List[Load] = field(default_factory=list)


# --- XML serialization -------------------------------------------------------

def _set(el: ET.Element, key: str, value) -> None:
    el.set(key, str(value))


def _b(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def to_xml(rod: RodConfig, sim: SimConfig, path: str | Path) -> None:
    """Write ``rod`` and ``sim`` to an ``<elastica_config>`` XML file."""
    root = ET.Element("elastica_config", version="2")

    rod_el = ET.SubElement(root, "rod")
    for k, v in asdict(rod).items():
        _set(rod_el, k, v)

    sim_el = ET.SubElement(root, "simulation")
    _set(sim_el, "damping_constant", sim.damping_constant)
    _set(sim_el, "final_time", sim.final_time)
    _set(sim_el, "time_step", sim.time_step)
    _set(sim_el, "recording_fps", sim.recording_fps)

    env = sim.env
    env_el = ET.SubElement(sim_el, "environment")
    for k, v in asdict(env).items():
        _set(env_el, k, v)

    fx_el = ET.SubElement(sim_el, "fixtures")
    for f in sim.fixtures:
        e = ET.SubElement(fx_el, "fixture")
        _set(e, "kind", f.kind)
        _set(e, "location", f.location)
        _set(e, "node_id", f.node_id)

    ld_el = ET.SubElement(sim_el, "loads")
    for ld in sim.loads:
        e = ET.SubElement(ld_el, "load")
        _set(e, "kind", ld.kind)
        fx, fy, fz = ld.force
        _set(e, "fx", fx); _set(e, "fy", fy); _set(e, "fz", fz)
        _set(e, "location", ld.location)
        _set(e, "node_id", ld.node_id)
        _set(e, "start_time", ld.start_time)
        _set(e, "end_time", ld.end_time)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(Path(path), encoding="utf-8", xml_declaration=True)


def from_xml(path: str | Path) -> tuple[RodConfig, SimConfig]:
    """Read an ``<elastica_config>`` XML file into (RodConfig, SimConfig)."""
    root = ET.parse(Path(path)).getroot()

    rod_el = root.find("rod")
    if rod_el is None:
        raise ValueError("Invalid config: missing <rod> section")
    g = rod_el.get
    rod = RodConfig(
        length=float(g("length", 0.2)),
        radius_base=float(g("radius_base", 0.012)),
        radius_tip=float(g("radius_tip", 0.0012)),
        n_elements=int(g("n_elements", 100)),
        density=float(g("density", 1050.0)),
        youngs_modulus=float(g("youngs_modulus", 10_000.0)),
        poisson_ratio=float(g("poisson_ratio", 0.5)),
        direction=g("direction", "up"),
    )

    sim_el = root.find("simulation")
    if sim_el is None:
        raise ValueError("Invalid config: missing <simulation> section")
    sg = sim_el.get

    env_el = sim_el.find("environment")
    if env_el is not None:
        eg = env_el.get
        env = EnvConfig(
            gravity_on=_b(eg("gravity_on", "True")),
            gravity_g=float(eg("gravity_g", -9.80665)),
            plane_on=_b(eg("plane_on", "False")),
            plane_z=float(eg("plane_z", 0.0)),
            plane_collision=_b(eg("plane_collision", "False")),
            collision_k=float(eg("collision_k", 50.0)),
            collision_nu=float(eg("collision_nu", 10.0)),
        )
    else:
        env = EnvConfig()

    fixtures: list[Fixture] = []
    fx_el = sim_el.find("fixtures")
    if fx_el is not None:
        for e in fx_el.findall("fixture"):
            fixtures.append(Fixture(
                kind=e.get("kind", "fixed"),
                location=e.get("location", "start"),
                node_id=int(e.get("node_id", 0)),
            ))

    loads: list[Load] = []
    ld_el = sim_el.find("loads")
    if ld_el is not None:
        for e in ld_el.findall("load"):
            loads.append(Load(
                kind=e.get("kind", "endpoint"),
                force=(float(e.get("fx", 0.0)), float(e.get("fy", 0.0)), float(e.get("fz", 0.0))),
                location=e.get("location", "end"),
                node_id=int(e.get("node_id", 50)),
                start_time=float(e.get("start_time", 0.0)),
                end_time=float(e.get("end_time", 1.0)),
            ))

    sim = SimConfig(
        damping_constant=float(sg("damping_constant", 0.25)),
        final_time=float(sg("final_time", 2.0)),
        time_step=float(sg("time_step", 1.0e-5)),
        recording_fps=int(sg("recording_fps", 30)),
        env=env,
        fixtures=fixtures,
        loads=loads,
    )
    return rod, sim
