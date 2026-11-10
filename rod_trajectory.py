"""Portable one-or-many rod DAT persistence, independent of PyElastica."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


DAT_FORMAT = "pyelastica-rod-trajectory"
DAT_FORMAT_VERSION = 1


def _pack_array(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(size) for size in array.shape),
        "data": array.tobytes(order="C"),
    }


def _unpack_array(value):
    if isinstance(value, dict) and all(
        key in value for key in ("dtype", "shape", "data")
    ):
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(
            tuple(value["shape"])
        ).copy()
    return np.asarray(value).copy()


def _normalise_history(system: dict, index: int) -> dict:
    system_type = system.get("type", "cosserat_rod")
    time = _unpack_array(system["time"]).astype(np.float64, copy=False)
    position = _unpack_array(system["position"]).astype(np.float32, copy=False)
    radius = _unpack_array(system["radius"]).astype(np.float32, copy=False)
    if radius.ndim == 2:
        radius = radius[0]
    if time.ndim != 1 or time.size < 1 or position.ndim != 3 or position.shape[1] != 3:
        raise ValueError(f"Rod {index} has invalid time or position dimensions")
    if system_type == "rigid_cylinder":
        if position.shape[2] < 2 or radius.shape != (1,):
            raise ValueError(f"Rigid cylinder {index} requires endpoints and one radius")
        position = position[:, :, [0, -1]]
    if position.shape[0] != time.size or radius.shape != (position.shape[2] - 1,):
        raise ValueError(f"Rod {index} frame/node/radius counts do not match")
    if time.size > 1 and np.any(np.diff(time) <= 0.0):
        raise ValueError(f"Rod {index} time samples must be strictly increasing")
    if (
        not np.all(np.isfinite(time))
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(radius))
        or np.any(radius <= 0.0)
    ):
        raise ValueError(f"Rod {index} contains non-finite time or position data")
    output = {
        "name": str(system.get("name", f"rod_{index}")),
        "type": system_type,
        "time": time,
        "position": position,
        "radius": radius,
    }
    directors = system.get("directors", system.get("director"))
    if directors is not None:
        directors = _unpack_array(directors).astype(
            np.float32, copy=False
        )
        expected = (time.size, 3, 3, position.shape[2] - 1)
        if directors.shape != expected or not np.all(np.isfinite(directors)):
            raise ValueError(f"Rod {index} directors must have shape {expected}")
        output["directors"] = directors
    return output


def save_dat(history: dict | list[dict], recording_fps: int, path: str | Path,
             **extra) -> None:
    """Save one history or a list of histories in the shared DAT format."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(".dat")
    histories = [history] if isinstance(history, dict) else list(history)
    if not histories:
        raise ValueError("At least one rod history is required")
    systems = []
    reference_time = None
    for index, item in enumerate(histories):
        normal = _normalise_history(item, index)
        if reference_time is None:
            reference_time = normal["time"]
        elif not np.array_equal(reference_time, normal["time"]):
            raise ValueError("All rods must use identical time samples")
        system = {
            "name": normal["name"],
            "type": normal["type"],
            "time": _pack_array(normal["time"], np.float64),
            "position": _pack_array(normal["position"], np.float32),
            "radius": _pack_array(normal["radius"], np.float32),
        }
        if "directors" in normal:
            system["directors"] = _pack_array(normal["directors"], np.float32)
        systems.append(system)
    if len({system["name"] for system in systems}) != len(systems):
        raise ValueError("Rod names must be unique")
    data = dict(
        format=DAT_FORMAT,
        format_version=DAT_FORMAT_VERSION,
        recording_fps=float(recording_fps),
        systems=systems,
        metadata=extra.pop("metadata", {}),
        **extra,
    )
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dat(path: str | Path) -> tuple[list[dict], int]:
    """Load unified or legacy DAT; return (rod histories, recording_fps)."""
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("DAT file must contain a dictionary")
    format_name = data.get("format")
    if format_name not in (
        None,
        DAT_FORMAT,
        "simulated-octopus-trajectory",
    ):
        raise ValueError(f"Unsupported DAT format: {format_name!r}")
    if format_name == DAT_FORMAT and int(data.get("format_version", -1)) != 1:
        raise ValueError("Unsupported pyelastica-rod-trajectory version")
    systems = data.get("systems", [])
    if not systems:
        raise ValueError("DAT file contains no rod systems")
    named_legacy_systems = isinstance(systems, dict)
    if named_legacy_systems:
        if format_name is not None:
            raise ValueError("Versioned DAT systems must be a list")
        # Older multi-rod callbacks used names as dictionary keys. The rigid
        # cylinder's first and last position nodes are its two endpoints.
        rods = []
        for name, system in systems.items():
            if name == "rigid":
                if not isinstance(system, dict):
                    raise ValueError("Legacy rigid system must be a dictionary")
                rods.append(dict(system, name=name, type="rigid_cylinder"))
                continue
            if not isinstance(name, str) or not name.startswith("rod") or not name[3:].isdigit():
                continue
            if not isinstance(system, dict):
                raise ValueError(f"Legacy system {name!r} must be a dictionary")
            rods.append(dict(system, name=name))
        if not rods:
            raise ValueError("Legacy DAT contains no supported systems")
        systems = rods
    elif not isinstance(systems, list):
        raise ValueError("DAT systems must be a list or legacy named dictionary")
    histories = [
        _normalise_history(system, index) for index, system in enumerate(systems)
    ]
    if len({history["name"] for history in histories}) != len(histories):
        raise ValueError("Rod names must be unique")
    reference_time = histories[0]["time"]
    if any(
        not np.array_equal(reference_time, item["time"])
        for item in histories[1:]
    ):
        raise ValueError("All rods must use identical time samples")
    if named_legacy_systems and "recording_fps" not in data and reference_time.size > 1:
        fps = int(round(1.0 / float(np.median(np.diff(reference_time)))))
    else:
        fps = int(round(float(data.get("recording_fps", 30))))
    return histories, fps


def history_to_arrays(history: dict):
    """Return one rod's positions, static radii, and frame times."""
    normal = _normalise_history(history, 0)
    return normal["position"], normal["radius"], normal["time"]


def histories_to_arrays(histories: list[dict]):
    """Return position/radius lists plus their common time vector."""
    if not histories:
        raise ValueError("At least one rod history is required")
    converted = [history_to_arrays(history) for history in histories]
    times = converted[0][2]
    if any(not np.array_equal(times, item[2]) for item in converted[1:]):
        raise ValueError("All rods must use identical time samples")
    return [item[0] for item in converted], [item[1] for item in converted], times
