from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import rod_trajectory as rt


def _history(name: str, offset: float = 0.0):
    time = np.asarray([0.0, 0.1, 0.2])
    position = np.zeros((3, 3, 5), dtype=np.float32)
    position[:, 0] = np.linspace(0.0, 0.3, 5) + offset
    radius = np.linspace(0.02, 0.01, 4, dtype=np.float32)
    directors = np.tile(np.eye(3)[None, :, :, None], (3, 1, 1, 4))
    return {
        "name": name,
        "time": time,
        "position": position,
        "radius": radius,
        "directors": directors,
    }


class RodTrajectoryTests(unittest.TestCase):
    def test_multi_rod_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "two_rods.dat"
            rt.save_dat([_history("first"), _history("second", 0.2)], 30, path)
            histories, fps = rt.load_dat(path)
            with path.open("rb") as handle:
                payload = pickle.load(handle)
        self.assertEqual(payload["format"], rt.DAT_FORMAT)
        self.assertEqual(len(histories), 2)
        self.assertEqual(fps, 30)
        self.assertEqual(histories[1]["position"].shape, (3, 3, 5))
        self.assertIsInstance(payload["systems"][0]["position"]["data"], bytes)

    def test_legacy_single_rod_callback_load(self):
        history = _history("legacy")
        legacy = {
            "recording_fps": 24,
            "systems": [
                {
                    "time": list(history["time"]),
                    "position": list(history["position"]),
                    "radius": [history["radius"]] * 3,
                    "directors": list(history["directors"]),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.dat"
            with path.open("wb") as handle:
                pickle.dump(legacy, handle)
            histories, fps = rt.load_dat(path)
        self.assertEqual(len(histories), 1)
        self.assertEqual(fps, 24)
        self.assertEqual(histories[0]["radius"].shape, (4,))


if __name__ == "__main__":
    unittest.main()
