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

    def test_legacy_named_multi_rod_callback_load_and_resave(self):
        first = _history("unused")
        second = _history("unused", 0.2)
        legacy = {
            "systems": {
                "rod0": {
                    "time": list(first["time"]),
                    "position": list(first["position"]),
                    "radius": [first["radius"]] * 3,
                    "director": list(first["directors"]),
                },
                "rod1": {
                    "time": list(second["time"]),
                    "position": list(second["position"]),
                    "radius": [second["radius"] * (1 + 0.1 * t) for t in range(3)],
                },
                "rigid": {
                    "time": list(first["time"]),
                    "position": [np.tile(np.linspace(0, 0.7, 10), (3, 1))] * 3,
                    "radius": [np.array([0.03])] * 3,
                },
            },
            "LK": {"rod0": [0.0] * 3},
        }
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old.dat"
            new_path = Path(directory) / "new.dat"
            with old_path.open("wb") as handle:
                pickle.dump(legacy, handle)
            histories, fps = rt.load_dat(old_path)
            rt.save_dat(histories, fps, new_path)
            converted, converted_fps = rt.load_dat(new_path)
        self.assertEqual([item["name"] for item in histories], ["rod0", "rod1", "rigid"])
        self.assertEqual(histories[2]["type"], "rigid_cylinder")
        self.assertEqual(histories[2]["position"].shape, (3, 3, 2))
        self.assertAlmostEqual(float(histories[2]["radius"][0]), 0.03)
        self.assertEqual(fps, 10)
        self.assertEqual(converted_fps, fps)
        np.testing.assert_array_equal(histories[1]["radius"], second["radius"])
        np.testing.assert_array_equal(converted[1]["position"], histories[1]["position"])
        np.testing.assert_array_equal(converted[2]["position"], histories[2]["position"])
        self.assertEqual(converted[2]["type"], "rigid_cylinder")
        self.assertEqual(histories[0]["directors"].shape, (3, 3, 3, 4))


if __name__ == "__main__":
    unittest.main()
