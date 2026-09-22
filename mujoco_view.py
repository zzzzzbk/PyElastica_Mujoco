"""Embedded MuJoCo 3D view for the Elastica GUI.

Renders a Cosserat rod as a chain of tapered capsule "connector" geoms injected
into a MuJoCo scene each frame (via ``mjv_connector``), then blits the offscreen
render into a Qt widget. A minimal static model (ground plane + light) provides
the world; the rod itself is decorative scene geometry, so arbitrary taper and
bending are handled without building a per-node body tree.

Supports a static preview and playback of one or multiple rod trajectories.
"""
from __future__ import annotations

import numpy as np
import mujoco

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QColor
from PySide6.QtWidgets import QWidget


_STATIC_XML = """
<mujoco model="elastica_scene">
  <visual>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="2048"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".18 .2 .24" rgb2=".22 .25 .3"
             width="512" height="512" mark="edge" markrgb=".3 .35 .4"/>
    <material name="gridmat" texture="grid" texrepeat="8 8" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.2" dir="-0.4 0.4 -1.2" directional="true"
           diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 0" group="2"
          material="gridmat" condim="1"/>
  </worldbody>
</mujoco>
"""

_FLOOR_GROUP = 2  # geom group used to show/hide the ground plane

ROD_COLORS = np.asarray(
    [
        [0.20, 0.55, 0.90, 1.0],
        [0.95, 0.45, 0.20, 1.0],
        [0.35, 0.75, 0.40, 1.0],
        [0.75, 0.40, 0.85, 1.0],
        [0.95, 0.75, 0.20, 1.0],
        [0.20, 0.75, 0.75, 1.0],
        [0.90, 0.35, 0.55, 1.0],
        [0.65, 0.65, 0.70, 1.0],
    ],
    dtype=np.float32,
)


class MujocoRodView(QWidget):
    """Widget that renders a rod (static or animated) with MuJoCo offscreen."""

    def __init__(self, parent=None, width=1280, height=960):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)

        self._render_w = width
        self._render_h = height

        self._model = mujoco.MjModel.from_xml_string(_STATIC_XML)
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        self._cam = mujoco.MjvCamera()
        self._opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self._cam)
        mujoco.mjv_defaultOption(self._opt)
        # Ground plane starts hidden; toggled via set_plane().
        self._floor_gid = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._opt.geomgroup[_FLOOR_GROUP] = 0
        self._cam.azimuth = 90.0
        self._cam.elevation = -15.0
        self._cam.distance = 0.6
        self._cam.lookat[:] = [0.0, 0.0, 0.1]

        # Trajectory state
        self._positions = None   # list[(T, 3, N+1)]
        self._radii = None       # list[(N,)]
        self._names = None
        self._times = None       # (T,)
        self._frame = 0

        self._qimage = None
        self._last_mouse = None

        # Placeholder message before any rod is loaded.
        self._message = "No rod loaded"

    # --- data loading --------------------------------------------------------

    def set_static(self, positions: np.ndarray, radii: np.ndarray, autoframe=True):
        """Show a single undeformed rod. positions: (3, N+1), radii: (N,)."""
        positions = np.asarray(positions)[None, ...]
        self.set_trajectories([positions], [radii], np.zeros(1), autoframe=autoframe)

    def set_trajectory(self, positions: np.ndarray, radii: np.ndarray,
                       times: np.ndarray = None, autoframe=True):
        """Load a full trajectory. positions: (T, 3, N+1), radii: (N,)."""
        positions = np.asarray(positions)
        if times is None:
            times = np.arange(len(positions), dtype=float)
        self.set_trajectories([positions], [radii], times, autoframe=autoframe)

    def set_trajectories(self, positions, radii, times=None, names=None,
                         autoframe=True):
        """Load one or more rods, all sampled at the same frame times."""
        if not positions or len(positions) != len(radii):
            raise ValueError("positions and radii require the same nonzero rod count")
        position_arrays = [np.asarray(value) for value in positions]
        radius_arrays = [np.asarray(value) for value in radii]
        frame_count = position_arrays[0].shape[0]
        if times is None:
            times = np.arange(frame_count, dtype=float)
        times = np.asarray(times)
        for index, (position, radius) in enumerate(zip(position_arrays, radius_arrays)):
            if position.ndim != 3 or position.shape[1] != 3:
                raise ValueError(f"Rod {index} positions must have shape [T, 3, N+1]")
            if position.shape[0] != frame_count:
                raise ValueError("All rods must have the same frame count")
            if radius.shape != (position.shape[2] - 1,):
                raise ValueError(f"Rod {index} radius count does not match its elements")
        if times.shape != (frame_count,):
            raise ValueError("times must have one value per frame")
        self._positions = position_arrays
        self._radii = radius_arrays
        self._times = times
        self._names = list(names) if names is not None else [
            f"rod_{index}" for index in range(len(position_arrays))
        ]
        self._frame = 0
        self._message = None
        if autoframe:
            self._autoframe()
        self._render_current()

    def set_plane(self, on: bool, z: float = 0.0):
        """Show/hide the ground plane and set its height."""
        self._opt.geomgroup[_FLOOR_GROUP] = 1 if on else 0
        if self._floor_gid >= 0:
            self._model.geom_pos[self._floor_gid, 2] = float(z)
            mujoco.mj_forward(self._model, self._data)
        self._render_current()

    @property
    def n_frames(self) -> int:
        return 0 if self._positions is None else len(self._times)

    def current_time(self) -> float:
        if self._times is None or self.n_frames == 0:
            return 0.0
        return float(self._times[min(self._frame, len(self._times) - 1)])

    # --- frame control -------------------------------------------------------

    def show_frame(self, index: int):
        if self._positions is None:
            return
        self._frame = int(np.clip(index, 0, self.n_frames - 1))
        self._render_current()

    # --- camera --------------------------------------------------------------

    def _autoframe(self):
        """Center and scale the camera to the full extent of the rod motion."""
        if self._positions is None:
            return
        allpts = np.concatenate(
            [value.transpose(0, 2, 1).reshape(-1, 3) for value in self._positions]
        )
        lo = allpts.min(axis=0)
        hi = allpts.max(axis=0)
        center = (lo + hi) / 2.0
        extent = float(np.linalg.norm(hi - lo))
        self._cam.lookat[:] = center
        self._cam.distance = max(extent * 1.6, 0.05)

    def reset_view(self):
        self._cam.azimuth = 90.0
        self._cam.elevation = -15.0
        self._autoframe()
        self._render_current()

    # --- rendering -----------------------------------------------------------

    def _render_current(self):
        """Render the current frame into a QImage and repaint."""
        if self._positions is None:
            self.update()
            return

        self._renderer.update_scene(self._data, self._cam, self._opt)
        scene = self._renderer.scene

        frame = min(self._frame, self.n_frames - 1)
        for rod_index, (positions, radii) in enumerate(
            zip(self._positions, self._radii)
        ):
            nodes = positions[frame].T
            color = ROD_COLORS[rod_index % len(ROD_COLORS)]
            for k in range(min(len(radii), nodes.shape[0] - 1)):
                if scene.ngeom >= scene.maxgeom:
                    break
                g = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                    np.zeros(3), np.zeros(3), np.zeros(9), color,
                )
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_CAPSULE, float(radii[k]),
                    nodes[k].astype(np.float64), nodes[k + 1].astype(np.float64),
                )
                scene.ngeom += 1

        rgb = self._renderer.render()  # (H, W, 3) uint8
        h, w, _ = rgb.shape
        # QImage needs a contiguous buffer that outlives the paint; copy it.
        self._qimage = QImage(
            np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format_RGB888
        ).copy()
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render so the aspect matches (image is scaled in paintEvent).
        if self._positions is not None:
            self._render_current()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 33, 38))
        if self._qimage is not None:
            scaled = self._qimage.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        if self._message:
            painter.setPen(QColor(180, 185, 195))
            painter.drawText(self.rect(), Qt.AlignCenter, self._message)
        painter.end()

    # --- mouse interaction ---------------------------------------------------

    def mousePressEvent(self, event):
        self._last_mouse = event.position()

    def mouseMoveEvent(self, event):
        if self._last_mouse is None:
            return
        p = event.position()
        dx = p.x() - self._last_mouse.x()
        dy = p.y() - self._last_mouse.y()
        self._last_mouse = p

        buttons = event.buttons()
        shift = event.modifiers() & Qt.ShiftModifier
        if buttons & Qt.LeftButton and not shift:
            # Orbit
            self._cam.azimuth = (self._cam.azimuth - dx * 0.4) % 360.0
            self._cam.elevation = float(
                np.clip(self._cam.elevation - dy * 0.4, -89.0, 89.0)
            )
            self._render_current()
        elif (buttons & Qt.RightButton) or (buttons & Qt.LeftButton and shift):
            # Pan in the camera plane
            scale = self._cam.distance * 0.001
            az = np.radians(self._cam.azimuth)
            right = np.array([np.sin(az), -np.cos(az), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            self._cam.lookat[:] = (
                np.asarray(self._cam.lookat) - right * dx * scale + up * dy * scale
            )
            self._render_current()

    def mouseReleaseEvent(self, event):
        self._last_mouse = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._cam.distance = float(np.clip(self._cam.distance * factor, 0.01, 100.0))
        self._render_current()
