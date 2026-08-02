"""PyElastica GUI with an embedded MuJoCo 3D view.

A three-panel PySide6 application:

  * Left   - rod geometry / material parameters (live undeformed preview) and an
             environment panel (gravity + ground plane / collision).
  * Center - the embedded MuJoCo view (geometry check + animation) and playback.
  * Right  - a SolidWorks-style model tree: lists of fixtures (boundary
             conditions) and loads (external forces) at the top, an editor to
             define and add new ones below, then timing, Run, and save/load.

Run with the python311 conda environment:
    python elastica_gui.py
"""
from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QSize
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QSlider,
    QDoubleSpinBox, QSpinBox, QComboBox, QCheckBox, QGridLayout, QVBoxLayout,
    QHBoxLayout, QSplitter, QGroupBox, QProgressBar, QScrollArea, QFileDialog,
    QMessageBox, QStyle, QFrame, QListWidget, QListWidgetItem, QSizePolicy,
    QAbstractSpinBox,
)

import elastica_sim as es
from rod_config import (
    RodConfig, SimConfig, EnvConfig, Fixture, Load,
    DIRECTIONS, LOCATIONS, FIXTURE_KINDS, LOAD_KINDS, to_xml, from_xml,
)
from mujoco_view import MujocoRodView


LOCATION_LABELS = {"start": "Start (node 0)", "end": "End (tip)", "node": "Node id"}
FIXTURE_KIND_LABELS = {"fixed": "Fixed (position + orientation)", "pinned": "Pinned (position only)"}
LOAD_KIND_LABELS = {"endpoint": "Endpoint (tip)", "point": "Point (node)", "uniform": "Uniform (all nodes)"}


# --- background simulation worker -------------------------------------------

class SimWorker(QObject):
    """Runs a simulation off the GUI thread and reports progress."""

    progress = Signal(float)
    finished = Signal(dict, int)   # history, recording_fps
    failed = Signal(str)

    def __init__(self, rod_cfg: RodConfig, sim_cfg: SimConfig):
        super().__init__()
        self._rod_cfg = rod_cfg
        self._sim_cfg = sim_cfg
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            def cb(frac):
                self.progress.emit(frac)
                return self._cancel
            history = es.run(self._rod_cfg, self._sim_cfg, progress_cb=cb)
            self.finished.emit(history, self._sim_cfg.recording_fps)
        except Exception as exc:  # surface any error to the GUI
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc))


# --- control helpers (label + slider + spinbox, kept in sync) ---------------

def _add_double_control(grid, row, label, lo, hi, value, decimals=4,
                        step=None, scale=10000):
    """Add a labelled double control. Returns (spinbox, slider)."""
    grid.addWidget(QLabel(label), row, 0)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(int(lo * scale), int(hi * scale))
    slider.setValue(int(value * scale))
    slider.setMinimumWidth(40)
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setMaximumWidth(90)
    spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    if step is not None:
        spin.setSingleStep(step)
    grid.addWidget(slider, row, 1)
    grid.addWidget(spin, row, 2)

    _guard = {"on": False}

    def on_slider(v):
        if _guard["on"]:
            return
        _guard["on"] = True
        spin.setValue(v / scale)
        _guard["on"] = False

    def on_spin(v):
        if _guard["on"]:
            return
        _guard["on"] = True
        slider.setValue(int(v * scale))
        _guard["on"] = False

    slider.valueChanged.connect(on_slider)
    spin.valueChanged.connect(on_spin)
    return spin, slider


def _add_int_control(grid, row, label, lo, hi, value):
    grid.addWidget(QLabel(label), row, 0)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(lo, hi)
    slider.setValue(value)
    slider.setMinimumWidth(40)
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    spin.setMaximumWidth(80)
    grid.addWidget(slider, row, 1)
    grid.addWidget(spin, row, 2)
    slider.valueChanged.connect(spin.setValue)
    spin.valueChanged.connect(slider.setValue)
    return spin, slider


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyElastica + MuJoCo Design Tool")
        self.resize(1450, 820)

        self._thread = None
        self._worker = None
        self._playing = False
        self._history = None

        # Model-tree data (SolidWorks-style lists).
        self._fixtures: list[Fixture] = [Fixture(kind="fixed", location="start")]
        self._loads: list[Load] = []

        self.view = MujocoRodView()

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left_panel())
        split.addWidget(self._build_center_panel())
        split.addWidget(self._build_right_panel())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 6)
        split.setStretchFactor(2, 3)
        split.setSizes([330, 720, 400])
        self.setCentralWidget(split)

        # Make every spin box a plain text field (no up/down step buttons).
        for sb in self.findChildren(QAbstractSpinBox):
            sb.setButtonSymbols(QAbstractSpinBox.NoButtons)

        # Debounced live preview timer.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(120)
        self._preview_timer.timeout.connect(self.refresh_preview)

        # Playback timer.
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

        self._refresh_lists()
        self.refresh_preview()

    # --- left panel: rod parameters + environment ---------------------------

    def _build_left_panel(self) -> QWidget:
        container = QWidget()
        col = QVBoxLayout(container)

        # Rod parameters
        rod_box = QGroupBox("Rod parameters")
        grid = QGridLayout()
        grid.setColumnStretch(0, 0)   # label: natural width
        grid.setColumnStretch(1, 1)   # slider: absorbs extra width
        grid.setColumnStretch(2, 0)   # spinbox: compact
        r = 0
        self.sp_length, _ = _add_double_control(grid, r, "Length (m)", 0.01, 1.0, 0.2); r += 1
        self.sp_rbase, _ = _add_double_control(grid, r, "Base radius (m)", 0.0005, 0.05, 0.012, decimals=5); r += 1
        self.sp_rtip, _ = _add_double_control(grid, r, "Tip radius (m)", 0.0001, 0.05, 0.003, decimals=5); r += 1
        self.sp_nelem, _ = _add_int_control(grid, r, "Elements", 5, 300, 50); r += 1
        self.sp_density, _ = _add_double_control(grid, r, "Density (kg/m^3)", 10.0, 5000.0, 1050.0, decimals=1, scale=10); r += 1
        self.sp_E, _ = _add_double_control(grid, r, "Young's E (Pa)", 100.0, 1_000_000.0, 10000.0, decimals=1, scale=1); r += 1
        self.sp_nu, _ = _add_double_control(grid, r, "Poisson ratio", 0.0, 0.5, 0.5, decimals=3); r += 1
        grid.addWidget(QLabel("Direction"), r, 0)
        self.cb_direction = QComboBox()
        self.cb_direction.addItems(list(DIRECTIONS))
        grid.addWidget(self.cb_direction, r, 1, 1, 2); r += 1
        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet("color:#777; font-size:11px;")
        self.lbl_info.setWordWrap(True)
        grid.addWidget(self.lbl_info, r, 0, 1, 3); r += 1
        rod_box.setLayout(grid)
        col.addWidget(rod_box)

        # Environment
        env_box = QGroupBox("Environment")
        eg = QGridLayout()
        eg.setColumnStretch(0, 1)
        r = 0
        self.chk_gravity = QCheckBox("Gravity")
        self.chk_gravity.setChecked(True)
        eg.addWidget(self.chk_gravity, r, 0)
        self.sp_g = QDoubleSpinBox()
        self.sp_g.setDecimals(4); self.sp_g.setRange(-100.0, 100.0)
        self.sp_g.setValue(-9.80665)
        self.sp_g.setMaximumWidth(80)
        eg.addWidget(QLabel("g (m/s^2)"), r, 1)
        eg.addWidget(self.sp_g, r, 2); r += 1

        self.chk_plane = QCheckBox("Ground plane")
        eg.addWidget(self.chk_plane, r, 0)
        self.sp_plane_z = QDoubleSpinBox()
        self.sp_plane_z.setDecimals(4); self.sp_plane_z.setRange(-10.0, 10.0)
        self.sp_plane_z.setValue(0.0); self.sp_plane_z.setSingleStep(0.005)
        self.sp_plane_z.setMaximumWidth(80)
        eg.addWidget(QLabel("plane z (m)"), r, 1)
        eg.addWidget(self.sp_plane_z, r, 2); r += 1

        self.chk_collision = QCheckBox("Plane collision")
        eg.addWidget(self.chk_collision, r, 0, 1, 3); r += 1

        eg.addWidget(QLabel("Contact k / nu"), r, 0)
        self.sp_ck = QDoubleSpinBox(); self.sp_ck.setDecimals(2); self.sp_ck.setRange(0.0, 1e6); self.sp_ck.setValue(50.0); self.sp_ck.setMaximumWidth(80)
        self.sp_cnu = QDoubleSpinBox(); self.sp_cnu.setDecimals(2); self.sp_cnu.setRange(0.0, 1e6); self.sp_cnu.setValue(10.0); self.sp_cnu.setMaximumWidth(80)
        eg.addWidget(self.sp_ck, r, 1)
        eg.addWidget(self.sp_cnu, r, 2); r += 1
        env_box.setLayout(eg)
        col.addWidget(env_box)
        col.addStretch(1)

        # Connections: rod params -> debounced preview
        for w in (self.sp_length, self.sp_rbase, self.sp_rtip, self.sp_nelem,
                  self.sp_density, self.sp_E, self.sp_nu):
            w.valueChanged.connect(self._schedule_preview)
        self.cb_direction.currentIndexChanged.connect(self._schedule_preview)
        # Environment plane affects the preview immediately.
        self.chk_plane.toggled.connect(self._update_plane)
        self.sp_plane_z.valueChanged.connect(self._update_plane)

        wrap = QScrollArea()
        wrap.setWidgetResizable(True)
        wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrap.setWidget(container)
        return wrap

    # --- center panel: MuJoCo view + playback -------------------------------

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(4, 4, 4, 4)

        hint = QLabel("Drag: orbit   |   Right-drag / Shift+drag: pan   |   Scroll: zoom")
        hint.setStyleSheet("color:#888; font-size:11px;")
        hint.setAlignment(Qt.AlignCenter)
        v.addWidget(hint)

        v.addWidget(self.view, 1)

        controls = QHBoxLayout()
        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setIconSize(QSize(18, 18))
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._toggle_play)
        controls.addWidget(self.btn_play)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self._on_slider)
        controls.addWidget(self.frame_slider, 1)

        self.lbl_time = QLabel("t = 0.000 s")
        self.lbl_time.setMinimumWidth(90)
        controls.addWidget(self.lbl_time)

        btn_reset = QPushButton("Reset view")
        btn_reset.clicked.connect(self.view.reset_view)
        controls.addWidget(btn_reset)

        v.addLayout(controls)
        return panel

    # --- right panel: model tree + timing + run + io ------------------------

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        outer = QVBoxLayout(container)
        outer.setContentsMargins(6, 6, 6, 6)

        # ---- Fixtures (boundary conditions) ----
        fx_box = QGroupBox("Fixtures (boundary conditions)")
        fxl = QVBoxLayout(fx_box)
        self.list_fixtures = QListWidget()
        self.list_fixtures.setMaximumHeight(110)
        fxl.addWidget(self.list_fixtures)

        fx_edit = QGridLayout(); r = 0
        fx_edit.setColumnStretch(0, 0)
        fx_edit.setColumnStretch(1, 1)
        fx_edit.addWidget(QLabel("Type"), r, 0)
        self.cb_fx_kind = QComboBox()
        for k in FIXTURE_KINDS:
            self.cb_fx_kind.addItem(FIXTURE_KIND_LABELS[k], k)
        fx_edit.addWidget(self.cb_fx_kind, r, 1); r += 1
        fx_edit.addWidget(QLabel("Location"), r, 0)
        self.cb_fx_loc = QComboBox()
        for k in LOCATIONS:
            self.cb_fx_loc.addItem(LOCATION_LABELS[k], k)
        fx_edit.addWidget(self.cb_fx_loc, r, 1); r += 1
        fx_edit.addWidget(QLabel("Node id"), r, 0)
        self.sp_fx_node = QSpinBox(); self.sp_fx_node.setRange(0, 10000)
        fx_edit.addWidget(self.sp_fx_node, r, 1); r += 1
        fxl.addLayout(fx_edit)

        fx_btns = QHBoxLayout()
        b_add_fx = QPushButton("Add fixture")
        b_del_fx = QPushButton("Remove selected")
        b_add_fx.clicked.connect(self._add_fixture)
        b_del_fx.clicked.connect(self._remove_fixture)
        fx_btns.addWidget(b_add_fx); fx_btns.addWidget(b_del_fx)
        fxl.addLayout(fx_btns)
        outer.addWidget(fx_box)

        # ---- Loads (external forces) ----
        ld_box = QGroupBox("Loads (external forces)")
        ldl = QVBoxLayout(ld_box)
        self.list_loads = QListWidget()
        self.list_loads.setMaximumHeight(110)
        ldl.addWidget(self.list_loads)

        ld_edit = QGridLayout(); r = 0
        ld_edit.setColumnStretch(0, 0)
        ld_edit.setColumnStretch(1, 1)
        ld_edit.setColumnStretch(2, 1)
        ld_edit.setColumnStretch(3, 1)
        ld_edit.addWidget(QLabel("Type"), r, 0)
        self.cb_ld_kind = QComboBox()
        for k in LOAD_KINDS:
            self.cb_ld_kind.addItem(LOAD_KIND_LABELS[k], k)
        ld_edit.addWidget(self.cb_ld_kind, r, 1, 1, 3); r += 1

        ld_edit.addWidget(QLabel("Force x,y,z"), r, 0)
        self.sp_fx_ = QDoubleSpinBox(); self.sp_fx_.setDecimals(5); self.sp_fx_.setRange(-1e6, 1e6); self.sp_fx_.setValue(0.01); self.sp_fx_.setMinimumWidth(0)
        self.sp_fy_ = QDoubleSpinBox(); self.sp_fy_.setDecimals(5); self.sp_fy_.setRange(-1e6, 1e6); self.sp_fy_.setMinimumWidth(0)
        self.sp_fz_ = QDoubleSpinBox(); self.sp_fz_.setDecimals(5); self.sp_fz_.setRange(-1e6, 1e6); self.sp_fz_.setMinimumWidth(0)
        for _sb in (self.sp_fx_, self.sp_fy_, self.sp_fz_):
            _sb.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        ld_edit.addWidget(self.sp_fx_, r, 1); ld_edit.addWidget(self.sp_fy_, r, 2); ld_edit.addWidget(self.sp_fz_, r, 3); r += 1

        ld_edit.addWidget(QLabel("Location"), r, 0)
        self.cb_ld_loc = QComboBox()
        for k in LOCATIONS:
            self.cb_ld_loc.addItem(LOCATION_LABELS[k], k)
        self.cb_ld_loc.setCurrentIndex(1)  # end
        ld_edit.addWidget(self.cb_ld_loc, r, 1, 1, 2)
        self.sp_ld_node = QSpinBox(); self.sp_ld_node.setRange(0, 10000); self.sp_ld_node.setValue(25)
        ld_edit.addWidget(self.sp_ld_node, r, 3); r += 1

        ld_edit.addWidget(QLabel("Start / End (s)"), r, 0)
        self.sp_ld_ts = QDoubleSpinBox(); self.sp_ld_ts.setDecimals(3); self.sp_ld_ts.setRange(0.0, 1e4); self.sp_ld_ts.setValue(0.0)
        self.sp_ld_te = QDoubleSpinBox(); self.sp_ld_te.setDecimals(3); self.sp_ld_te.setRange(0.0, 1e4); self.sp_ld_te.setValue(1.0)
        ld_edit.addWidget(self.sp_ld_ts, r, 1); ld_edit.addWidget(self.sp_ld_te, r, 2, 1, 2); r += 1
        ldl.addLayout(ld_edit)

        ld_btns = QHBoxLayout()
        b_add_ld = QPushButton("Add load")
        b_del_ld = QPushButton("Remove selected")
        b_add_ld.clicked.connect(self._add_load)
        b_del_ld.clicked.connect(self._remove_load)
        ld_btns.addWidget(b_add_ld); ld_btns.addWidget(b_del_ld)
        ldl.addLayout(ld_btns)
        outer.addWidget(ld_box)

        # ---- Timing ----
        tim = QGroupBox("Simulation timing")
        gt = QGridLayout(); r = 0
        gt.setColumnStretch(0, 0)
        gt.setColumnStretch(1, 1)
        gt.addWidget(QLabel("Damping const"), r, 0)
        self.sp_damp = QDoubleSpinBox(); self.sp_damp.setDecimals(4); self.sp_damp.setRange(0.0, 1e4); self.sp_damp.setValue(0.25)
        gt.addWidget(self.sp_damp, r, 1); r += 1
        gt.addWidget(QLabel("Final time (s)"), r, 0)
        self.sp_final = QDoubleSpinBox(); self.sp_final.setDecimals(3); self.sp_final.setRange(0.001, 1e4); self.sp_final.setValue(2.0)
        gt.addWidget(self.sp_final, r, 1); r += 1
        gt.addWidget(QLabel("Time step (s)"), r, 0)
        self.sp_dt = QDoubleSpinBox(); self.sp_dt.setDecimals(7); self.sp_dt.setRange(1e-7, 1e-2); self.sp_dt.setValue(1e-5)
        gt.addWidget(self.sp_dt, r, 1); r += 1
        gt.addWidget(QLabel("Recording FPS"), r, 0)
        self.sp_fps = QSpinBox(); self.sp_fps.setRange(1, 240); self.sp_fps.setValue(30)
        gt.addWidget(self.sp_fps, r, 1); r += 1
        tim.setLayout(gt)
        outer.addWidget(tim)

        # ---- Run + progress ----
        self.btn_run = QPushButton("Run simulation")
        self.btn_run.clicked.connect(self._on_run)
        outer.addWidget(self.btn_run)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        outer.addWidget(self.progress)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        outer.addWidget(line)

        # ---- Save / load ----
        io = QGridLayout()
        b_savedata = QPushButton("Save data...")
        b_loaddata = QPushButton("Load .dat...")
        b_savecfg = QPushButton("Save config...")
        b_loadcfg = QPushButton("Load config...")
        b_savedata.clicked.connect(self._on_save_data)
        b_loaddata.clicked.connect(self._on_load_data)
        b_savecfg.clicked.connect(self._on_save_config)
        b_loadcfg.clicked.connect(self._on_load_config)
        io.addWidget(b_savedata, 0, 0); io.addWidget(b_loaddata, 0, 1)
        io.addWidget(b_savecfg, 1, 0); io.addWidget(b_loadcfg, 1, 1)
        outer.addLayout(io)
        outer.addStretch(1)

        wrap = QScrollArea()
        wrap.setWidgetResizable(True)
        wrap.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        wrap.setWidget(container)
        wrap.setMinimumWidth(300)
        wrap.setMaximumWidth(380)
        return wrap

    # --- model-tree list handling -------------------------------------------

    def _refresh_lists(self):
        self.list_fixtures.clear()
        for f in self._fixtures:
            self.list_fixtures.addItem(QListWidgetItem(f.label()))
        self.list_loads.clear()
        for ld in self._loads:
            self.list_loads.addItem(QListWidgetItem(ld.label()))

    def _add_fixture(self):
        self._fixtures.append(Fixture(
            kind=self.cb_fx_kind.currentData(),
            location=self.cb_fx_loc.currentData(),
            node_id=self.sp_fx_node.value(),
        ))
        self._refresh_lists()

    def _remove_fixture(self):
        row = self.list_fixtures.currentRow()
        if 0 <= row < len(self._fixtures):
            del self._fixtures[row]
            self._refresh_lists()

    def _add_load(self):
        self._loads.append(Load(
            kind=self.cb_ld_kind.currentData(),
            force=(self.sp_fx_.value(), self.sp_fy_.value(), self.sp_fz_.value()),
            location=self.cb_ld_loc.currentData(),
            node_id=self.sp_ld_node.value(),
            start_time=self.sp_ld_ts.value(),
            end_time=self.sp_ld_te.value(),
        ))
        self._refresh_lists()

    def _remove_load(self):
        row = self.list_loads.currentRow()
        if 0 <= row < len(self._loads):
            del self._loads[row]
            self._refresh_lists()

    # --- config <-> widgets --------------------------------------------------

    def rod_config(self) -> RodConfig:
        return RodConfig(
            length=self.sp_length.value(),
            radius_base=self.sp_rbase.value(),
            radius_tip=self.sp_rtip.value(),
            n_elements=self.sp_nelem.value(),
            density=self.sp_density.value(),
            youngs_modulus=self.sp_E.value(),
            poisson_ratio=self.sp_nu.value(),
            direction=self.cb_direction.currentText(),
        )

    def env_config(self) -> EnvConfig:
        return EnvConfig(
            gravity_on=self.chk_gravity.isChecked(),
            gravity_g=self.sp_g.value(),
            plane_on=self.chk_plane.isChecked(),
            plane_z=self.sp_plane_z.value(),
            plane_collision=self.chk_collision.isChecked(),
            collision_k=self.sp_ck.value(),
            collision_nu=self.sp_cnu.value(),
        )

    def sim_config(self) -> SimConfig:
        return SimConfig(
            damping_constant=self.sp_damp_value(),
            final_time=self.sp_final.value(),
            time_step=self.sp_dt.value(),
            recording_fps=self.sp_fps.value(),
            env=self.env_config(),
            fixtures=[Fixture(f.kind, f.location, f.node_id) for f in self._fixtures],
            loads=[Load(l.kind, tuple(l.force), l.location, l.node_id, l.start_time, l.end_time)
                   for l in self._loads],
        )

    def sp_damp_value(self) -> float:
        return self.sp_damp.value()

    def _apply_configs(self, rod: RodConfig, sim: SimConfig):
        blk = [self.sp_length, self.sp_rbase, self.sp_rtip, self.sp_nelem,
               self.sp_density, self.sp_E, self.sp_nu, self.cb_direction]
        for w in blk:
            w.blockSignals(True)
        self.sp_length.setValue(rod.length)
        self.sp_rbase.setValue(rod.radius_base)
        self.sp_rtip.setValue(rod.radius_tip)
        self.sp_nelem.setValue(rod.n_elements)
        self.sp_density.setValue(rod.density)
        self.sp_E.setValue(rod.youngs_modulus)
        self.sp_nu.setValue(rod.poisson_ratio)
        idx = self.cb_direction.findText(rod.direction)
        if idx >= 0:
            self.cb_direction.setCurrentIndex(idx)
        for w in blk:
            w.blockSignals(False)

        env = sim.env
        self.chk_gravity.setChecked(env.gravity_on)
        self.sp_g.setValue(env.gravity_g)
        self.chk_plane.setChecked(env.plane_on)
        self.sp_plane_z.setValue(env.plane_z)
        self.chk_collision.setChecked(env.plane_collision)
        self.sp_ck.setValue(env.collision_k)
        self.sp_cnu.setValue(env.collision_nu)

        self.sp_damp.setValue(sim.damping_constant)
        self.sp_final.setValue(sim.final_time)
        self.sp_dt.setValue(sim.time_step)
        self.sp_fps.setValue(sim.recording_fps)

        self._fixtures = [Fixture(f.kind, f.location, f.node_id) for f in sim.fixtures]
        self._loads = [Load(l.kind, tuple(l.force), l.location, l.node_id, l.start_time, l.end_time)
                       for l in sim.loads]
        self._refresh_lists()
        self.refresh_preview()

    # --- live preview --------------------------------------------------------

    def _schedule_preview(self, *_):
        self._preview_timer.start()

    def refresh_preview(self):
        cfg = self.rod_config()
        self.sp_fx_node.setMaximum(cfg.n_elements)
        self.sp_ld_node.setMaximum(cfg.n_elements)
        pos, rad = es.rod_geometry(cfg)
        self.view.set_static(pos, rad)
        self._update_plane()
        self.lbl_info.setText(
            f"taper {cfg.taper_ratio * 100:.1f}%   "
            f"base r={cfg.radius_base * 1000:.2f} mm   "
            f"tip r={cfg.radius_tip * 1000:.2f} mm   "
            f"L={cfg.length * 100:.1f} cm"
        )

    def _update_plane(self, *_):
        self.view.set_plane(self.chk_plane.isChecked(), self.sp_plane_z.value())

    # --- run -----------------------------------------------------------------

    def _on_run(self):
        if self._thread is not None:
            return
        self._stop_playback()
        rod = self.rod_config()
        sim = self.sim_config()

        self.btn_run.setEnabled(False)
        self.btn_run.setText("Running...")
        self.progress.setValue(0)

        self._thread = QThread(self)
        self._worker = SimWorker(rod, sim)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, frac):
        self.progress.setValue(int(frac * 100))

    def _cleanup_thread(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None
        self.btn_run.setEnabled(True)
        self.btn_run.setText("Run simulation")

    def _on_failed(self, msg):
        self._cleanup_thread()
        QMessageBox.critical(self, "Simulation failed", msg)

    def _on_finished(self, history, fps):
        self._cleanup_thread()
        self._load_history(history, fps)

    def _load_history(self, history, fps):
        if not history.get("time"):
            QMessageBox.warning(self, "No data", "Simulation produced no frames.")
            return
        self._history = history
        self._history_fps = fps
        positions, radii, times = es.history_to_arrays(history)
        self.view.set_trajectory(positions, radii, times)
        n = self.view.n_frames
        self.frame_slider.setRange(0, max(0, n - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(n > 1)
        self.btn_play.setEnabled(n > 1)
        self._update_time_label()

    # --- playback ------------------------------------------------------------

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            if self.frame_slider.value() >= self.frame_slider.maximum():
                self.frame_slider.setValue(0)
            fps = getattr(self, "_history_fps", 30)
            self._play_timer.start(int(1000 / max(1, fps)))
            self._playing = True
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def _stop_playback(self):
        self._play_timer.stop()
        self._playing = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _advance_frame(self):
        v = self.frame_slider.value()
        if v >= self.frame_slider.maximum():
            self._stop_playback()
            return
        self.frame_slider.setValue(v + 1)

    def _on_slider(self, value):
        self.view.show_frame(value)
        self._update_time_label()

    def _update_time_label(self):
        self.lbl_time.setText(f"t = {self.view.current_time():.3f} s")

    # --- save / load ---------------------------------------------------------

    def _on_save_data(self):
        if self._history is None:
            QMessageBox.information(self, "No data", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save simulation data", "simulation.dat", "Data files (*.dat)"
        )
        if not path:
            return
        try:
            es.save_dat(self._history, getattr(self, "_history_fps", 30), path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load simulation data", "", "Data files (*.dat);;All files (*)"
        )
        if not path:
            return
        try:
            history, fps = es.load_dat(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._stop_playback()
        self._load_history(history, fps)

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "config.xml", "XML files (*.xml)"
        )
        if not path:
            return
        try:
            to_xml(self.rod_config(), self.sim_config(), path)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", "", "XML files (*.xml);;All files (*)"
        )
        if not path:
            return
        try:
            rod, sim = from_xml(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self._apply_configs(rod, sim)

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.cancel()
        self._cleanup_thread()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
