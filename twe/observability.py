"""
Compute engine for observability / minimum error variance.

MPC trajectory simulation + stochastic/empirical observability & constructability
Gramians + EKF/UKF/AI-KF, for the two dynamical systems (fly-in-wind, drone). The
public entry points used by the analysis scripts are ``ObservabilityEngine``,
``SYSTEMS`` and ``build_setpoints``. The interactive ipywidgets notebook GUI that
used to live here now lives in ``twe/gui.py`` (optional ``gui`` extra), so this
module has no ipywidgets/IPython dependency.

What the engine provides
------------------------
* **Two dynamical systems** — the insect-in-wind model from the pybounds
  example (`fly_wind_example.ipynb`, 18 states) and the 3D kinematic drone
  (11 states, `drone_model.py`). The fly's default trajectory is *exactly*
  the pybounds example (MPC tracking v_para/v_perp ramps + a -90 deg heading
  step at t = 0.2 s in wind w = 0.4, zeta = 180 deg) for direct comparison.
* **Trajectory builder** — compose maneuvers from preset motifs (straight,
  turn, circle, speed change, sinusoidal weave) or quick flight-command
  buttons; control inputs always come from the pybounds MPC.
* **Observability panels** — the trajectory colored by min EV of a chosen
  state (linearized Eq. 33 | pybounds empirical, shared color scale), the
  min-EV time series, and min EV vs window length at a chosen start.
* **Interactive parameters** — window size `w`, empirical perturbation `ε`,
  regularizer `λ` (range restricted to stay above the floating-point noise
  floor of F), per-sensor `R` (default 1e-3 each), per-state `Q` presets.

Caching: the linearized pipeline (~0.1 s) recomputes on every change; the
pybounds empirical pipeline is cached per (trajectory, w, ε, R, λ).
"""

import io
import sys
import time
import contextlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .drone_model import (f as drone_f, h as drone_h,
                         STATE_NAMES as DRONE_STATES,
                         INPUT_NAMES as DRONE_INPUTS,
                         MEASUREMENT_NAMES as DRONE_MEAS)
# the AI-KF's motif -> R map (vendored alongside this module; ann_estimator.py).
from .ann_estimator import motif_R                      # noqa: E402

from .stochastic_observability import (linearize_along_trajectory,
                                      stochastic_observability_gramian,
                                      stochastic_constructability_gramian)
from pybounds import (Simulator, SlidingEmpiricalObservabilityMatrix,
                      SlidingFisherObservability, EmpiricalObservabilityMatrix,
                      colorline)



# ───────────── fly-in-wind model (verbatim from fly_wind_example) ─────────────

FLY_STATES = ['x', 'y', 'z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta',
              'm', 'I', 'C_para', 'C_perp', 'C_phi', 'km1', 'km2', 'km3', 'km4']
FLY_INPUTS = ['u_para', 'u_perp', 'u_phi']
FLY_MEAS = ['phi', 'psi', 'gamma', 'a', 'g', 'r']

FLY_M = 0.25e-6 * 1e6                     # [mg]
FLY_I = 5.2e-13 * 1e6 * (1e3) ** 2        # [mg*mm^2]
FLY_C_PHI = 27.36e-12 * 1e6 * (1e3) ** 2
FLY_C_PARA = (0.25e-6 / 0.170) * 1e6


def fly_f(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta, m, I, C_para, C_perp, C_phi, km1, km2, km3, km4 = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    v_para_dot = ((km1 * u_para - C_para * a_para) / m) + (v_perp * phi_dot)
    v_perp_dot = ((km3 * u_perp - C_perp * a_perp) / m) - (v_para * phi_dot)
    phi_ddot = (km4 * u_phi / I) - (C_phi * phi_dot / I) + (km2 * u_para / I)
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)
    z0 = 0 * x
    return [x_dot, y_dot, z0, v_para_dot, v_perp_dot, phi_dot, phi_ddot,
            z0, z0, z0, z0, z0, z0, z0, z0, z0, z0, z0]


def fly_h(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta, m, I, C_para, C_perp, C_phi, km1, km2, km3, km4 = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)
    g = np.sqrt(v_para ** 2 + v_perp ** 2)
    psi = np.arctan2(v_perp, v_para)
    r = g / z
    if np.array(phi).ndim > 0 and np.array(phi).shape[0] > 1:
        phi = np.unwrap(phi); psi = np.unwrap(psi); gamma = np.unwrap(gamma)
    return [phi, psi, gamma, a, g, r]


# ────── fly-in-wind, 7 primary states (constants folded into parameters) ──────
# Same dynamics/measurements as fly_f/fly_h, but m, I, C_para, C_perp, C_phi and
# km1-4 are fixed parameters instead of states (cf. drone_model.py). This keeps
# the process-noise covariance Q genuinely positive definite for the recursive
# stochastic Gramian — no jitter needed on constant parameter states — at the
# cost of assuming perfect calibration. States: the CDC-2021 paper's 7 primary
# states plus x, y for trajectory plotting.

FLY7_STATES = ['x', 'y', 'z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta']
FLY_KM1, FLY_KM2, FLY_KM3, FLY_KM4 = 1.0, 0.0, 1.0, 1.0


def fly7_f(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta = X
    u_para, u_perp, u_phi = U
    m, I = FLY_M, FLY_I
    C_para, C_perp, C_phi = FLY_C_PARA, FLY_C_PARA, FLY_C_PHI
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    v_para_dot = ((FLY_KM1 * u_para - C_para * a_para) / m) + (v_perp * phi_dot)
    v_perp_dot = ((FLY_KM3 * u_perp - C_perp * a_perp) / m) - (v_para * phi_dot)
    phi_ddot = (FLY_KM4 * u_phi / I) - (C_phi * phi_dot / I) + (FLY_KM2 * u_para / I)
    x_dot = v_para * np.cos(phi) - v_perp * np.sin(phi)
    y_dot = v_para * np.sin(phi) + v_perp * np.cos(phi)
    z0 = 0 * x
    return [x_dot, y_dot, z0, v_para_dot, v_perp_dot, phi_dot, phi_ddot, z0, z0]


def fly7_h(X, U):
    x, y, z, v_para, v_perp, phi, phi_dot, w, zeta = X
    u_para, u_perp, u_phi = U
    a_para = v_para - w * np.cos(phi - zeta)
    a_perp = v_perp + w * np.sin(phi - zeta)
    a = np.sqrt(a_para ** 2 + a_perp ** 2)
    gamma = np.arctan2(a_perp, a_para)
    g = np.sqrt(v_para ** 2 + v_perp ** 2)
    psi = np.arctan2(v_perp, v_para)
    r = g / z
    if np.array(phi).ndim > 0 and np.array(phi).shape[0] > 1:
        phi = np.unwrap(phi); psi = np.unwrap(psi); gamma = np.unwrap(gamma)
    return [phi, psi, gamma, a, g, r]


# ── altitude 2D model (Cellini et al. 2025, Fig. 4, Eq. 6-7; + x for plots) ──
# States [x, z, v_z, v_x]; inputs are the (measured) accelerations u_z, u_x;
# the only measurement is forward ventral optic flow r_x = v_x / z. Altitude
# is observable only during nonzero acceleration (paper Fig. 2c-v / Fig. 4b).

ALT2D_STATES = ['x', 'z', 'v_z', 'v_x']
ALT2D_INPUTS = ['u_z', 'u_x']
ALT2D_MEAS = ['r_x']


def alt2d_f(X, U):
    x, z, v_z, v_x = X
    u_z, u_x = U
    return [v_x, v_z, u_z, u_x]


def alt2d_h(X, U):
    x, z, v_z, v_x = X
    return [v_x / z]


# ───────────────────────── generic RK4 simulator ──────────────────────────

class GenericRK4Simulator:
    """ Minimal fixed-step RK4 simulator, duck-typed for pybounds'
    EmpiricalObservabilityMatrix / SlidingEmpiricalObservabilityMatrix. """

    def __init__(self, f, h, dt, state_names, input_names, measurement_names):
        self.f, self.h, self.dt = f, h, dt
        self.state_names = list(state_names)
        self.input_names = list(input_names)
        self.measurement_names = list(measurement_names)
        self.n, self.m, self.p = (len(self.state_names), len(self.input_names),
                                  len(self.measurement_names))

    def _rk4_step(self, x, u):
        dt, f = self.dt, self.f
        k1 = np.asarray(f(x, u), dtype=float)
        k2 = np.asarray(f(x + 0.5 * dt * k1, u), dtype=float)
        k3 = np.asarray(f(x + 0.5 * dt * k2, u), dtype=float)
        k4 = np.asarray(f(x + dt * k3, u), dtype=float)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def simulate(self, x0=None, u=None, aux=None):
        if isinstance(x0, dict):
            x0 = np.array([x0[k] for k in self.state_names], dtype=float)
        else:
            x0 = np.array(x0, dtype=float).squeeze()
        if isinstance(u, dict):
            u = np.vstack([u[k] for k in self.input_names]).T
        u = np.atleast_2d(np.array(u, dtype=float))
        w = u.shape[0]
        x = np.zeros((w, self.n))
        y = np.zeros((w, self.p))
        x[0] = x0
        y[0] = np.asarray(self.h(x[0], u[0]), dtype=float)
        for k in range(w - 1):
            x[k + 1] = self._rk4_step(x[k], u[k])
            y[k + 1] = np.asarray(self.h(x[k + 1], u[k + 1]), dtype=float)
        return y


# ───────────────────────────── system specs ────────────────────────────────

class SystemSpec:
    """ Everything the engine/GUI needs to know about one dynamical system. """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_fly_spec(dt=0.01):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 0.2 * ones,
                'v_para': speed_sp, 'v_perp': 0.0 * ones,
                'phi': heading_sp, 'phi_dot': 0.0 * ones,
                'w': w_wind * ones, 'zeta': zeta * ones,
                'm': FLY_M * ones, 'I': FLY_I * ones,
                'C_para': FLY_C_PARA * ones, 'C_perp': FLY_C_PARA * ones,
                'C_phi': FLY_C_PHI * ones,
                'km1': 1.0 * ones, 'km2': 0.0 * ones,
                'km3': 1.0 * ones, 'km4': 1.0 * ones}

    def default_setpoint(w_wind, zeta):
        # exactly the pybounds fly_wind_example trajectory set-points
        tsim = np.arange(0, 0.4, step=dt)
        sp = make_setpoint(0.3 * np.ones_like(tsim) + 0.01 * tsim,
                           (np.pi / 4) * np.ones_like(tsim), w_wind, zeta)
        sp['v_perp'] = 0.01 * np.ones_like(tsim) + 0.01 * tsim
        k_turn = min(int(round(0.2 / dt)), len(tsim) - 1)
        sp['phi'][k_turn:] = sp['phi'][k_turn] - np.pi / 2   # turn at t = 0.2 s
        return sp

    def build_mpc(sim):
        cost = ((sim.model.x['v_para'] - sim.model.tvp['v_para_set']) ** 2 +
                (sim.model.x['v_perp'] - sim.model.tvp['v_perp_set']) ** 2 +
                (sim.model.x['phi'] - sim.model.tvp['phi_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_para=1e-6, u_perp=1e-6, u_phi=1e-6)

    # NB these are PER-STEP Q_d [units²] since 2026-08-17, when the whole Q
    # path moved to Q_d. They are the previous PSD values x dt, so the effective
    # process noise is unchanged — only what the number means changed.
    q_tiny = {s: 1e-10 for s in FLY_STATES}         # jitter floor (see fly nb §4b)
    q_real = dict(q_tiny)
    q_real.update(dict(v_para=2e-4, v_perp=2e-4, phi=1e-7, phi_dot=5e-2,
                       w=2e-3, zeta=2e-3))
    return SystemSpec(
        name='fly', dt=dt, f=fly_f, h=fly_h,
        fim_sensors=('phi', 'psi', 'gamma'),
        fim_states=('v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta',
                    'z', 'm', 'I', 'C_para', 'C_perp', 'C_phi'),
        state_names=FLY_STATES, input_names=FLY_INPUTS,
        measurement_names=FLY_MEAS,
        disc_method='rk4var',      # frozen-Jacobian expm is off during the turn
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.4, zeta_default=np.pi,
        v0_default=0.3, heading0=np.pi / 4,
        w_default=4, eps_default=1e-4,
        r_default={s: 1e-3 for s in FLY_MEAS},
        q_tiny=q_tiny, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('phi', 'zeta', 'v_para', 'phi_dot'),
        angle_measurements=('phi', 'psi', 'gamma'),
        angle_states=('phi', 'zeta'),
        seg_duration=0.1, quick_turn_dur=0.06, quick_straight_dur=0.12,
        speed_step=0.1, rec_xlim=(-1.0, 1.0), rec_ylim=(-0.5, 0.5),
        default_traj_label='pybounds fly_wind_example trajectory '
                           '(hold π/4, −π/2 step at t = 0.2 s)')


def make_fly7_spec(dt=0.01):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 0.2 * ones,
                'v_para': speed_sp, 'v_perp': 0.0 * ones,
                'phi': heading_sp, 'phi_dot': 0.0 * ones,
                'w': w_wind * ones, 'zeta': zeta * ones}

    def default_setpoint(w_wind, zeta):
        # same trajectory set-points as the pybounds fly_wind_example
        tsim = np.arange(0, 0.4, step=dt)
        sp = make_setpoint(0.3 * np.ones_like(tsim) + 0.01 * tsim,
                           (np.pi / 4) * np.ones_like(tsim), w_wind, zeta)
        sp['v_perp'] = 0.01 * np.ones_like(tsim) + 0.01 * tsim
        k_turn = min(int(round(0.2 / dt)), len(tsim) - 1)
        sp['phi'][k_turn:] = sp['phi'][k_turn] - np.pi / 2
        return sp

    def build_mpc(sim):
        cost = ((sim.model.x['v_para'] - sim.model.tvp['v_para_set']) ** 2 +
                (sim.model.x['v_perp'] - sim.model.tvp['v_perp_set']) ** 2 +
                (sim.model.x['phi'] - sim.model.tvp['phi_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_para=1e-6, u_perp=1e-6, u_phi=1e-6)

    # NB these are PER-STEP Q_d [units²] since 2026-08-17, when the whole Q
    # path moved to Q_d. They are the previous PSD values x dt, so the effective
    # process noise is unchanged — only what the number means changed.
    q_tiny = {s: 1e-11 for s in FLY7_STATES}
    q_real = dict(q_tiny)
    q_real.update(dict(v_para=2e-4, v_perp=2e-4, phi=1e-7, phi_dot=5e-2,
                       w=2e-3, zeta=2e-3))
    return SystemSpec(
        name='fly7', dt=dt, f=fly7_f, h=fly7_h,
        fim_sensors=('phi', 'psi', 'gamma'),
        fim_states=('z', 'v_para', 'v_perp', 'phi', 'phi_dot', 'w', 'zeta'),
        state_names=FLY7_STATES, input_names=FLY_INPUTS,
        measurement_names=FLY_MEAS,
        disc_method='rk4var',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.4, zeta_default=np.pi,
        v0_default=0.3, heading0=np.pi / 4,
        w_default=4, eps_default=1e-4,
        r_default={s: 1e-3 for s in FLY_MEAS},
        q_tiny=q_tiny, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('phi', 'zeta', 'v_para', 'phi_dot'),
        angle_measurements=('phi', 'psi', 'gamma'),
        angle_states=('phi', 'zeta'),
        seg_duration=0.1, quick_turn_dur=0.06, quick_straight_dur=0.12,
        speed_step=0.1, rec_xlim=(-1.0, 1.0), rec_ylim=(-0.5, 0.5),
        default_traj_label='pybounds fly_wind_example trajectory '
                           '(hold π/4, −π/2 step at t = 0.2 s)')


def make_alt2d_spec(dt=0.1):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        # heading and wind are meaningless for this 1-D longitudinal model
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'z': 1.5 * ones, 'v_z': 0.0 * ones,
                'v_x': speed_sp}

    def default_setpoint(w_wind, zeta):
        # paper Fig. 4b-style profile: straight flight at z = 1.5 m with a
        # brief 3 s deceleration (-1 m/s^2) and a later 3 s acceleration
        tsim = np.arange(0.0, 50.0 + dt / 2, dt)
        v = np.full_like(tsim, 10.0)
        m1 = (tsim >= 12.0) & (tsim < 15.0)
        v[m1] = 10.0 - 1.0 * (tsim[m1] - 12.0)
        v[(tsim >= 15.0) & (tsim < 35.0)] = 7.0
        m2 = (tsim >= 35.0) & (tsim < 38.0)
        v[m2] = 7.0 + 1.0 * (tsim[m2] - 35.0)
        v[tsim >= 38.0] = 10.0
        return make_setpoint(v, None, w_wind, zeta)

    def build_mpc(sim):
        mx, mtvp = sim.model.x, sim.model.tvp
        cost = ((mx['z'] - mtvp['z_set']) ** 2 +
                (mx['v_z'] - mtvp['v_z_set']) ** 2 +
                (mx['v_x'] - mtvp['v_x_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.set_rterm(u_z=1e-4, u_x=1e-4)

    return SystemSpec(
        name='alt2d', dt=dt, f=alt2d_f, h=alt2d_h,
        fim_sensors=('r_x',), fim_states=('z', 'v_z', 'v_x'),
        state_names=ALT2D_STATES, input_names=ALT2D_INPUTS,
        measurement_names=ALT2D_MEAS,
        disc_method='vanloan',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=0.0, zeta_default=0.0,
        v0_default=10.0, heading0=0.0,
        w_default=20, eps_default=1e-5,
        r_default={'r_x': 1e-3},   # paper Methods: optic-flow baseline 10^-3
        u_noise_default=1e-2,      # paper Fig. 4b: measured-accel variance 10^-2
        u_bias_default=('u_z', 0.25, 20.0, 28.0),  # biased v-accel episode
        q_tiny={s: 1e-11 for s in ALT2D_STATES},        # per-step Q_d (= old PSD*dt)
        q_realistic={'x': 1e-8, 'z': 1e-6, 'v_z': 1e-4, 'v_x': 1e-4},
        color_state_default='z',
        ev_states_default=('z', 'v_x'),
        angle_measurements=(), angle_states=(),
        seg_duration=3.0, quick_turn_dur=3.0, quick_straight_dur=5.0,
        speed_step=1.0, rec_xlim=(-250.0, 250.0), rec_ylim=(-125.0, 125.0),
        default_traj_label='paper Fig. 4b: straight flight at z = 1.5 m, '
                           '3 s decel @ t=12, 3 s accel @ t=35 (±1 m/s²)')


def make_drone_spec(dt=0.1):

    def make_setpoint(speed_sp, heading_sp, w_wind, zeta):
        ones = np.ones_like(speed_sp)
        return {'x': 0.0 * ones, 'y': 0.0 * ones, 'z': 10.0 * ones,
                'v_x': speed_sp, 'v_y': 0.0 * ones, 'v_z': 0.0 * ones,
                'phi': 0.0 * ones, 'theta': 0.0 * ones, 'psi': heading_sp,
                'w': w_wind * ones, 'zeta': zeta * ones}

    def default_setpoint(w_wind, zeta):
        speed_sp, heading_sp = build_setpoints(DRONE_DEFAULT_SEGMENTS, dt,
                                               v0=2.0, heading0=0.0)
        return make_setpoint(speed_sp, heading_sp, w_wind, zeta)

    def build_mpc(sim):
        mx, mtvp = sim.model.x, sim.model.tvp
        cost = ((mx['v_x'] - mtvp['v_x_set']) ** 2 +
                (mx['v_y'] - mtvp['v_y_set']) ** 2 +
                (mx['z'] - mtvp['z_set']) ** 2 +
                (mx['psi'] - mtvp['psi_set']) ** 2 +
                (mx['w'] - mtvp['w_set']) ** 2 +
                (mx['zeta'] - mtvp['zeta_set']) ** 2)
        sim.mpc.set_objective(mterm=cost, lterm=cost)
        sim.mpc.bounds['lower', '_x', 'z'] = 0
        sim.mpc.bounds['lower', '_x', 'phi'] = -np.pi / 4
        sim.mpc.bounds['upper', '_x', 'phi'] = np.pi / 4
        sim.mpc.bounds['lower', '_x', 'theta'] = -np.pi / 4
        sim.mpc.bounds['upper', '_x', 'theta'] = np.pi / 4
        sim.mpc.bounds['lower', '_u', 'u_thrust'] = 0
        sim.mpc.set_rterm(u_thrust=1e-4, u_phi=1e-4, u_theta=1e-4,
                          u_psi=1e-2, u_w=1e-2, u_zeta=1e-2)

    # NB these are PER-STEP Q_d [units²] since 2026-08-17, when the whole Q
    # path moved to Q_d. They are the previous PSD values x dt, so the effective
    # process noise is unchanged — only what the number means changed.
    q_real = {'x': 1e-8, 'y': 1e-8, 'z': 1e-6, 'v_x': 5e-4, 'v_y': 5e-4,
              'v_z': 1e-4, 'phi': 1e-6, 'theta': 1e-6, 'psi': 1e-6,
              'w': 1e-4, 'zeta': 1e-4}
    return SystemSpec(
        name='drone', dt=dt, f=drone_f, h=drone_h,
        fim_sensors=tuple(DRONE_MEAS), fim_states=tuple(DRONE_STATES),
        state_names=DRONE_STATES, input_names=DRONE_INPUTS,
        measurement_names=DRONE_MEAS,
        disc_method='vanloan',
        make_setpoint=make_setpoint, default_setpoint=default_setpoint,
        build_mpc=build_mpc,
        wind_default=1.2, zeta_default=np.pi / 4,
        v0_default=2.0, heading0=0.0,
        w_default=20, eps_default=1e-5,
        r_default={s: 1e-3 for s in DRONE_MEAS},
        q_tiny={s: 1e-14 for s in DRONE_STATES}, q_realistic=q_real,
        color_state_default='zeta',
        ev_states_default=('w', 'zeta', 'z', 'v_x'),
        angle_measurements=('beta', 'gamma', 'psi_m'),
        angle_states=('phi', 'theta', 'psi', 'zeta'),
        seg_duration=3.0, quick_turn_dur=2.0, quick_straight_dur=3.0,
        speed_step=1.0, rec_xlim=(-120.0, 120.0), rec_ylim=(-60.0, 60.0),
        default_traj_label='straight → −π/2 turn → straight → accelerate '
                           '→ straight')


SYSTEMS = {'fly': make_fly_spec, 'fly7': make_fly7_spec,
           'drone': make_drone_spec, 'alt2d': make_alt2d_spec}

DRONE_DEFAULT_SEGMENTS = [
    dict(motif='straight', duration=4.0),
    dict(motif='turn', duration=3.0, angle=-np.pi / 2),
    dict(motif='straight', duration=3.0),
    dict(motif='speed', duration=3.0, speed=4.0),
    dict(motif='straight', duration=3.0),
]


# ─────────────────────────── trajectory building ───────────────────────────

def build_setpoints(segments, dt, v0, heading0=0.0):
    """ Compile a list of motif segments into per-step (speed, heading)
    set-point arrays. Each segment dict has keys: motif ('straight' | 'turn' |
    'circle' | 'speed' | 'weave'), duration [s], and per-motif parameters
    angle [deg], speed [m/s], freq [Hz], amp [deg]. Set-points continue from
    where the previous segment ends. """
    v, hd = [v0], [heading0]
    for seg in segments:
        n_k = max(int(round(seg['duration'] / dt)), 1)
        tau = np.arange(1, n_k + 1) * dt
        v_prev, h_prev = v[-1], hd[-1]
        motif = seg['motif']
        if motif == 'straight':
            v += [v_prev] * n_k
            hd += [h_prev] * n_k
        elif motif in ('turn', 'circle'):
            dh = seg['angle']
            v += [v_prev] * n_k
            hd += list(h_prev + dh * tau / tau[-1])
        elif motif == 'speed':
            v += list(v_prev + (seg['speed'] - v_prev) * tau / tau[-1])
            hd += [h_prev] * n_k
        elif motif == 'weave':
            amp = seg['amp']
            v += [v_prev] * n_k
            hd += list(h_prev + amp * np.sin(2 * np.pi * seg['freq'] * tau))
        else:
            raise ValueError(f'unknown motif {motif!r}')
    return np.array(v), np.array(hd)


# ───────────────────────────── compute engine ──────────────────────────────

class ObservabilityEngine:
    """ MPC trajectory simulation + both EV pipelines for one SystemSpec,
    with caching keyed on the trajectory version. """

    def __init__(self, spec):
        self.spec = spec
        self.dt = spec.dt
        self.t = self.X = self.U = None
        self.N = 0
        self.truth_last = None    # truth of the last sampled estimation problem
        self._version = 0
        self._lin_cache = {}      # (version, Qc key)             -> (Phis, Cs, Qds)
        self._seom_cache = {}     # (version, w, eps)             -> SEOM
        self._ev_emp_cache = {}   # (version, w, eps, R, lam)     -> EV DataFrame
        self._eom_cache = {}      # (version, k0, W, eps)         -> EOM
        self._emp_vs_w_cache = {} # (version, k0, W, eps, R, lam) -> EV DataFrame
        self._mc_cache = {}       # nonlinear-MC sliding EV DataFrames
        self._mc_vs_w_cache = {}  # nonlinear-MC EV-vs-window DataFrames
        self._lin_ev_cache = {}   # linearized sliding EV DataFrames
        # trained ANNs for the AI-KF, keyed on (spec, target, shape). Populated
        # from OUTSIDE (zeta_ann_estimator.attach); never trained here.
        self._ann_cache = {}
        self._filt_cache = {}     # (filter, Q, R, seed, scale) -> (X_hat, P_diag)
        self._mpc_sim = None
        self.lam_noise_floor = 0.0

    def _rk4_sim(self):
        s = self.spec
        return GenericRK4Simulator(s.f, s.h, s.dt, s.state_names,
                                   s.input_names, s.measurement_names)

    # -- trajectory ----------------------------------------------------------
    def set_trajectory(self, t, X, U):
        self.t, self.X, self.U = t, X, U
        self.N = len(t)
        self.truth_last = None
        self._version += 1
        for c in (self._lin_cache, self._seom_cache, self._ev_emp_cache,
                  self._eom_cache, self._emp_vs_w_cache, self._mc_cache,
                  self._mc_vs_w_cache, self._lin_ev_cache, self._filt_cache):
            c.clear()

    def _get_mpc_sim(self):
        if self._mpc_sim is None:
            s = self.spec
            sim = Simulator(s.f, s.h, dt=s.dt, state_names=s.state_names,
                            input_names=s.input_names,
                            measurement_names=s.measurement_names,
                            mpc_horizon=10)
            s.build_mpc(sim)
            self._mpc_sim = sim
        return self._mpc_sim

    def simulate_mpc(self, setpoint):
        """ Closed-loop MPC rollout (x0 defaults to the t=0 set-point). """
        sim = self._get_mpc_sim()
        sim.update_dict(setpoint, name='setpoint')
        t, x_sim, u_sim, _ = sim.simulate(x0=None, mpc=True,
                                          return_full_output=True)
        X = np.vstack(list(x_sim.values())).T
        U = np.vstack(list(u_sim.values())).T
        self.set_trajectory(t, X, U)
        return t, X, U

    # -- linearized pipeline (fast) -------------------------------------------
    def _lin(self, q_diag, q_noise='uncorrelated'):
        s = self.spec
        key = (self._version, q_noise,
               tuple(f'{q_diag[n]:.3e}' for n in s.state_names))
        if key not in self._lin_cache:
            # q_diag IS the per-step discrete covariance Q_d — the Q of the
            # discrete Eq.30/33 recursions and of the Kalman update alike. No dt
            # scaling and no continuous PSD anywhere in this path.
            Qd = np.diag([q_diag[n] for n in s.state_names])
            self._lin_cache[key] = linearize_along_trajectory(
                s.f, s.h, self.X, self.U, self.dt, Qd,
                method=s.disc_method, q_noise=q_noise)
        return self._lin_cache[key]

    def _fim_subsets(self, sensors, states):
        s = self.spec
        sensors = list(sensors) if sensors else list(s.measurement_names)
        states = list(states) if states else list(s.state_names)
        sidx = [s.measurement_names.index(m) for m in sensors]
        sub = np.array([s.state_names.index(n) for n in states], dtype=int)
        return sensors, states, sidx, sub

    def _ev_diag(self, F, lam):
        """ diag((F + lam I)^-1) via eigendecomposition — never raises on
        singular input. Eigenvalues below the floating-point noise floor of F
        (n * eps * lam_max) are numerically indistinguishable from zero and are
        truncated to 0, so unresolvable directions honestly report 1/lam
        instead of round-off garbage. """
        evals, evecs = np.linalg.eigh(F)
        self._last_fmax = float(max(evals[-1], 0.0))
        tol = F.shape[0] * np.finfo(float).eps * self._last_fmax
        evals = np.where(evals > tol, evals, 0.0)
        return ((evecs ** 2) / (evals + lam)).sum(axis=1)

    def linearized_ev(self, w, q_diag, r_diag, lam, sensors=None, states=None,
                      basis='observability', q_noise='uncorrelated'):
        """ Sliding min EV via the recursive Gramian, using only the selected
        sensors (rows of C), reported for the selected states.

        The FULL Gramian is inverted and the selected states' diagonal entries are
        then read off — the MARGINAL error variance, i.e. every other state is also
        unknown and jointly estimated. That is what the manuscript's Eq. (29) means:
        the Gramians appear as blocks of ℱ_w^{-1}, so a single state's bound is a
        diagonal entry of the INVERTED Gramian. Cropping the Gramian to the
        selection first would instead give the 'all other states known' conditional
        bound, which is smaller and would make one state's number depend on which
        OTHER states happen to be selected. (NB pybounds' FisherObservability does
        crop — it column-slices O before forming O^T R^-1 O — so its numbers are the
        conditional ones and agree with these only when ALL states are selected.)

        `basis` picks which state the bound refers to:
          'observability'    Eq. (33), the window's INITIAL state — the CRB for a
                             fixed-point smoother of x_0.
          'constructability' Eq. (30), the window's FINAL state — the a-posteriori
                             CRB for a causal filter of the current state, which is
                             the right metric when the question is about process
                             noise.
        Both are reported at the window CENTRE in the `time` column
        (time_initial + round(w/2)·dt, matching pybounds' shift_index), so the two
        curves share one x-axis; `time_initial` (window start) is kept alongside.
        The state each bound refers to still differs — observability the window's
        first state, constructability its last.

        Also records `lam_noise_floor` = machine-eps * max eigenvalue of the
        inverted block: lam below this cannot guarantee PD in floating point. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        key = (self._version, w, q_noise,
               tuple(f'{q_diag[k]:.3e}' for k in self.spec.state_names),
               tuple(f'{r_diag[m]:.3e}' for m in sensors), f'{lam:.3e}',
               tuple(sensors), tuple(states), basis)
        if key in self._lin_ev_cache:
            df, self.lam_noise_floor = self._lin_ev_cache[key]
            return df.copy()
        gramian = (stochastic_constructability_gramian
                   if basis == 'constructability'
                   else stochastic_observability_gramian)
        Phis, Cs, Qds = self._lin(q_diag, q_noise)
        Cs_sel = [C[sidx, :] for C in Cs]
        Rinv = np.diag(1.0 / np.array([r_diag[m] for m in sensors]))
        Rinvs = [Rinv] * w
        starts = np.arange(0, self.N - w + 1)
        EV, fmax = [], 0.0
        for k0 in starts:
            F = gramian(Phis[k0:k0 + w], Cs_sel[k0:k0 + w], Qds[k0:k0 + w], Rinvs)
            # invert the FULL Gramian, then read the selected states (marginal —
            # see the docstring); do NOT crop F first
            EV.append(self._ev_diag(F, lam)[sub])
            fmax = max(fmax, self._last_fmax)
        self.lam_noise_floor = np.finfo(float).eps * fmax
        df = pd.DataFrame(np.array(EV), columns=states)
        df.insert(0, 'time_initial', self.t[starts])
        # window CENTRE, matching pybounds SlidingFisherObservability.shift_index
        df.insert(1, 'time', self.t[starts] + int(round(0.5 * w)) * self.dt)
        self._lin_ev_cache[key] = (df, self.lam_noise_floor)
        return df.copy()

    def linearized_ev_vs_window(self, k0, ws, q_diag, r_diag, lam,
                                sensors=None, states=None,
                                q_noise='uncorrelated'):
        """ Min EV for growing window lengths starting at k0 (same sensor /
        state restrictions as linearized_ev). """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        Phis, Cs, Qds = self._lin(q_diag, q_noise)
        Cs_sel = [C[sidx, :] for C in Cs]
        Rinv = np.diag(1.0 / np.array([r_diag[m] for m in sensors]))
        EV = []
        for wv in ws:
            F = stochastic_observability_gramian(
                Phis[k0:k0 + wv], Cs_sel[k0:k0 + wv], Qds[k0:k0 + wv],
                [Rinv] * wv)
            EV.append(self._ev_diag(F, lam)[sub])   # marginal; see linearized_ev
        return pd.DataFrame(EV, columns=states, index=np.array(ws) * self.dt)

    # -- pybounds empirical pipeline (slow, cached) ----------------------------
    def empirical_ev(self, w, eps, r_diag, lam, sensors=None, states=None):
        s = self.spec
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        r_key = tuple(f'{r_diag[m]:.3e}' for m in s.measurement_names)
        ev_key = (self._version, w, f'{eps:.3e}', r_key, f'{lam:.3e}',
                  tuple(sensors), tuple(states))
        if ev_key in self._ev_emp_cache:
            return self._ev_emp_cache[ev_key], True
        seom_key = (self._version, w, f'{eps:.3e}')
        if seom_key not in self._seom_cache:
            self._seom_cache[seom_key] = SlidingEmpiricalObservabilityMatrix(
                self._rk4_sim(), self.t, self.X, self.U,
                w=w, eps=eps, parallel_sliding=True)
        SEOM = self._seom_cache[seom_key]
        SFO = SlidingFisherObservability(SEOM.O_df_sliding, R=dict(r_diag),
                                         lam=lam, time=SEOM.O_time,
                                         sensors=list(sensors),
                                         states=list(states),
                                         time_steps=np.arange(0, w), w=None)
        EV = SFO.get_minimum_error_variance()
        self._ev_emp_cache[ev_key] = EV
        return EV, False

    def empirical_ev_vs_window(self, k0, ws, eps, r_diag, lam,
                               sensors=None, states=None):
        """ Min EV vs window length at fixed start k0, from the pybounds
        empirical O: one EmpiricalObservabilityMatrix over the longest window,
        then F = O^T R^{-1} O restricted to the first wv time steps and the
        selected sensors / states (pybounds FisherObservability semantics). """
        s = self.spec
        sensors, states, _, _ = self._fim_subsets(sensors, states)
        W_max = int(max(ws))
        r_key = tuple(f'{r_diag[m]:.3e}' for m in s.measurement_names)
        key = (self._version, k0, W_max, f'{eps:.3e}', r_key, f'{lam:.3e}',
               tuple(sensors), tuple(states))
        if key in self._emp_vs_w_cache:
            return self._emp_vs_w_cache[key]
        eom_key = (self._version, k0, W_max, f'{eps:.3e}')
        if eom_key not in self._eom_cache:
            self._eom_cache[eom_key] = EmpiricalObservabilityMatrix(
                self._rk4_sim(), self.X[k0], self.U[k0:k0 + W_max], eps=eps)
        O_df = self._eom_cache[eom_key].O_df
        O = O_df[list(states)].values
        tsteps = O_df.index.get_level_values('time_step').values
        snames = O_df.index.get_level_values('sensor')
        smask = np.array([m in set(sensors) for m in snames])
        rinv = np.array([1.0 / r_diag[m] for m in snames])
        EV = []
        for wv in ws:
            rows = smask & (tsteps < wv)
            Om = O[rows]
            F = (Om * rinv[rows, None]).T @ Om
            evals, evecs = np.linalg.eigh(F)
            EV.append(((evecs ** 2) / (np.clip(evals, 0.0, None) + lam))
                      .sum(axis=1))
        df = pd.DataFrame(EV, columns=list(states),
                          index=np.array(ws) * self.dt)
        self._emp_vs_w_cache[key] = df
        return df

    # -- nonlinear Monte-Carlo stochastic Gramian (repo-root implementation) ------
    def _mc_callables(self, sidx):
        """ Wrap the system as the discrete-time x_{k+1} = f_k(x_k) + w_k,
        y_k = h_k(x_k) + v_k expected by gramian_recursive_nonlinear:
        f_k = RK4 step with the recorded input u_{k0+k} baked in; Jacobians by
        central finite differences (same stencil as the EKF). """
        s = self.spec
        sim = self._rk4_sim()
        n = len(s.state_names)

        def jac(func, x, eps=1e-6):
            cols = []
            for i in range(n):
                dx = np.zeros(n); dx[i] = eps * max(abs(x[i]), 1.0)
                cols.append((np.asarray(func(x + dx), dtype=float)
                             - np.asarray(func(x - dx), dtype=float))
                            / (2 * dx[i]))
            return np.array(cols).T

        def make(k0):
            f_k = lambda x, k: sim._rk4_step(np.asarray(x, float),
                                             self.U[k0 + k])
            Fj = lambda x, k: jac(lambda z: sim._rk4_step(z, self.U[k0 + k]),
                                  np.asarray(x, float))
            h_k = lambda x, k: np.asarray(s.h(np.asarray(x, float),
                                              self.U[k0 + k]),
                                          dtype=float)[sidx]
            Hj = lambda x, k: jac(lambda z: np.asarray(
                s.h(z, self.U[k0 + k]), dtype=float)[sidx],
                np.asarray(x, float))
            return f_k, Fj, h_k, Hj
        return make

    def _mc_blocks(self, k0, W, Qds, R, sidx, n_mc, rng):
        """ Monte-Carlo expected blocks over [k0, k0+W), conditioned on the
        nominal x0 = X[k0]: M_k = E[HᵀR⁻¹H] (W entries), G_k = E[F] and
        Dev_k = E[(F−G)ᵀQ⁻¹(F−G)] (W−1 entries each). Same expectations as
        gramian_recursive_nonlinear(mode='mc'), accumulated in a form that
        avoids the huge-minus-huge cancellation of M + S − GᵀQ⁻¹(·)Q⁻¹G when
        any Qd entry is tiny (jittered constant states): algebraically
        S − GᵀQ⁻¹(I+Q⁻¹)⁻¹Q⁻¹G = Dev + Gᵀ[Q⁻¹ − Q⁻¹(I+Q⁻¹)⁻¹Q⁻¹]G, and Dev
        is small×Q⁻¹×small (exactly zero in constant-state directions). """
        n = len(self.spec.state_names)
        f_k, Fj, h_k, Hj = self._mc_callables(sidx)(k0)
        Qsl = Qds[k0:k0 + W - 1]
        Qinvs = [np.linalg.inv(Q) for Q in Qsl]
        chols = [np.linalg.cholesky(Q) for Q in Qsl]
        Rinv = np.linalg.inv(R)
        Ms = [np.zeros((n, n)) for _ in range(W)]
        F_samp = [[] for _ in range(W - 1)]
        for _ in range(n_mc):
            x = self.X[k0].copy()
            for k in range(W):
                H = Hj(x, k)
                Ms[k] += H.T @ Rinv @ H
                if k < W - 1:
                    F_samp[k].append(Fj(x, k))
                    x = f_k(x, k) + chols[k] @ rng.standard_normal(n)
        Ms = [0.5 * (M + M.T) / n_mc for M in Ms]
        Gs, Devs = [], []
        for k in range(W - 1):
            Fs = np.array(F_samp[k])
            G = Fs.mean(axis=0)
            D = Fs - G
            Dev = sum(d.T @ Qinvs[k] @ d for d in D) / n_mc
            Gs.append(G)
            Devs.append(0.5 * (Dev + Dev.T))
        return Ms, Gs, Devs, Qinvs

    @staticmethod
    def _mc_recursion(Ms, Gs, Devs, Qinvs, w):
        """ Backward recursion of the nonlinear stochastic Gramian on the
        first w measurement blocks (stable bracket form). """
        I = Ms[w - 1].copy()
        for k in range(w - 2, -1, -1):
            Qi = Qinvs[k]
            B = Qi - Qi @ np.linalg.solve(I + Qi, Qi)
            I = Ms[k] + Devs[k] + Gs[k].T @ B @ Gs[k]
            I = 0.5 * (I + I.T)
        return I

    def mc_ev(self, w, q_diag, r_diag, lam, sensors=None, states=None,
              q_noise='uncorrelated',
              n_mc=100, seed=0):
        """ Sliding min EV via the nonlinear Monte-Carlo stochastic Gramian
        (the repo-root gramian_recursive_nonlinear math, mode='mc', in the
        stiff-Q-stable form — see _mc_blocks). Conditioned on the nominal
        window-initial state; discrete Q_k is the same Van-Loan Qd the
        linearized pipeline uses, so the methods share the noise model. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        r_key = tuple(f'{r_diag[m]:.3e}' for m in sensors)
        q_key = (q_noise,) + tuple(f'{q_diag[k]:.3e}'
                                  for k in self.spec.state_names)
        key = (self._version, w, q_key, r_key, f'{lam:.3e}', tuple(sensors),
               tuple(states), int(n_mc), int(seed))
        if key in self._mc_cache:
            return self._mc_cache[key], True
        _, _, Qds = self._lin(q_diag, q_noise)
        R = np.diag([r_diag[m] for m in sensors])
        starts = np.arange(0, self.N - w + 1)
        EV = []
        for k0 in starts:
            Ms, Gs, Devs, Qinvs = self._mc_blocks(
                k0, w, Qds, R, sidx, n_mc,
                np.random.default_rng(100003 * seed + k0))
            F = self._mc_recursion(Ms, Gs, Devs, Qinvs, w)
            EV.append(self._ev_diag(F, lam)[sub])   # marginal; see linearized_ev
        df = pd.DataFrame(np.array(EV), columns=states)
        df.insert(0, 'time_initial', self.t[starts])
        df.insert(1, 'time', self.t[starts] + int(round(0.5 * w)) * self.dt)
        self._mc_cache[key] = df
        return df, False

    def mc_ev_vs_window(self, k0, ws, q_diag, r_diag, lam, sensors=None,
                        q_noise='uncorrelated',
                        states=None, n_mc=100, seed=0):
        """ Min EV vs window length via the nonlinear MC Gramian. The MC
        M/S/G blocks along [k0, k0+max(ws)) are estimated once (block k's
        distribution does not depend on the window horizon), then the
        backward recursion runs on each prefix — same math as
        gramian_recursive_nonlinear, factored for shared sampling. """
        sensors, states, sidx, sub = self._fim_subsets(sensors, states)
        W_max = int(max(ws))
        r_key = tuple(f'{r_diag[m]:.3e}' for m in sensors)
        q_key = (q_noise,) + tuple(f'{q_diag[k]:.3e}'
                                  for k in self.spec.state_names)
        key = (self._version, k0, W_max, q_key, r_key, f'{lam:.3e}',
               tuple(sensors), tuple(states), int(n_mc), int(seed))
        if key in self._mc_vs_w_cache:
            return self._mc_vs_w_cache[key]
        _, _, Qds = self._lin(q_diag, q_noise)
        R = np.diag([r_diag[m] for m in sensors])
        Ms, Gs, Devs, Qinvs = self._mc_blocks(
            k0, W_max, Qds, R, sidx, n_mc,
            np.random.default_rng(200003 * seed + k0))
        EV = []
        for wv in ws:   # backward recursion on each prefix of the blocks
            F = self._mc_recursion(Ms, Gs, Devs, Qinvs, wv)
            EV.append(self._ev_diag(F, lam)[sub])   # marginal; see linearized_ev
        df = pd.DataFrame(EV, columns=states, index=np.array(ws) * self.dt)
        self._mc_vs_w_cache[key] = df
        return df

    # -- estimators: true vs estimated states ------------------------------------
    def _estimation_problem(self, r_diag, seed, x0_scale=1.0, x0_offset=None,
                            u_noise_var=0.0, u_bias=None, p0_sigma=None,
                            process_noise=None):
        """ Sample one estimation-problem realization: noisy measurements
        y_k = h(x_k, u_k) + v_k with v ~ N(0, R) from the true trajectory, plus
        an initial estimate with ~10%% error scaled by x0_scale, and optional
        per-state offsets (e.g. {'z': 7.5} for the paper's z0 = 9 experiment).
        P0 stays consistent with the total initial spread.

        x0_scale may be a scalar (every state) or a {state: scale} dict, states
        absent from it keeping 1.0. Use the dict form to make the guess bad only
        on SOME states — e.g. everything the sensors do not report directly, to
        ask whether a badly-initialized model wrecks the states that ARE
        observable. The draw and P0 scale together, so the filter is told how
        wrong it is; the realization stays keyed on the same seed, but note that
        changing any scale changes the DRAW, so a run with a different x0_scale is
        not the same realization as one without it.

        p0_sigma = {state: sigma} overrides sigma_0 for those states AFTER the
        offset bookkeeping, so P0 can be set independently of the offset. Use it
        for a DIFFUSE prior — e.g. {'zeta': pi/sqrt(3)}, the variance of a
        uniform distribution on the circle, i.e. "the wind direction is unknown"
        — where the honest-P0 rule above would instead encode the offset as a
        merely-somewhat-uncertain belief. Only P0 changes; the sampled xh0 does
        not, so the guess itself is unaffected.

        u_noise_var / u_bias model *measured* inputs (paper Fig. 4b): the truth
        flies the clean U, but the filters receive U + N(0, u_noise_var)
        (+ a constant bias on one channel inside a time window,
        u_bias = (channel_index, magnitude, t0, t1)). Same seed -> same
        realization, so the EKF and UKF face identical data.

        `process_noise` = {state: variance} makes the TRUTH stochastic:
        x_{k+1} = f_d(x_k, u_k) + w_k with w ~ N(0, Q_d), starting from the
        nominal x_0, and the measurements come from that wandering truth
        instead of self.X. The realized truth is stashed as `self.truth_last`
        (self.X when process_noise is None), because with a stochastic truth
        the estimation error is x_hat - truth_last, NOT x_hat - self.X.

        DEFAULT None keeps the historical behaviour — a deterministic truth. Note
        that a filter carrying Q_d > 0 against a deterministic truth is
        CONSERVATIVE by construction: it budgets for wander that never happens,
        so its realized RMSE lands below both its own sqrt(P) and the Cramer-Rao
        bound for the same Q. Pass process_noise = the filter's own q_diag to
        make truth and model agree, which is what any RMSE-vs-CRB comparison
        needs. """
        s = self.spec
        rng = np.random.default_rng(seed + 1000 * self._version)
        R = np.diag([r_diag[m] for m in s.measurement_names])
        X_true = self.X
        if process_noise is not None:
            sim_t = self._rk4_sim()
            q_sig = np.sqrt(np.array([process_noise[nm] for nm in s.state_names]))
            ang_t = [i for i, nm in enumerate(s.state_names)
                     if nm in getattr(s, 'angle_states', ())]
            X_true = np.zeros_like(self.X)
            X_true[0] = self.X[0]
            for k in range(self.N - 1):
                xk = sim_t._rk4_step(X_true[k], self.U[k]) + rng.normal(0.0, q_sig)
                if ang_t:
                    xk[ang_t] = (xk[ang_t] + np.pi) % (2 * np.pi) - np.pi
                X_true[k + 1] = xk
        self.truth_last = X_true
        Y = np.array([np.asarray(s.h(X_true[k], self.U[k]), dtype=float)
                      for k in range(self.N)])
        Y += rng.normal(0.0, np.sqrt(np.diag(R)), size=Y.shape)
        sc = (np.array([x0_scale.get(nm, 1.0) for nm in s.state_names], dtype=float)
              if isinstance(x0_scale, dict) else float(x0_scale))
        sig0 = sc * 0.1 * np.maximum(np.abs(self.X[0]), 0.1)
        xh0 = self.X[0] + rng.normal(0.0, sig0)
        if x0_offset:
            for name, off in x0_offset.items():
                i = s.state_names.index(name)
                xh0[i] += off
                sig0[i] = np.hypot(sig0[i], off)   # keep P0 honest
        if p0_sigma:
            for name, sg in p0_sigma.items():
                sig0[s.state_names.index(name)] = float(sg)
        U_used = self.U.copy()
        if u_noise_var > 0:
            U_used = U_used + rng.normal(0.0, np.sqrt(u_noise_var),
                                         size=U_used.shape)
        if u_bias is not None:
            ch, mag, t0, t1 = u_bias
            U_used[(self.t >= t0) & (self.t < t1), ch] += mag
        return Y, xh0, np.diag(sig0 ** 2), R, U_used

    def _sensor_subset(self, sensors, Y, R):
        """ Restrict an estimation problem to a subset of measurement channels, so
        the filters can be driven by exactly the sensors the observability analysis
        uses (spec.fim_sensors) rather than all of h(). Returns
        (h_sel, Y_sel, R_sel, ang_idx) where h_sel(x, u) yields only those channels
        and ang_idx indexes the angular ones WITHIN the subset. 
        """
        s = self.spec
        names = list(s.measurement_names)
        sensors = list(sensors) if sensors else names
        sidx = np.array([names.index(m) for m in sensors], dtype=int)
        ang = [j for j, m in enumerate(sensors) if m in s.angle_measurements]
        h_sel = lambda x, u: np.asarray(s.h(x, u), dtype=float)[sidx]

        return h_sel, Y[:, sidx], R[np.ix_(sidx, sidx)], ang

    def run_ekf(self, q_diag, r_diag, seed=0, full_cov=False,
                x0_scale=1.0, x0_offset=None, u_noise_var=0.0, u_bias=None,
                sensors=None, f_jac_at='pre', fd_eps_scaled=False,
                return_rho=False, q_noise='uncorrelated', p0_sigma=None,
                start_at='prior', process_noise=None):
        """ EKF along the true trajectory (finite-difference Jacobians of the RK4
        step and of h). Returns (X_hat, P_diag), post-update values; with
        full_cov=True also returns [P_k] for consistency tests (NEES); with
        return_rho=True also returns the per-step spectral radius of the error
        dynamics (see below).

        ORDER (`start_at`): with the default 'prior' the supplied initial estimate is
        the PRIOR at k = 0, so the cycle is entered at the correction step and y_0 is
        consumed. Equivalently x0 = x_hat_{0|-1}. This keeps the filter aligned with the
        constructability Gramian, whose recursion starts the same way
        (F_0 = C_0^T R^-1 C_0), which is what makes the min-EV curve and this filter's P
        directly comparable. The BOUNDS reference driver instead predicts first and
        drops y_0.

        `start_at='posterior'` is the other standard convention (Kalman's original
        indexing; Bar-Shalom, Gelb, Simon): the initial estimate IS x_hat_{0|0}, the
        k = 0 correction is skipped and y_0 is never consumed, so X_hat[0] is the
        supplied guess verbatim rather than a value one update old. Identical from
        k = 1 on, given the same information; it exists here so a figure can show the
        guess itself at t = 0 instead of silently starting one update in. NOTE the
        Gramian alignment above does NOT hold under this convention.

        Cross-checked against the BOUNDS reference implementation,
        ``Discovering-and-exploiting-active-sensing-motifs-for-estimation/util/
        extended_kalman_filter.py::EKF``. Verified IDENTICAL there: the RK4 step
        x + dt/6·(k1+2k2+2k3+k4); the Joseph-form covariance update
        (I−KH)P(I−KH)ᵀ + KRKᵀ; central-difference numerical Jacobians; wrapping the
        innovation on circular measurement channels; and the gain P Hᵀ S⁻¹ (computed
        here with a linear solve rather than an explicit inverse — same value, better
        conditioned).

        Two conventions are exposed because the reference makes a different choice:

        `f_jac_at` — WHERE F is evaluated. DEFAULT 'pre' since 2026-08-24.

            'pre' linearizes about x̂_{k|k}, which is what the EKF derivation
            requires: the prediction expands f about the current estimate, so
            F = ∂f/∂x at THAT point, and P_{k+1|k} = F P_{k|k} Fᵀ + Q is the
            covariance of exactly that linearization. Bar-Shalom, Simon, Gelb,
            Särkkä and Anderson & Moore all write it this way.

            'post' evaluates ∂f/∂x at f(x̂_{k|k}) instead. Since f here is the
            discrete RK4 map, that is the Jacobian of the step carrying
            x̂_{k+1|k} to x̂_{k+2|k+1} — the linearization of the NEXT interval,
            not the one whose covariance is being propagated.

            The BOUNDS reference does 'post', apparently as an ordering artifact:
            its predict() assigns self.x = f(self.x, u) and only then calls
            F_jacobian(self.x, u), so swapping those two lines would give 'pre'.
            Pass f_jac_at='post' EXPLICITLY when reproducing that reference —
            ekf_reference_compare.py and ekf_driver_asymmetry.py do.

            The two agree as dt → 0 and differ by O(dt·ẋ) in F: measured on the
            fly model, 2.9e-3 relative at rest but 0.44 through a −π/2 heading
            step at t = 0.2 s. On that trajectory 'pre' also estimated ζ better
            (RMS 0.036 vs 0.051), and the choice alone moved ζ̂ by 4.5e-2 —
            against 9.1e-11 from finite-difference vs autodiff Jacobians.
        `fd_eps_scaled` — the reference perturbs by an ABSOLUTE 1e-6. This model has a
            state of magnitude 5.2e5 (the inertia I), for which an absolute 1e-6 is a
            relative step of 2e-12 — far below the useful differencing range. Measured
            against a refined Jacobian, the absolute convention is off by 7.6e-6 while
            the scaled one (1e-6·max(|xᵢ|,1)) is off by 6.8e-8, i.e. ~110× better.
            Default False (absolute) to MATCH the reference; set True for the more
            accurate scaled step. Measured end-to-end on the ζ estimate the two agree
            to 5 decimals, so this is latent robustness rather than a live difference.

        NOTE on Q: `q_diag` is the PER-STEP covariance Q_d, used verbatim — the same
        convention as the reference and as the Gramian pipeline.

        `process_noise` (default None) makes the TRUTH stochastic with that Q_d —
        see _estimation_problem. Compare X_hat against `self.truth_last`, not
        self.X, whenever it is used.
        """
        s = self.spec
        n = len(s.state_names)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var,
                                                  u_bias, p0_sigma,
                                                  process_noise)
        # q_diag IS Q_d. With q_noise='vanloan' the per-step covariance is
        # rebuilt from the LOCAL dynamics, so it varies along the trajectory
        # and is no longer diagonal — use the same stack the Gramians see.
        # 'uncorrelated' keeps the filters self-contained (no trajectory
        # linearization, as before); 'vanloan' needs A_k, so it reuses the same
        # cached stack the stochastic gramian sees.
        Qds = (self._lin(q_diag, q_noise)[2] if q_noise == 'vanloan' else
               [np.diag([q_diag[nm] for nm in s.state_names])] * self.N)
        h_sel, Y, R, ang_m = self._sensor_subset(sensors, Y, R)
        # ANGULAR STATES ARE KEPT CANONICAL. zeta is a wind direction and phi a
        # heading: only (-pi, pi] are legal values, so they are re-wrapped after
        # each update, exactly as run_ukf does. Without this the EKF's state
        # drifts outside the range it is defined on (measured: zeta_hat reaching
        # -3.196 against a truth of -pi) and anything downstream -- a plot, a
        # table, a comparison against another estimator -- reports an angle that
        # cannot exist. It is numerically NEUTRAL: f and h depend on these states
        # only through cos/sin, so every quantity in the recursion is periodic in
        # them (verified to 1e-15).
        ang_s = [i for i, nm in enumerate(s.state_names)
                 if nm in getattr(s, 'angle_states', ())]
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

        def jac(func, x, u, eps=1e-6):
            cols = []
            for i in range(n):
                step = eps * max(abs(x[i]), 1.0) if fd_eps_scaled else eps
                dx = np.zeros(n); dx[i] = step
                fp = np.asarray(func(x + dx, u), dtype=float)
                fm = np.asarray(func(x - dx, u), dtype=float)
                cols.append((fp - fm) / (2 * step))
            return np.array(cols).T

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        P_hist = []
        rho = []
        Phi_prev = None
        I_n = np.eye(n)
        for k in range(self.N):
            # measurement update — skipped at k = 0 under the 'posterior' convention,
            # where the supplied estimate already IS x_hat_{0|0} and y_0 is not ours
            IKC = I_n
            if not (k == 0 and start_at == 'posterior'):
                C = jac(h_sel, xh, U[k])
                innov = Y[k] - h_sel(xh, U[k])
                if ang_m:
                    innov[ang_m] = wrap(innov[ang_m])
                S = C @ P @ C.T + R
                K = np.linalg.solve(S.T, (P @ C.T).T).T
                xh = xh + K @ innov
                # angular STATES are kept canonical, as run_ukf does. This must
                # sit INSIDE the correction branch: on the k = 0 posterior start
                # there is no C or K, so anything below that references them
                # would raise (it did -- UnboundLocalError on start_at
                # 'posterior', which is every no-prior and bad-guess pass).
                if ang_s:
                    xh[ang_s] = wrap(xh[ang_s])
                IKC = I_n - K @ C
                P = IKC @ P @ IKC.T + K @ R @ K.T   # Joseph form
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            if full_cov:
                P_hist.append(P.copy())
            if return_rho:
                # spectral radius of the error dynamics (I − K H) F, as the reference
                # logs it: |eig| < 1 means the filter is contracting
                Fprev = Phi_prev if Phi_prev is not None else np.eye(n)
                rho.append(float(np.max(np.abs(
                    np.linalg.eigvals(IKC @ Fprev)))))
            # time update
            if k < self.N - 1:
                xnew = sim._rk4_step(xh, U[k])
                # 'pre' (default) is the derivation-correct point — F at the state
                # the prediction was expanded about; 'post' reproduces the BOUNDS
                # reference's ordering artifact (see the docstring)
                Phi = jac(sim._rk4_step, xnew if f_jac_at == 'post' else xh, U[k])
                xh = xnew
                P = Phi @ P @ Phi.T + Qds[k]
                Phi_prev = Phi
        out = [X_hat, P_diag]
        if full_cov:
            out.append(P_hist)
        if return_rho:
            out.append(np.array(rho))
        return tuple(out) if len(out) > 2 else (X_hat, P_diag)

    def run_ukf(self, q_diag, r_diag, seed=0, alpha=1.0, beta=2.0, kappa=0.0,
                full_cov=False, x0_scale=1.0, x0_offset=None, u_noise_var=0.0,
                u_bias=None, sensors=None, wrap_states=True,
                q_noise='uncorrelated', p0_sigma=None,
                start_at='prior', process_noise=None):
        """ UKF on the identical problem realization as run_ekf (same seed → same
        measurements and initial estimate). Merwe SCALED unscented transform:

            lam = alpha^2 (n + kappa) - n,   sigma points at sqrt(n+lam)*sigma
                                                          = alpha*sqrt(n+kappa)*sigma

        alpha=1, kappa=0 puts them at sqrt(n)*sigma, which for this 18-state model
        is 4.24 sigma — far enough out that the arctan2 measurements get evaluated at
        meaningless states as soon as an angular state's sigma approaches a radian.
        Shrink alpha (1e-3 is the MATLAB / StateEstimator.m default) or use Julier's
        kappa = 3 - n to keep the transform local.


        `p0_sigma`, `start_at` and `process_noise` are exactly as in run_ekf — same
        meaning, same realization, so the two filters stay comparable under either
        convention. With `process_noise` set, compare X_hat against
        `self.truth_last` rather than self.X.

        `wrap_states` gives spec.angle_states a CIRCULAR mean and wrapped
        residuals inside the unscented transform, and re-wraps them after each
        update. DEFAULT True — this is the standard treatment (Kraft 2003;
        Crassidis & Markley 2003; FilterPy exposes x_mean_fn / residual_x for
        exactly this), not an optional extra.

        It is needed because h() wraps the angular states implicitly through
        atan2: the measurement deviations dY come back wrapped while the state
        deviations dX would not, so the cross-covariance mixes the two and the
        gain is wrong. That only shows up once the sigma points span a
        meaningful fraction of a circle. Measured on the fly at Q_zeta = 1e-3,
        turn rate 10 rad/s, varying only the initial zeta error:

            sigma0(zeta)  sigma spread  EKF rms   UKF rms (wrap_states off)
               0.401          1.70       0.030      0.041
               0.591          2.51       0.031      0.080
               0.813          3.45       0.032      0.216
               1.048          4.45       0.035      2.008

        i.e. it degrades as the spread approaches pi and fails outright past it,
        while the EKF -- which never spreads points -- is untouched. """
        s = self.spec
        n = len(s.state_names)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var,
                                                  u_bias, p0_sigma,
                                                  process_noise)
        # q_diag IS Q_d. With q_noise='vanloan' the per-step covariance is
        # rebuilt from the LOCAL dynamics, so it varies along the trajectory
        # and is no longer diagonal — use the same stack the Gramians see.
        # 'uncorrelated' keeps the filters self-contained (no trajectory
        # linearization, as before); 'vanloan' needs A_k, so it reuses the same
        # cached stack the stochastic gramian sees.
        Qds = (self._lin(q_diag, q_noise)[2] if q_noise == 'vanloan' else
               [np.diag([q_diag[nm] for nm in s.state_names])] * self.N)
        h_sel, Y, R, ang_m = self._sensor_subset(sensors, Y, R)
        ang_s = ([i for i, nm in enumerate(s.state_names)
                  if nm in getattr(s, 'angle_states', ())] if wrap_states
                 else [])
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

        lam_s = alpha ** 2 * (n + kappa) - n      # Merwe scaling parameter
        scale = max(n + lam_s, 1e-9)              # = alpha^2 (n+kappa)
        Wm = np.full(2 * n + 1, 1.0 / (2 * scale)); Wm[0] = lam_s / scale
        Wc = Wm.copy(); Wc[0] = lam_s / scale + (1.0 - alpha ** 2 + beta)

        def sqrt_psd(M):
            try:
                return np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                e, V = np.linalg.eigh((M + M.T) / 2)
                return V @ np.diag(np.sqrt(np.clip(e, 1e-15, None)))

        def psd(M):
            """ Symmetrize and project onto the PSD cone.

            Not a tuning knob: the UKF measurement update below is
            P - K S K^T, the NON-Joseph form, and with wrapped angular
            residuals the S it subtracts is no longer exactly consistent with P,
            so the result can leave the PSD cone. The next step then asks for a
            Cholesky of it. Projecting back is the numerical companion to the
            circular treatment -- the two were introduced together and removing
            either alone makes the fly diverge on the app's own default
            settings. The principled alternative is Van der Merwe's square-root
            UKF, which keeps the factor PSD by construction. """
            M = (M + M.T) / 2
            if not np.all(np.isfinite(M)):
                raise np.linalg.LinAlgError('filter diverged: covariance is '
                                            'not finite')
            e, V = np.linalg.eigh(M)
            return (V * np.clip(e, 0.0, None)) @ V.T

        def sigma_points(x, P):
            L = sqrt_psd(scale * P)
            pts = np.empty((2 * n + 1, n))
            pts[0] = x
            pts[1:n + 1] = x + L.T
            pts[n + 1:] = x - L.T
            return pts

        def circ_mean(A, W, idx):
            """ Weighted mean of rows of A, circular on the columns in `idx`.
            Used for the angular MEASUREMENT channels: the arithmetic mean of
            angles straddling ±π is meaningless, so those columns take
            atan2(Σ W sin, Σ W cos) instead. Angular STATES need nothing —
            nothing here wraps them, so they stay on a continuous branch
            where the plain arithmetic mean is already exact. """
            m = A.T @ W
            for i in idx:
                m[i] = np.arctan2(np.sum(W * np.sin(A[:, i])),
                                  np.sum(W * np.cos(A[:, i])))
            return m

        def resid(A, m, idx):
            D = A - m
            if idx:
                D[:, idx] = wrap(D[:, idx])
            return D

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        P_hist = []
        for k in range(self.N):
            # measurement update — see run_ekf on `start_at`
            if not (k == 0 and start_at == 'posterior'):
                pts = sigma_points(xh, P)
                Ys = np.array([h_sel(pt, U[k]) for pt in pts])
                ym = circ_mean(Ys, Wm, ang_m)
                dY = resid(Ys, ym, ang_m)
                dX = resid(pts, xh, ang_s)
                S_cov = dY.T @ (Wc[:, None] * dY) + R
                Cxy = dX.T @ (Wc[:, None] * dY)
                try:
                    K = np.linalg.solve(S_cov.T, Cxy.T).T
                except np.linalg.LinAlgError:      # singular innovation covariance
                    K = (np.linalg.pinv(S_cov.T) @ Cxy.T).T
                innov = Y[k] - ym
                if ang_m:
                    innov[ang_m] = wrap(innov[ang_m])
                xh = xh + K @ innov
                P = psd(P - K @ S_cov @ K.T)
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            if full_cov:
                P_hist.append(P.copy())
            # time update
            if k < self.N - 1:
                pts = sigma_points(xh, P)
                Xs = np.array([sim._rk4_step(pt, U[k]) for pt in pts])
                xh = circ_mean(Xs, Wm, ang_s)
                dXp = resid(Xs, xh, ang_s)
                P = psd(dXp.T @ (Wc[:, None] * dXp) + Qds[k])
        if full_cov:
            return X_hat, P_diag, P_hist
        return X_hat, P_diag

    def attach_ann(self, ann, seed=0, n_traj=16, epochs=100, batch=256,
                   noise_std=0.0):
        """ Register an already-built ANN estimator under the exact cache key
        ``run_aikf`` looks up, and return the run_aikf kwargs that identify it.
        This engine's run_aikf does not train (it raises if the net is missing),
        so attaching is mandatory here; the method exists so no caller has to
        hand-build the key tuple:

            kw = engine.attach_ann(net, seed=SEED)
            engine.run_aikf(net.target, q, r, seed=SEED, **kw, ...)

        `ann` must expose predict_series(Y, U) plus .target, .time_steps,
        .layers. n_traj/epochs/batch/noise_std only KEY the cache; returning them
        keeps both sides in step. Attach under the same seed run_aikf will use. """
        key = (self.spec.name, ann.target, int(ann.time_steps),
               tuple(ann.layers), int(n_traj), int(epochs), int(batch),
               round(float(noise_std), 6), int(seed))
        self._ann_cache[key] = ann
        return dict(time_steps=int(ann.time_steps), layers=tuple(ann.layers),
                    n_traj=int(n_traj), epochs=int(epochs), batch=int(batch),
                    noise_std=float(noise_std))

    def run_aikf(self, target, q_diag, r_diag, seed=0, time_steps=4,
                 layers=(64, 64, 64), n_traj=16, epochs=100, batch=256,
                 noise_std=0.01, motif_input=None, motif_window=20,
                 motif_upper=0.5, R_ann_lo=1e-3, R_ann_hi=1e12,
                 alpha=1e-3, beta=1.0, kappa=0.0,
                 x0_scale=1.0, x0_offset=None, u_noise_var=0.0, u_bias=None,
                 p0_sigma=None, sensors=None, start_at='prior'):
        """ The repo's AI-KF (util/StateEstimator.m): a UKF on the real dynamics
        --- PORTED from ../observability-gui/engine/observability_gui.py.

        The two engines are NOT interchangeable, which is why this is a port and
        not an import: that one computes the MARGINAL empirical FIM (all states
        inverted, then columns selected) while this one is CONDITIONAL (columns
        of O cropped), and its _estimation_problem takes p0_diag/x0_guess where
        this one takes p0_sigma. Swapping engines would change every min-EV
        number this module reports. Only the ANN training path is dropped ---
        whose measurement vector is AUGMENTED with the ANN's estimate of `target`
        as a pseudo-measurement, and whose noise entry for that channel is
        modulated by an active-sensing motif.

            R[ann,ann]_k = clip(logarithmic_map(
                backward_window_mean(motif, motif_window),
                [0, motif_upper], (R_ann_hi, R_ann_lo)))

        No motif -> R = R_ann_hi (1e12: the channel is ignored and the filter
        coasts on its dynamics); motif at/above `motif_upper` -> R = R_ann_lo
        (1e-3: the ANN is trusted). Set R_ann_lo = R_ann_hi to disable the channel
        and recover the plain EKF — the repo's own ablation (r_sweep = [12, -3]).

        `motif_input` names the input channel used as the motif (repo:
        motif = |u_x|); it defaults to the last input, which is the horizontal
        acceleration for the altitude/monocular models.

        Returns (X_hat, P_diag, raw, R_ann): filter output over all states, the
        raw ANN estimate, and the per-step noise assigned to the ANN channel. """
        s = self.spec
        n = len(s.state_names)
        j_t = s.state_names.index(target)
        sim = self._rk4_sim()
        Y, xh, P, R, U = self._estimation_problem(r_diag, seed, x0_scale,
                                                  x0_offset, u_noise_var, u_bias,
                                                  p0_sigma)
        # This engine does not TRAIN networks: the ANN must have been put in the
        # cache already (see Qanalysis/zeta_ann_estimator.attach). Training lives
        # in train_zeta_ann.py, offline, so a sweep never silently fits a net.
        key = (self.spec.name, target, time_steps, tuple(layers), n_traj,
               epochs, batch, round(float(noise_std), 6), seed)
        if key not in self._ann_cache:
            raise KeyError(
                'no ANN attached for %r: call zeta_ann_estimator.attach(engine, '
                'spec) first. Cached keys: %s' % (key, list(self._ann_cache)))
        ann = self._ann_cache[key]
        raw = ann.predict_series(Y, U)
        # the motif: mean |input| over a backward window (repo: |u_x|)
        mi = (s.input_names.index(motif_input)
              if motif_input in s.input_names else len(s.input_names) - 1)
        R_ann = motif_R(np.abs(U[:, mi]), motif_window, motif_upper,
                        (float(R_ann_hi), float(R_ann_lo)))
        # augmented measurement: [real sensors..., ANN estimate of `target`].
        # `sensors` restricts the REAL channels exactly as in run_ekf/run_ukf, so
        # the AI-KF can be driven by the same subset the observability analysis
        # uses; the ANN pseudo-measurement is appended after them.
        h_sel, Y, R, ang_m = self._sensor_subset(sensors, Y, R)
        p_real = Y.shape[1]
        circ = target in s.angle_states
        # angular STATES need the same circular treatment run_ukf gives
        # them — this is a UKF too, and h() wraps them through atan2
        ang_s = [i for i, nm in enumerate(s.state_names)
                 if nm in getattr(s, 'angle_states', ())]
        ang_aug = ang_m + ([p_real] if circ else [])   # angular cols of the
        wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi   # augmented vector
        Qd = np.diag([q_diag[k] for k in s.state_names])

        # ── sigma-point machinery, identical to run_ukf ──────────────────────
        lam_s = alpha ** 2 * (n + kappa) - n     # Merwe scaling parameter
        scale = max(n + lam_s, 1e-6)             # = α²(n+κ); sigma spread √scale
        Wm = np.full(2 * n + 1, 1.0 / (2 * scale)); Wm[0] = lam_s / scale
        Wc = Wm.copy(); Wc[0] = lam_s / scale + (1.0 - alpha ** 2 + beta)

        def sqrt_psd(M):
            try:
                return np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                e, V = np.linalg.eigh((M + M.T) / 2)
                return V @ np.diag(np.sqrt(np.clip(e, 1e-15, None)))

        def sigma_points(x, P):
            L = sqrt_psd(scale * P)
            pts = np.empty((2 * n + 1, n))
            pts[0] = x
            pts[1:n + 1] = x + L.T
            pts[n + 1:] = x - L.T
            return pts

        def psd_local(M):
            """ As run_ukf's psd(): the wrapped-residual update can leave the
            PSD cone, and the next Cholesky would fail. """
            M = (M + M.T) / 2
            if not np.all(np.isfinite(M)):
                raise np.linalg.LinAlgError('AI-KF diverged: covariance is '
                                            'not finite')
            e, V = np.linalg.eigh(M)
            return (V * np.clip(e, 0.0, None)) @ V.T

        def circ_mean(A, W, idx):
            m = A.T @ W
            for i in idx:
                m[i] = np.arctan2(np.sum(W * np.sin(A[:, i])),
                                  np.sum(W * np.cos(A[:, i])))
            return m

        def resid(A, m, idx):
            D = A - m
            if idx:
                D[:, idx] = wrap(D[:, idx])
            return D

        def y_stats(Ys, cols):
            ym = Ys.T @ Wm
            for i in cols:               # circular mean for angular channels
                ym[i] = np.arctan2(np.sum(Wm * np.sin(Ys[:, i])),
                                   np.sum(Wm * np.cos(Ys[:, i])))
            dY = Ys - ym
            if cols:
                dY[:, cols] = wrap(dY[:, cols])
            return ym, dY

        X_hat = np.zeros_like(self.X)
        P_diag = np.zeros((self.N, n))
        for k in range(self.N):
            # measurement update — see run_ekf on `start_at`
            if not (k == 0 and start_at == 'posterior'):
                # ── measurement update over the augmented vector ────────────────
                use_ann = bool(np.isfinite(raw[k]))
                pts = sigma_points(xh, P)
                Ys = np.array([h_sel(pt, U[k]) for pt in pts])
                if use_ann:
                    # the ANN channel observes the target state directly, so its
                    # sigma-point image is just that component
                    Ys = np.hstack([Ys, pts[:, [j_t]]])
                    Rk = np.zeros((p_real + 1, p_real + 1))
                    Rk[:p_real, :p_real] = R
                    Rk[-1, -1] = R_ann[k]
                    zk = np.append(Y[k], raw[k])
                    cols = ang_aug
                else:
                    Rk, zk, cols = R, Y[k], ang_m
                ym, dY = y_stats(Ys, cols)
                dX = resid(pts, xh, ang_s)
                S_cov = dY.T @ (Wc[:, None] * dY) + Rk
                Cxy = dX.T @ (Wc[:, None] * dY)
                K = np.linalg.solve(S_cov.T, Cxy.T).T
                innov = zk - ym
                if cols:
                    innov[cols] = wrap(innov[cols])
                xh = xh + K @ innov
                if ang_s:
                    xh[ang_s] = wrap(xh[ang_s])
                P = psd_local(P - K @ S_cov @ K.T)
            X_hat[k] = xh
            P_diag[k] = np.diag(P)
            # ── time update ─────────────────────────────────────────────────
            if k < self.N - 1:
                pts = sigma_points(xh, P)
                Xs = np.array([sim._rk4_step(pt, U[k]) for pt in pts])
                xh = circ_mean(Xs, Wm, ang_s)
                dXp = resid(Xs, xh, ang_s)
                P = psd_local(dXp.T @ (Wc[:, None] * dXp) + Qd)
        return X_hat, P_diag, raw, R_ann


# ─────────────────────────────────── GUI ───────────────────────────────────
