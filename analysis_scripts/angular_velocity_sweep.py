#!/opt/anaconda3/envs/Python311/bin/python
"""
Parameter sweep: wind-direction (ζ) observability vs TURN SHARPNESS (radial
curvature) and process noise — for a family of trajectories that share the same
start, the same end and the same apex, so that a sharper turn does NOT buy itself a
much longer path (total distance stays within a few percent, ~7% by default).

The question
------------
Is a FAST, SHARP turn or a SLOW, SHALLOW turn better for observing wind direction
ζ, especially under process noise?

The kinematic constraint that shapes this
-----------------------------------------
The MPC tracks speed + heading, which fully determine the ground track, whose
curvature obeys, POINTWISE,

        ω(t) = g(t) · κ(t)                                                (★)

Only two of {turn rate ω, speed g, curvature κ} are ever free. So:
  * holding g constant and sweeping ω FORCES a proportional change of path
    (κ = ω/g) — this script's first design, whose sharp rows travelled 48% farther
    (0.396 → 0.588 m) and bulged 2.3× higher: a spatial confound;
  * holding the path fixed and sweeping g leaves κ CONSTANT — no sharpness
    variation at all (that is ``path_matched_speed_sweep.py``, kept as the
    pure-speed control).
Genuinely different curvature therefore REQUIRES different positions somewhere;
the goal here is to keep that difference small and bounded.

The family
----------
Every row is: straight leg → circular arc of radius ``RADII[i]`` → straight leg,
symmetric about the chord, from a shared A = (0,0) to a shared B, with the arc's
apex pinned at height ``APEX_H`` above the chord midpoint. The legs are the tangent
lines from A and B to that arc, so ONE parameter — the radius — sets the whole
shape, and the net turn angle is an OUTCOME, not an input. At the defaults:

    R       0.145 → 0.015 m     (9.7× radial curvature)
    turn    137° → 84°          (net heading change, an outcome)
    length  0.399 → 0.429 m     (7.3% spread — deliberately NOT pinned exactly)
    max lateral deviation between rows ≈ 15% of the chord

The shallowest row is almost a pure sweeping arc (legs ~0.04 m, arc ~0.39 m); the
sharpest is two nearly-straight legs meeting at a tight corner (legs ~0.20 m, arc
~0.02 m). That is the shallow-vs-sharp contrast, at nearly constant distance.

Speed is deliberately NOT constant along a trajectory — (★) holds pointwise, so
varying it is the only way to make ω range more widely than curvature alone allows.
Each row cruises the straight legs at the COMMON ``LEG_SPEED`` (so every row's
approach and exit are identical) and flies the arc at its own ``TURN_SPEEDS[i]``,
blended over ``BLEND`` of arc length. Hence

        ω_turn = TURN_SPEEDS[i] / RADII[i]

sweeps ~30× (1.0 → 31 rad/s) — sharper AND faster together, the "aggressive
maneuver" axis — while total time falls ~2.9 s → ~1.4 s. NOTE the fly's rotational
dynamics cap the achievable rate near ~30 rad/s, so the sharpest row may not reach
its commanded ω; the achieved peak |φ̇| is printed and stored per row.

Construction is PATH-FIRST: the curve is defined by arc length s (`path_point`
gives position/heading/curvature at s), then marched at the prescribed speed
profile (`build_row`). That guarantees the commanded track is exactly the designed
curve whatever the speed profile does.

The wind-ratio confound, and the two WIND_MODEs
-----------------------------------------------
ζ enters the model ONLY through ``a_para = v_para − w·cos(φ−ζ)``,
``a_perp = v_perp + w·sin(φ−ζ)``, so ζ observability depends on w/g.
Differentiating γ = atan2(a_perp, a_para) at the along-wind heading (the apex,
where v_perp ≈ 0) gives

        ∂γ/∂ζ  =  −w / (g − w)                                           (†)

The turn speed varies row to row, so wind speed does too — for a reason unrelated to
turning. Both modes are run, into separate subdirectories:

  'fixed'   w = WIND_FIXED for every row (one wind, the natural reading). wind speed has a
        POLE where the turn speed equals w: there the apparent airflow VANISHES at
        the apex — the fly drifts exactly with the wind and momentarily has no wind
        cue at all — and rows either side get large ∂γ/∂ζ of OPPOSITE SIGN. Note
        the legs still share w/g across rows, so the confound is confined to the
        turn.
  'scaled'  w = WIND_OVER_SPEED · TURN_SPEEDS[i], holding w/g fixed AT THE TURN
        (where the along-wind crossing, and hence the ζ information, happens).
        Then wind speed is the same constant for every row and the confound is removed
        exactly.

Outputs (``Qanalysis/angular_velocity_sweep_output/wind_<mode>/``)
  * ``family.png`` — the whole family overlaid, with the measured max lateral
    deviation and the path-length spread in the title.
  * one figure per row: path colored by min-EV(ζ) (pybounds Q=0 vs Gramian) +
    min-EV(ζ) vs time (pybounds vs Eq. 33 at every Q) + the ζ(t) state.
  * TWO summary matrices (rows = sharpness, shallow→sharp, WITH a trajectory
    thumbnail, cols = [pybounds Q=0] + Q values):
        - ``matrix_best_min_ev_zeta`` — cell = best (min over trajectory) min-EV;
        - ``matrix_mean_min_ev_zeta`` — cell = mean min-EV over the turn.
    PNG + CSV each; CSVs also carry R, ω_turn, turn angle, apex, path length,
    T_total, w, w/g, ∂γ/∂ζ and the achieved peak |φ̇|.


"""

import os
import io
import contextlib

import numpy as np
import matplotlib
matplotlib.use('Agg')                      # headless: save PNGs, no display
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twe.observability import ObservabilityEngine, SYSTEMS
from twe.paths import output_dir
from pybounds import colorline


# ─────────────────────────────── configuration ──────────────────────────────
SYSTEM = 'fly'                              # 'fly' | 'fly7' (needs a ζ state)
ZETA = 0.0                                  # the ONE wind direction [rad]
ZLABEL = '0'                                # its label for titles
CHORD_LEN = 0.30                            # |B − A| [m], shared by every row
CHORD_DIR_DEG = 0.0                         # direction A→B [deg]; legs are
#                                             symmetric about it, so 0 makes the
#                                             heading sweep straddle the wind axis
BULGE_SIGN = +1                             # which side the path bulges / turn sense
# The apex height above the chord midpoint. Radius and apex are INDEPENDENT inputs,
# and together they fix the shape — so the choice of apex per row is what decides
# how the family nests spatially. Two families are run and compared:
#   'apex_pinned' — one apex for every row. Distance stays nearly constant (~7%),
#         but the shallow rows then bulge OUTSIDE the sharp ones (a broad arc has to
#         travel farther to reach the same apex).
#   'nested'      — apex shrinks with the radius, so the shallow rows are SHORTER
#         and sit entirely INSIDE / underneath the sharper ones. Distance then
#         varies on purpose (~22%): shallow = short & low, sharp = long & tall.
FAMILIES = {
    'apex_pinned': 0.13,                                    # scalar → same apex
    'nested': np.array([0.050, 0.065, 0.080, 0.095,         # array → one per row
                        0.110, 0.120, 0.130]),
}
# The sweep: one row per (radius, turn speed) pair — shallow & slow (top) → sharp &
# fast (bottom). The radius sets radial curvature; the turn speed sets how fast that
# curvature is flown, so ω_turn = TURN_SPEEDS/RADII ranges far more than either
# alone. Straight legs are flown at the COMMON LEG_SPEED in every row.
RADII = np.array([0.145, 0.0993, 0.0681, 0.0466, 0.0320, 0.0219, 0.0150])  # [m]
TURN_SPEEDS = np.array([0.15, 0.19, 0.23, 0.28, 0.34, 0.40, 0.46])        # [m/s]
LEG_SPEED = 0.30                            # [m/s] cruise on the ENTRY leg, all rows
BLEND = 0.02                                # [m] arc length to ramp leg→turn speed
EXIT_RAMP = False                           # False: the exit leg STAYS at the turn
#                                             speed — the end-of-trajectory velocity
#                                             is deliberately NOT pinned, so a fast
#                                             sharp row also exits fast. True would
#                                             ramp back to LEG_SPEED.
WIND_MODES = ('fixed', 'scaled')            # both are run, into subdirectories
WIND_FIXED = 0.4                            # [m/s] wind speed when mode='fixed'
WIND_OVER_SPEED = 0.4 / 0.3                 # w/g held at the turn when 'scaled'
# PER-STEP Q_d [rad²] — the Q of the discrete Eq.30/33 recursion, passed to the
# engine verbatim. (Was Q_ZETA_PSD, a continuous PSD that the column labels then
# multiplied by dt; the values below are those old labels, so every published
# column keeps its number and only the variable's meaning changed.)
Q_ZETA_D = np.array([1e-5, 1e-4, 1e-3, 1.5e-3, 2e-3, 3e-3,
                     4e-3, 5e-3, 1e-2, 1e-1, 1.0])
GRAMIAN_PANEL_QD = 1.0                      # Q_d shown in the traj Gramian panel
LAM = 1e-6                                  # Tikhonov regularizer (1/λ floor)
WINDOW = 4                                  # sliding window w — matches fly example
CMAP = 'inferno_r'                          # low EV (observable) = light
EV_VMIN, EV_VMAX = 1e-4, 1e6               # fixed color range for all EV maps
OUTDIR = output_dir('angular_velocity_sweep_output')


# ──────────────── the apex-pinned, curvature-swept geometry ─────────────────
def chord_points():
    """ Shared start A = (0,0) and end B (chord CHORD_LEN at CHORD_DIR_DEG). """
    a = np.radians(CHORD_DIR_DEG)
    A = np.zeros(2)
    return A, A + CHORD_LEN * np.array([np.cos(a), np.sin(a)])


def apex_heights(family):
    """ Per-row apex heights for a family (a scalar in FAMILIES is broadcast). """
    h = np.asarray(FAMILIES[family], dtype=float)
    return np.full(len(RADII), float(h)) if h.ndim == 0 else h


def raw_geometry(R, H):
    """ Leg/arc geometry for turn radius R and apex height H, in the CHORD FRAME (A
    at the origin, B at (CHORD_LEN, 0), bulge toward +y). The straight legs are the
    tangent lines from A and B to the circle of radius R whose topmost point is
    (CHORD_LEN/2, H); both legs are equal by symmetry. R and H are INDEPENDENT.

    Returns (theta, turn, leg, length):
        theta  — leg heading [rad] (leg1 = +θ, leg2 = −θ)
        turn   — net heading change = 2θ [rad]  (an OUTCOME of R and H)
        leg    — length of EACH straight leg [m]
        length — total path length 2·leg + R·turn [m]
    """
    hx = CHORD_LEN / 2.0
    dy = H - R
    d = np.hypot(hx, dy)
    if d <= R:
        raise ValueError(
            f'R={R:g} m is too large for chord {CHORD_LEN:g} m at apex '
            f'{H:g} m (the circle swallows the tangent point); need R < '
            f'{d:.4f} m.')
    leg = np.sqrt(d * d - R * R)
    theta = np.arctan2(dy, hx) + np.arcsin(R / d)
    if not (0.0 < theta < np.pi / 2):
        raise ValueError(f'R={R:g} m at apex {H:g} m: leg heading '
                         f'{np.degrees(theta):.1f}° does not progress toward B; '
                         'lower R or raise the apex.')
    turn = 2.0 * theta
    return theta, turn, leg, 2.0 * leg + R * turn


def row_geometry(R, H, leg_extra=0.0):
    """ Per-row geometry dict, with `leg_extra` added to the EXIT leg only (the MPC
    endpoint correction; see `simulate_trajectory`). """
    theta, turn, leg, _ = raw_geometry(R, H)
    leg2 = max(leg + leg_extra, 1e-4)
    return dict(R=R, H=H, theta=theta, turn=turn, leg1=leg, leg2=leg2,
                arc=R * turn, length=leg + leg2 + R * turn)


def path_point(gm, s):
    """ Position, heading and curvature at arc length s along the row's path,
    rotated into the world frame by CHORD_DIR_DEG and mirrored by BULGE_SIGN. """
    R, th, leg1, arc = gm['R'], gm['theta'], gm['leg1'], gm['arc']
    P1 = leg1 * np.array([np.cos(th), np.sin(th)])
    C = P1 + R * np.array([np.sin(th), -np.cos(th)])       # centre: 90° right of θ
    if s <= leg1:                                          # entry leg
        q, h, k = s * np.array([np.cos(th), np.sin(th)]), th, 0.0
    elif s <= leg1 + arc:                                  # the arc (right turn)
        h = th - (s - leg1) / R
        q, k = C + R * np.array([-np.sin(h), np.cos(h)]), 1.0 / R
    else:                                                  # exit leg
        P2 = C + R * np.array([-np.sin(-th), np.cos(-th)])
        h = -th
        q, k = P2 + (s - leg1 - arc) * np.array([np.cos(h), np.sin(h)]), 0.0
    sgn = BULGE_SIGN
    p = np.array([q[0], sgn * q[1]])
    h, k = sgn * h, -sgn * k
    a0 = np.radians(CHORD_DIR_DEG)
    if a0:
        c, sn = np.cos(a0), np.sin(a0)
        p = np.array([c * p[0] - sn * p[1], sn * p[0] + c * p[1]])
        h = h + a0
    return p, h, k


def speed_at(gm, s, v_turn):
    """ Speed profile along the path: the common LEG_SPEED on the ENTRY leg (so every
    row shares its approach), blended over BLEND into `v_turn` for the arc. The exit
    leg then STAYS at `v_turn` unless EXIT_RAMP — the end-of-trajectory velocity is
    deliberately not pinned, so a fast sharp row also exits fast. Speed is
    deliberately NOT constant along a trajectory: that is what lets ω = g·κ range
    more widely than curvature alone allows. """
    leg1, arc, leg2 = gm['leg1'], gm['arc'], gm['leg2']
    b1 = min(BLEND, 0.9 * leg1)
    if s < leg1 - b1:
        return LEG_SPEED
    if s < leg1:
        return LEG_SPEED + (v_turn - LEG_SPEED) * (s - (leg1 - b1)) / b1
    if s <= leg1 + arc or not EXIT_RAMP:
        return v_turn
    b2 = min(BLEND, 0.9 * leg2)
    if s <= leg1 + arc + b2:
        return v_turn + (LEG_SPEED - v_turn) * (s - (leg1 + arc)) / b2
    return LEG_SPEED


def build_row(spec, gm, v_turn):
    """ March along the path at the prescribed speed profile, emitting per-step
    (speed, heading) set-points. Path-first: the commanded track is EXACTLY the
    designed curve whatever the speed profile does. Returns
    (speed_sp, heading_sp, k_turn0, n_turn, commanded_path). """
    dt, L = spec.dt, gm['length']
    s = 0.0
    speeds, headings, pts, ss = [], [], [], []
    while True:
        sc = min(s, L)
        p, h, _ = path_point(gm, sc)
        v = speed_at(gm, sc, v_turn)
        speeds.append(v); headings.append(h); pts.append(p); ss.append(sc)
        if s >= L:
            break
        s += v * dt
        if len(speeds) > 20000:
            raise RuntimeError('march did not terminate — check the speed profile')
    ss = np.array(ss)
    in_turn = np.flatnonzero((ss >= gm['leg1']) & (ss <= gm['leg1'] + gm['arc']))
    k0 = int(in_turn[0]) if len(in_turn) else 0
    nt = int(len(in_turn)) if len(in_turn) else 1
    return np.array(speeds), np.array(headings), k0, nt, np.array(pts)


ENDPOINT_TOL = 0.0015                       # [m] flown-endpoint tolerance to B
ENDPOINT_ITERS = 6                          # max exit-leg correction iterations


def wind_of(v_turn, mode):
    """ Wind speed for this row: constant, or scaled to hold w/g fixed AT THE TURN
    (where the along-wind crossing — hence the ζ information — happens). """
    return WIND_FIXED if mode == 'fixed' else WIND_OVER_SPEED * v_turn


def dgamma_dzeta(v_turn, mode):
    """ ∂γ/∂ζ = −w/(g−w) at the along-wind heading (the apex), g = the turn speed.
    Diverges where the turn speed equals w; constant under 'scaled'. """
    w = wind_of(v_turn, mode)
    return np.inf if abs(v_turn - w) < 1e-12 else -w / (v_turn - w)


def simulate_trajectory(engine, spec, R, H, v_turn, mode, zeta, B):
    """ Build the row and run the MPC rollout in constant wind ζ. The MPC tracks
    heading/velocity (not position), so the flown endpoint falls short; we extend
    the EXIT leg by the shortfall (leaving the approach and the turn untouched) and
    re-simulate until the flown path ends on the shared B. Returns a facts dict. """
    chord_hat = B / np.linalg.norm(B)
    leg_extra, gm, err, k0, nt = 0.0, None, np.inf, 0, 1
    for _ in range(ENDPOINT_ITERS):
        gm = row_geometry(R, H, leg_extra)
        speed_sp, heading_sp, k0, nt, _ = build_row(spec, gm, v_turn)
        sp = spec.make_setpoint(speed_sp, heading_sp, wind_of(v_turn, mode), zeta)
        with contextlib.redirect_stdout(io.StringIO()):
            engine.simulate_mpc(sp)
        d = engine.X[-1, :2] - B
        err = float(np.hypot(*d))
        if err <= ENDPOINT_TOL:
            break
        leg_extra -= float(np.dot(d, chord_hat))    # extend/trim the exit leg
    peak = np.nan
    if 'phi_dot' in spec.state_names:
        peak = float(np.max(np.abs(engine.X[:, spec.state_names.index('phi_dot')])))
    P = engine.X[:, :2]
    return dict(gm=gm, k_turn0=k0, n_turn=nt, dend=err, peak=peak,
                flown_len=float(np.sum(np.hypot(*np.diff(P, axis=0).T))),
                apex=float(np.max(np.abs(P[:, 1]))),
                T_total=float(engine.N * spec.dt), omega_turn=v_turn / R)


def path_mismatch(paths):
    """ Max pairwise lateral deviation between rows [m]: x is monotone along these
    paths, so interpolate every row's y onto a common x grid and take the largest
    spread. Quantifies what the differing curvature costs spatially. """
    xs = [p[0] for p in paths]
    lo, hi = max(x.min() for x in xs), min(x.max() for x in xs)
    if hi <= lo:
        return np.nan
    grid = np.linspace(lo, hi, 400)
    ys = []
    for x, y, _ in paths:
        o = np.argsort(x)
        ys.append(np.interp(grid, x[o], y[o]))
    ys = np.array(ys)
    return float(np.max(ys.max(axis=0) - ys.min(axis=0)))


# ─────────────────────────────── EV helpers ─────────────────────────────────
def turn_window_mean(ev_df, dt, k_turn0, n_turn, w):
    """ Mean of min-EV(ζ) over the sliding windows that OVERLAP the arc (window of
    length w starting at k covers [k, k+w]; the arc occupies steps
    [k_turn0, k_turn0+n_turn]). Captures how observable ζ is THROUGH the turn,
    without the straight legs (heading fixed, ζ pinned near 1/λ). """
    ti = ev_df['time_initial'].values
    finite = np.isfinite(ti)
    k = np.full(ti.shape, -1, dtype=int)
    k[finite] = np.rint(ti[finite] / dt).astype(int)
    lo, hi = k_turn0 - w + 1, k_turn0 + n_turn - 1
    sel = finite & (k >= lo) & (k <= hi)
    vals = ev_df['zeta'].values[sel]
    return np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan


def ev_along_path(engine, ev_df, state):
    """ Map a pipeline's min-EV of `state` onto the trajectory points
    (window-start alignment, edges padded) — same as the GUI. """
    ev_path = np.full(engine.N, np.nan)
    vals = ev_df[state].values
    ti = ev_df['time_initial'].values
    ok = np.isfinite(ti) & np.isfinite(vals)
    ev_path[np.rint(ti[ok] / engine.dt).astype(int)] = vals[ok]
    fin = np.flatnonzero(np.isfinite(ev_path))
    if len(fin):
        ev_path[:fin[0]] = ev_path[fin[0]]
        ev_path[fin[-1]:] = ev_path[fin[-1]]
    return ev_path


def draw_wind_glyph(ax, engine, q_zeta_d, w, show_sigma):
    """ Corner glyph: wind arrow (direction ζ, speed) with a ±1σ(ζ) drift fan
    (σ = √(Q_d·n_w), n_w = window in STEPS; drawn only when show_sigma). """
    s = engine.spec
    if 'w' not in s.state_names or 'zeta' not in s.state_names:
        return
    X = engine.X
    w_wind = X[0, s.state_names.index('w')]
    zeta = X[0, s.state_names.index('zeta')]
    n_w = min(w, max(engine.N - 1, 1))
    half = (min(np.sqrt(max(q_zeta_d, 0.0) * n_w), np.pi)
            if show_sigma else 0.0)
    cx, cy, r = 0.13, 0.88, 0.09
    tf = ax.transAxes

    def arrow(angle, color, width):
        ax.add_patch(mpatches.FancyArrow(
            cx - 0.85 * r * np.cos(angle), cy - 0.85 * r * np.sin(angle),
            1.7 * r * np.cos(angle), 1.7 * r * np.sin(angle), transform=tf,
            width=width, color=color, length_includes_head=True,
            zorder=6, clip_on=False))

    if half > 0.01:
        ax.add_patch(mpatches.Wedge(
            (cx, cy), r, np.degrees(zeta - half), np.degrees(zeta + half),
            transform=tf, facecolor='steelblue', alpha=0.22, lw=0,
            zorder=5, clip_on=False))
    arrow(zeta, 'steelblue', 0.012)
    label = f'wind {w_wind:.2g} m/s, ζ = {zeta:.2f} rad'
    if half > 0.01:
        label += f'\n±1σ(ζ) = {half:.2f} rad'
    ax.text(cx + r + 0.03, cy, label, transform=tf, ha='left', va='center',
            fontsize=6.5, color='steelblue', zorder=6, clip_on=False,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none',
                      alpha=0.6))


def draw_colored_traj(fig, ax, engine, ev_path, norm, title, B,
                      q_zeta_d, w, show_sigma):
    X = engine.X
    ax.plot(X[:, 0], X[:, 1], 'k-', lw=0.3, alpha=0.4)
    with contextlib.redirect_stdout(io.StringIO()):        # colorline prints
        lc = colorline(X[:, 0], X[:, 1], ev_path, ax=ax,
                       cmap=plt.get_cmap(CMAP), norm=norm, linewidth=4)
    cb = fig.colorbar(lc, ax=ax, pad=0.02)
    cb.set_label('min EV: ζ', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.plot(X[0, 0], X[0, 1], 'go', ms=7, zorder=4, label='start (shared)')
    ax.plot(B[0], B[1], 'r*', ms=12, zorder=4, label='end (shared)')
    if len(X) >= 3:
        d = X[-1, :2] - X[-3, :2]
        nrm = float(np.hypot(*d))
        if nrm > 1e-9:
            d = d / nrm
            span = max(np.ptp(X[:, 0]), np.ptp(X[:, 1]), 1e-9)
            tail = X[-1, :2] - d * 0.07 * span
            ax.annotate('', xy=(X[-1, 0], X[-1, 1]), xytext=(tail[0], tail[1]),
                        arrowprops=dict(arrowstyle='-|>', color='k', lw=1.8),
                        zorder=7)
    draw_wind_glyph(ax, engine, q_zeta_d, w, show_sigma)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)


def row_label(info, v_turn, sep='\n'):
    """ Row / figure label: sharpness, the rate it is flown at, and the timing. """
    gm = info['gm']
    return (f'R = {gm["R"]:.4f} m, ω = {info["omega_turn"]:.1f} rad/s{sep}'
            f'turn {np.degrees(gm["turn"]):.0f}°, v_turn = {v_turn:g} m/s, '
            f'T = {info["T_total"]:.2f} s, L = {info["flown_len"]:.3f} m')


def save_trajectory_figure(engine, info, v_turn, mode, w, ev_pb, ev_grams, B,
                           outdir):
    """ Per-row figure: colored paths (pybounds Q=0 vs Gramian at
    Q_d = GRAMIAN_PANEL_QD) + min-EV(ζ) time-series (pybounds vs Eq. 33 at each Q)
    + the ζ(t) state. """
    dt = engine.dt
    gi = int(np.argmin(np.abs(Q_ZETA_D - GRAMIAN_PANEL_QD)))
    qd_panel = Q_ZETA_D[gi]
    path_pb = ev_along_path(engine, ev_pb, 'zeta')
    path_gr = ev_along_path(engine, ev_grams[gi], 'zeta')
    norm = mpl.colors.LogNorm(EV_VMIN, EV_VMAX)
    lab = row_label(info, v_turn, sep='  ')

    fig = plt.figure(figsize=(11, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.3, 1.25, 0.75], hspace=0.35)
    ax_pb = fig.add_subplot(gs[0, 0])
    ax_gr = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, :])
    ax_z = fig.add_subplot(gs[2, :], sharex=ax_t)

    draw_colored_traj(fig, ax_pb, engine, path_pb, norm,
                      'min EV of ζ — pybounds empirical (Q = 0)', B,
                      0.0, w, show_sigma=False)
    draw_colored_traj(fig, ax_gr, engine, path_gr, norm,
                      f'min EV of ζ — Gramian Eq.(33), Q = {qd_panel:g}', B,
                      Q_ZETA_D[gi], w, show_sigma=True)

    tsh = 0.5 * w * dt
    ax_t.semilogy(ev_pb['time'], ev_pb['zeta'], 'k--', lw=2.2,
                  label='pybounds (Q = 0)')
    cmap = plt.get_cmap('viridis')
    for j, (q, ev) in enumerate(zip(Q_ZETA_D, ev_grams)):
        c = cmap(j / max(len(Q_ZETA_D) - 1, 1))
        lw = 2.4 if j == gi else 1.4
        ax_t.semilogy(ev['time'], ev['zeta'], '-', color=c, lw=lw,
                      label=f'Q = {q:g}')
    ax_t.set_ylim(EV_VMIN, EV_VMAX)
    ax_t.set_ylabel('min error variance of ζ')
    ax_t.set_title(f'ζ min error variance vs time — {lab}, w = {w}  '
                   '(legend: Q, rad²)', fontsize=9)
    ax_t.legend(fontsize=6, ncol=4)
    ax_t.grid(alpha=0.3, which='both')
    ax_t.tick_params(labelbottom=False)

    zi = engine.spec.state_names.index('zeta')
    zw = (engine.X[:, zi] + np.pi) % (2 * np.pi) - np.pi
    brk = np.flatnonzero(np.abs(np.diff(zw)) > np.pi) + 1
    ax_z.plot(np.insert(engine.t.astype(float), brk, np.nan),
              np.insert(zw, brk, np.nan), color='steelblue', lw=1.8)
    ax_z.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax_z.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax_z.set_yticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
    ax_z.set_ylabel('ζ (rad)')
    ax_z.set_xlabel('time (s)')
    ax_z.grid(alpha=0.3)

    fig.suptitle(f'ζ observability — {lab}   [wind {mode}]', fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir,
                       f'traj_R{int(round(info["gm"]["R"] * 1e4)):04d}e-4m.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────── summary matrix ─────────────────────────────
def save_matrix_figure(matrix, pb_col, traj_paths, spec, family, mode, infos,
                       outdir, metric):
    """ Heatmap: rows = sharpness (shallow → sharp, with a trajectory thumbnail),
    cols = [pybounds Q=0] + Q values, cell = `metric` of min-EV(ζ). PNG + CSV. """
    dt = spec.dt
    full = np.column_stack([pb_col, matrix])
    col_labels = ['pybounds\n(Q=0)'] + [f'{q:g}' for q in Q_ZETA_D]
    row_labels = [f'R={i["gm"]["R"]*1e3:.0f}mm\nω={i["omega_turn"]:.1f}, '
                  f'T={i["T_total"]:.2f}s' for i in infos]
    n_rows, n_cols = full.shape
    norm = mpl.colors.LogNorm(EV_VMIN, EV_VMAX)
    mlabel = ('best (min over trajectory)' if metric == 'best'
              else 'mean over the turn')
    lens = [i['flown_len'] for i in infos]
    wtxt = (f'wind FIXED at {WIND_FIXED:g} m/s (w/g varies at the turn — ∂γ/∂ζ '
            f'pole where v_turn = {WIND_FIXED:g})' if mode == 'fixed' else
            f'wind SCALED, w/g = {WIND_OVER_SPEED:.2f} at the turn (∂γ/∂ζ '
            'identical every row)')

    fig = plt.figure(figsize=(1.05 * n_cols + 3.6, 0.85 * n_rows + 2.8))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 8.0], wspace=0.03)

    tgs = outer[0].subgridspec(n_rows, 1, hspace=0.0)
    for k in range(n_rows):
        tax = fig.add_subplot(tgs[k])
        x, y, evp = traj_paths[k]
        tax.plot(x, y, 'k-', lw=0.4, alpha=0.3)
        with contextlib.redirect_stdout(io.StringIO()):
            colorline(x, y, evp, ax=tax, cmap=plt.get_cmap(CMAP), norm=norm,
                      linewidth=2.2)
        tax.plot(x[0], y[0], 'go', ms=3)
        tax.plot(x[-1], y[-1], 'r*', ms=5)
        tax.set_aspect('equal')
        tax.axis('off')

    ax = fig.add_subplot(outer[1])
    im = ax.imshow(full, cmap=CMAP, norm=norm, aspect='auto')
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_xlabel('ζ process noise  Q  (rad²)')
    # the thumbnail strip occupies the space an ylabel would use — put it further
    # out as figure text so it cannot overlap the thumbnails
    fig.text(0.0, 0.5, 'turn radius  R  (ω_turn, T_total)\n'
                       'shallow & slow → sharp & fast (top → bottom)',
             rotation=90, ha='right', va='center', fontsize=10)
    ax.axvline(0.5, color='deepskyblue', lw=2.5)     # pybounds | Gramian divider
    for r in range(n_rows):
        for c in range(n_cols):
            v = full[r, c]
            if np.isfinite(v):
                shade = norm(np.clip(v, EV_VMIN, EV_VMAX))
                ax.text(c, r, f'{v:.0e}', ha='center', va='center', fontsize=6.5,
                        color='white' if shade > 0.5 else 'black')
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(f'min error variance of ζ  ({mlabel})')
    nested = all(lens[k] <= lens[k + 1] for k in range(len(lens) - 1))
    ax.set_title(f'ζ error variance vs TURN SHARPNESS & process noise  —  family '
                 f'"{family}"  (wind ζ = {ZLABEL})\n'
                 f'cell = {mlabel} min-EV(ζ)  —  lower = more observable;  radial '
                 f'curvature varies {RADII[0] / RADII[-1]:.1f}×, '
                 f'L = {min(lens):.3f}–{max(lens):.3f} m '
                 f'({"shallow nested INSIDE sharp" if nested else "shallow OUTSIDE sharp"})'
                 f'\nsharper rows are flown faster through the turn (ω = v_turn/R); '
                 f'entry leg cruises at {LEG_SPEED:g} m/s in every row, exit speed '
                 f'not pinned  —  {wtxt}', fontsize=9.5)
    tag = 'best' if metric == 'best' else 'mean'
    out_png = os.path.join(outdir, f'matrix_{tag}_min_ev_zeta.png')
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)

    out_csv = os.path.join(outdir, f'matrix_{tag}_min_ev_zeta.csv')
    csv_cols = ['pybounds_Q0'] + [f'Q={q:g}' for q in Q_ZETA_D]
    with open(out_csv, 'w') as f:
        f.write(f'# cell = {mlabel} min-EV of zeta; Q in rad^2; family="{family}"; '
                f'every row shares start and end; the swept variable is the turn '
                f'radius R (radial curvature) with its own apex, flown at turn '
                f'speed v_turn so omega=v_turn/R; entry leg cruises at '
                f'{LEG_SPEED:g} m/s and the exit speed is not pinned '
                f'(EXIT_RAMP={EXIT_RAMP}); wind mode = {mode}\n')
        f.write('radius_m,apex_cmd_m,turn_speed_m_per_s,omega_turn_rad_per_s,'
                'turn_deg,apex_flown_m,flown_len_m,total_time_s,wind_m_per_s,'
                'wind_over_turnspeed,dgamma_dzeta,peak_phidot_rad_per_s,'
                + ','.join(csv_cols) + '\n')
        for r, (i, v) in enumerate(zip(infos, TURN_SPEEDS)):
            gm = i['gm']
            f.write(f'{gm["R"]:g},{gm["H"]:g},{v:g},{i["omega_turn"]:.4f},'
                    f'{np.degrees(gm["turn"]):.2f},{i["apex"]:.5f},'
                    f'{i["flown_len"]:.5f},{i["T_total"]:.4f},'
                    f'{wind_of(v, mode):.4f},{wind_of(v, mode) / v:.4f},'
                    f'{dgamma_dzeta(v, mode):.4f},{i["peak"]:.3f},' +
                    ','.join(f'{full[r, c]:.6e}' for c in range(n_cols)) + '\n')
    return out_png, out_csv


def draw_family(ax, paths, infos, B, mismatch, title, fontsize=9):
    """ Draw one family overlaid on `ax` — different radial curvature, shared start
    and end. Whether the shallow rows sit inside or outside the sharp ones is set by
    how the apex was chosen per row. """
    cmap = plt.get_cmap('viridis')
    for i, (x, y, _) in enumerate(paths):
        gm = infos[i]['gm']
        ax.plot(x, y, '-', lw=1.8, color=cmap(i / max(len(paths) - 1, 1)),
                label=f'R={gm["R"]*1e3:.0f}mm  (ω={infos[i]["omega_turn"]:.1f}, '
                      f'{np.degrees(gm["turn"]):.0f}°, '
                      f'L={infos[i]["flown_len"]:.3f}m, '
                      f'T={infos[i]["T_total"]:.2f}s)')
    ax.plot(paths[0][0][0], paths[0][1][0], 'go', ms=10, label='start (shared)')
    ax.plot(B[0], B[1], 'r*', ms=15, label='end (shared)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.legend(fontsize=6.5, title='turn radius R', title_fontsize=7.5)
    ax.set_title(title, fontsize=fontsize)


def family_title(infos, family, mode, mismatch):
    lens = [i['flown_len'] for i in infos]
    spread = 100 * (max(lens) - min(lens)) / float(np.mean(lens))
    nested = all(lens[k] <= lens[k + 1] for k in range(len(lens) - 1))
    apex = [i['gm']['H'] for i in infos]
    return (f'family "{family}" — radial curvature {RADII[0] / RADII[-1]:.1f}×, '
            f'apex {min(apex):.3f}–{max(apex):.3f} m\n'
            f'{"shallow rows NESTED INSIDE the sharper ones" if nested else "shallow rows OUTSIDE the sharper ones"}'
            f': distance {min(lens):.3f}–{max(lens):.3f} m ({spread:.1f}% spread)\n'
            f'flown FASTER as it sharpens (ω = v_turn/R), exit speed not pinned; '
            f'max lateral dev {mismatch * 1e3:.0f} mm '
            f'({100 * mismatch / CHORD_LEN:.0f}% of chord)  [wind {mode}]')


def save_family_figure(paths, infos, family, mode, B, mismatch, outdir):
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    draw_family(ax, paths, infos, B, mismatch,
                family_title(infos, family, mode, mismatch))
    fig.tight_layout()
    out = os.path.join(outdir, 'family.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


def save_families_compare(collected, mode, B, outdir):
    """ Side-by-side comparison of the families for one wind mode, on SHARED axes —
    this is the figure that shows the 'nested' family's shallow turns sitting
    underneath the sharper ones, while 'apex_pinned' has them outside. """
    names = list(collected.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(8.4 * len(names), 7.0))
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(wspace=0.22)          # keep the panel titles from colliding
    xs = [p[0] for fam in names for p in collected[fam][0]]
    ys = [p[1] for fam in names for p in collected[fam][0]]
    xlim = (min(x.min() for x in xs) - 0.01, max(x.max() for x in xs) + 0.01)
    ylim = (min(y.min() for y in ys) - 0.01, max(y.max() for y in ys) + 0.01)
    for ax, fam in zip(axes, names):
        paths, infos, mism = collected[fam]
        draw_family(ax, paths, infos, B, mism,
                    family_title(infos, fam, mode, mism), fontsize=8)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.suptitle('Trajectory families compared on shared axes — same start & end, '
                 f'radial curvature {RADII[0] / RADII[-1]:.1f}×  '
                 f'(wind ζ = {ZLABEL}, wind {mode})', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(outdir, f'families_compare_wind_{mode}.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────── one wind mode ──────────────────────────────
def run_mode(engine, spec, family, mode, outdir):
    w = WINDOW
    eps = spec.eps_default
    r_diag = dict(spec.r_default)
    sensors = tuple(spec.fim_sensors)
    states = tuple(spec.fim_states)
    if 'zeta' not in states:
        states = states + ('zeta',)
    q_floor = dict(spec.q_tiny)
    os.makedirs(outdir, exist_ok=True)

    A, B = chord_points()
    apex = apex_heights(family)
    print(f'\n{"=" * 78}\nFAMILY = {family}   WIND MODE = {mode}   → {outdir}\n'
          f'{"=" * 78}\n'
          f'chord {CHORD_LEN:g} m at {CHORD_DIR_DEG:g}°, apex {list(np.round(apex, 4))} m, '
          f'A=({A[0]:.3f},{A[1]:.3f}) → B=({B[0]:.3f},{B[1]:.3f})\n'
          f'entry leg cruises at {LEG_SPEED:g} m/s (all rows); turn speeds '
          f'{list(TURN_SPEEDS)} m/s; exit leg '
          f'{"ramps back to cruise" if EXIT_RAMP else "STAYS at the turn speed"}\n'
          f'radii {list(RADII)} m  →  ω_turn='
          f'{[round(v / R, 2) for R, v in zip(RADII, TURN_SPEEDS)]} rad/s\n'
          f'wind={[round(wind_of(v, mode), 3) for v in TURN_SPEEDS]} m/s,  '
          f'∂γ/∂ζ={[round(dgamma_dzeta(v, mode), 2) for v in TURN_SPEEDS]}\n',
          flush=True)

    n_rows, n_q = len(RADII), len(Q_ZETA_D)
    mat_best = np.full((n_rows, n_q), np.nan)
    mat_mean = np.full((n_rows, n_q), np.nan)
    pb_best = np.full(n_rows, np.nan)
    pb_mean = np.full(n_rows, np.nan)
    infos, traj_paths = [], []

    for i, (R, H, v_turn) in enumerate(zip(RADII, apex, TURN_SPEEDS)):
        print(f'  R={R * 1e3:5.1f} mm, apex={H:.3f} m, v_turn={v_turn:.2f} m/s '
              f'(ω={v_turn / R:5.1f} rad/s) — MPC …', flush=True)
        info = simulate_trajectory(engine, spec, R, H, v_turn, mode, ZETA, B)
        infos.append(info)
        gm = info['gm']
        end = engine.X[-1, :2]
        print(f'      turn={np.degrees(gm["turn"]):5.1f}° legs={gm["leg1"]:.4f}/'
              f'{gm["leg2"]:.4f} arc={gm["arc"]:.4f} apex={info["apex"]:.4f} '
              f'len={info["flown_len"]:.4f} N={engine.N} '
              f'T={info["T_total"]:.2f}s peak|φ̇|={info["peak"]:.1f} '
              f'end=({end[0]:.3f},{end[1]:.3f}) Δend={info["dend"]:.4f}',
              flush=True)

        with contextlib.redirect_stdout(io.StringIO()):
            ev_pb, _ = engine.empirical_ev(w, eps, r_diag, LAM,
                                           sensors=sensors, states=states)
        pb_best[i] = np.nanmin(ev_pb['zeta'].values)
        pb_mean[i] = turn_window_mean(ev_pb, spec.dt, info['k_turn0'],
                                      info['n_turn'], w)
        traj_paths.append((engine.X[:, 0].copy(), engine.X[:, 1].copy(),
                           ev_along_path(engine, ev_pb, 'zeta')))

        ev_grams = []
        for j, q in enumerate(Q_ZETA_D):
            q_diag = dict(q_floor)
            q_diag['zeta'] = float(q)
            try:
                ev = engine.linearized_ev(w, q_diag, r_diag, LAM,
                                          sensors=sensors, states=states)
                ev_grams.append(ev)
                mat_best[i, j] = np.nanmin(ev['zeta'].values)
                mat_mean[i, j] = turn_window_mean(ev, spec.dt, info['k_turn0'],
                                                  info['n_turn'], w)
            except np.linalg.LinAlgError:
                ev_grams.append(ev_pb.copy())
        save_trajectory_figure(engine, info, v_turn, mode, w, ev_pb, ev_grams, B,
                               outdir)
        print(f'        best min-EV(ζ): pybounds={pb_best[i]:.2e}  '
              f'Gramian[{Q_ZETA_D[0]:g}..{Q_ZETA_D[-1]:g}]='
              f'{mat_best[i, 0]:.2e}..{mat_best[i, -1]:.2e}', flush=True)

    mism = path_mismatch(traj_paths)
    lens = [i['flown_len'] for i in infos]
    spread = 100 * (max(lens) - min(lens)) / float(np.mean(lens))
    # does the family nest — is every shallower row strictly inside the sharper?
    nested = all(lens[k] <= lens[k + 1] for k in range(len(lens) - 1))
    print(f'  → radial curvature range {RADII[0] / RADII[-1]:.1f}×;  distance '
          f'{min(lens):.4f}..{max(lens):.4f} m ({spread:.1f}% spread, '
          f'{"shallow INSIDE sharp — nested" if nested else "shallow OUTSIDE sharp"}'
          f');  max lateral deviation {mism * 1e3:.1f} mm '
          f'({100 * mism / CHORD_LEN:.1f}% of the chord)', flush=True)
    save_family_figure(traj_paths, infos, family, mode, B, mism, outdir)
    save_matrix_figure(mat_best, pb_best, traj_paths, spec, family, mode, infos,
                       outdir, 'best')
    save_matrix_figure(mat_mean, pb_mean, traj_paths, spec, family, mode, infos,
                       outdir, 'mean')
    return traj_paths, infos


# ─────────────────────────────── main sweep ─────────────────────────────────
def main():
    if len(RADII) != len(TURN_SPEEDS):
        raise SystemExit('RADII and TURN_SPEEDS must have the same length')
    spec = SYSTEMS[SYSTEM]()
    if 'zeta' not in spec.state_names:
        raise SystemExit(f"system '{SYSTEM}' has no ζ state; use fly/fly7")
    engine = ObservabilityEngine(spec)
    os.makedirs(OUTDIR, exist_ok=True)
    print(f'system={SYSTEM}  wind ζ={ZLABEL} ({np.degrees(ZETA):.0f}°)  '
          f'w={WINDOW}  eps={spec.eps_default:g}  lam={LAM:g}  '
          f'sensors={tuple(spec.fim_sensors)}\nQ_zeta(per-step Q_d)={list(Q_ZETA_D)}',
          flush=True)
    for mode in WIND_MODES:
        collected = {}
        for family in FAMILIES:
            paths, infos = run_mode(engine, spec, family, mode,
                                    os.path.join(OUTDIR, family, f'wind_{mode}'))
            collected[family] = (paths, infos, path_mismatch(paths))
        print(f'  → {os.path.basename(save_families_compare(collected, mode, B_OF(), OUTDIR))}',
              flush=True)
    print(f'\noutputs under: {OUTDIR}/{{{",".join(FAMILIES)}}}/'
          f'wind_{{{",".join(WIND_MODES)}}}  + families_compare_wind_*.png')


def B_OF():
    return chord_points()[1]


if __name__ == '__main__':
    main()
